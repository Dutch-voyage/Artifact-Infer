
import torch
import math
import itertools

from src.core.artifact_base import Artifact
from src.core.service_base import AsyncBaseService

from src.artifacts.swift.block_manager import BlockManagerArtifact, BlockManagerArgs

from .layers.pre_layer import LlamaPreLayer
from .layers.transformer_layer import LlamaTransformerLayer
from .layers.post_layer import LlamaPostLayer
from src.core.utils import GB
from src.artifacts.swift.model_runner.utils import load_weights, dummy_load_weights, LlamaModelConfig, LlamaInferState
from src.artifacts.swift.model_runner.args import ModelRunnerArgs
from src.artifacts.swift.structs import SwiftRequest
import sys
sys.path.append('/home/yyx/efficient_inference/xylo-infer/csrc')
import swiftllm_c


class ModelRunnerArtifact(Artifact):
    def __init__(self, args: ModelRunnerArgs):
        super().__init__()
        self.args = args
        
        self.model_config = LlamaModelConfig.load_from_model_path(args.model_path)
        
        self.weight = None
        self._cos_cached = self._sin_cached = None
        
        self.pre_layer = None
        self.transfomer_layers = None
        self.post_layer = None
        
        self.num_blocks = None
        self.k_cache = self.v_cache = None
        self.k_swap = self.v_swap = None
        
        self.cpu_block_manager = self.gpu_block_manager = None
    
    def register(self, service: AsyncBaseService):
        methods_to_register = ["forward", "profile_num_blocks", "load_weights", "dummy_load_weights", "init_kvcache_and_swap", "dummy_init_kvcache_and_swap", "swap_out_seqs", "swap_in_seqs", "free_seqs_resources"]
        for method in methods_to_register:
            self._register_method(method, service)
    
    @property
    def name(self):
        return "swift_modelrunner"
    
    @torch.inference_mode()
    def dummy_load_weights(self):
        self.weight = dummy_load_weights(
            self.model_config,
            torch.float16,
            self.args.model_path,
            self.args.use_dummy
        )
        
        self._init_to_get_rotary()
        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.args,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)
    
    
    @torch.inference_mode()
    def load_weights(self):
        """
        Load weights and initialize layers
        """
        # Load weights
        self.weight = load_weights(
            self.model_config,
            torch.float16,
            self.args.model_path,
            self.args.use_dummy
        )

        # Initialize rotary embeddings
        self._init_to_get_rotary()

        # Initialize layers
        decoding_piggyback_stream = torch.cuda.Stream()
        self.pre_layer = LlamaPreLayer(self.model_config, self.weight)
        self.transformer_layers = [
            LlamaTransformerLayer(
                self.model_config,
                self.args,
                self.weight.layers[layer_id],
                decoding_piggyback_stream,
                layer_id
            )
            for layer_id in range(self.model_config.num_layers)
        ]
        self.post_layer = LlamaPostLayer(self.model_config, self.weight)

    @torch.inference_mode()
    def profile_num_blocks(self) -> int:
        """
        Profiler the number of GPU blocks

        We run a forged prefill batch with the maximum number of tokens and
        sequences, record the peak memory usage, and infer the number of blocks
        that can be allocated.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Synthesis a prefill batch
        num_tokens = self.args.max_tokens_in_batch
        batch_size = self.args.max_batch_size
        input_lens = [num_tokens // batch_size] * batch_size
        input_lens[-1] += num_tokens % batch_size
        
        # Create dummy SwiftRequest objects for profiling
        dummy_requests = []
        for i in range(batch_size):
            req = SwiftRequest({"output_len": 1})  # Minimal request for profiling
            req.prompt_token_ids = [0] * input_lens[i]  # Dummy tokens
            req.prompt_len = input_lens[i]
            req.request_id = i
            dummy_requests.append(req)
        
        self.k_cache = self.v_cache = None # pylint: disable=attribute-defined-outside-init
        _ = self.forward(dummy_requests, ignore_kvcache=True)
        torch.cuda.synchronize()

        # peak_memory = torch.cuda.max_memory_allocated()
        # total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        peak_memory = total_memory - free_memory
        useable_memory = total_memory*self.args.gpu_mem_utilization
        print(f"[Model.profile] GPU total memory: {total_memory/GB:.2f} GB, runtime peak memory: {peak_memory/GB:.2f} GB")
        if useable_memory < peak_memory:
            raise RuntimeError(f"Peak memory {peak_memory/GB:.2f} GB exceeds usable memory {useable_memory/GB:.2f} GB ({total_memory/GB:.2f} GB * {self.args.gpu_mem_utilization})")
        block_size_bytes = self.args.block_size * self.model_config.get_kvslot_size()
        num_gpu_blocks = math.floor((useable_memory - peak_memory) / block_size_bytes)

        torch.cuda.empty_cache()
        return num_gpu_blocks
    
    @torch.inference_mode()
    def dummy_init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks
        
        kvcache_shape = (
            self.num_blocks, 
            self.model_config.num_layers,
            self.model_config.num_kv_heads, 
            self.args.block_size, 
            self.model_config.head_dim
        )
        
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="meta")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="meta")
        
        kvswap_shape = (
            self.args.num_cpu_blocks, 
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.args.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="meta")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="meta")
    
    @torch.inference_mode()
    def init_kvcache_and_swap(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Initialize KV cache
        kvcache_shape = (
            self.num_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.args.block_size,
            self.model_config.head_dim
        )
        # Here we use torch.zeros instead of torch.empty, since that torch.empty
        # has the possibility to contain NaNs, which will cause the model to output NaNs.
        self.k_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")
        self.v_cache = torch.zeros(kvcache_shape, dtype=torch.float16, device="cuda")

        # Initialize KV swap space
        kvswap_shape = (
            self.args.num_cpu_blocks,
            self.model_config.num_layers,
            self.model_config.num_kv_heads,
            self.args.block_size,
            self.model_config.head_dim
        )
        self.k_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
        self.v_swap = torch.zeros(kvswap_shape, dtype=torch.float16, device="cpu")
    
    def _init_to_get_rotary(self):
        rope_scaling = self.model_config.rope_scaling
        base = self.model_config.rope_theta
        max_position_embeddings = self.model_config.max_position_embeddings

        # Handle the case where rope_scaling is a dictionary (Llama 3.2)
        if isinstance(rope_scaling, dict):
            scaling_factor = rope_scaling.get('factor', 4.0)
            low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
            high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
            rope_type = rope_scaling.get('rope_type', 'llama3')
            original_max_position_embeddings = rope_scaling.get('original_max_position_embeddings', max_position_embeddings)

            # Calculate maximum sequence length based on scaling factor
            max_seq_len = int(original_max_position_embeddings * scaling_factor)

            # Generate position indices
            dim = self.model_config.head_dim
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32)

            # Create frequency array with dimensions split between low and high frequency parts
            dim_half = dim // 2
            split_point = int(dim_half * low_freq_factor / (low_freq_factor + high_freq_factor))

            # Apply different scaling factors to different parts of the frequency spectrum
            inv_freq_low = 1.0 / (base ** (torch.arange(0, split_point * 2, 2, device="cuda", dtype=torch.float32) / dim))
            inv_freq_high = 1.0 / (base ** (torch.arange(split_point * 2, dim, 2, device="cuda", dtype=torch.float32) / dim))

            # Apply scaling factors
            low_positions = t / low_freq_factor
            high_positions = t / high_freq_factor

            # Calculate frequencies for both parts
            freqs_low = torch.outer(low_positions, inv_freq_low)
            freqs_high = torch.outer(high_positions, inv_freq_high)

            # Combine frequencies
            freqs = torch.cat([freqs_low, freqs_high], dim=-1)
        else:
            # Original implementation for scalar rope_scaling
            rope_scaling_factor = rope_scaling
            max_seq_len = max_position_embeddings * rope_scaling_factor

            inv_freq = 1.0 / (base ** (torch.arange(0, self.model_config.head_dim, 2, device="cuda", dtype=torch.float32) / self.model_config.head_dim))
            t = torch.arange(max_seq_len + 128, device="cuda", dtype=torch.float32) / rope_scaling_factor
            freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16)
        self._sin_cached = torch.sin(freqs).to(torch.float16)

    @torch.inference_mode()
    def _forward(
        self,
        input_ids: torch.Tensor,    # [total_token_num]
        infer_state: LlamaInferState,
    ) -> torch.Tensor:
        """
        Run a forward pass of the LlamaModel.
        """
        input_embds = self.pre_layer.forward(input_ids)
        residual_buf = torch.zeros_like(input_embds)
        for layer in self.transformer_layers:
            input_embds = layer.forward(
                input_embds,
                residual_buf,
                self.k_cache,
                self.v_cache,
                self.gpu_block_manager.block_table if not infer_state.ignore_kvcache else None,
                infer_state,
            )
        input_embds += residual_buf
        output_tokens = self.post_layer.forward(input_embds, infer_state)
        # breakpoint()
        return output_tokens
    
    @torch.inference_mode()
    def forward(
        self,
        cur_batch: list[SwiftRequest], 
        ignore_kvcache: bool = False,   # Skip actions related to kv cache, useful when profiling the number of kv blocks
    ) -> list[int]:
        """
        Run a forward pass of the LlamaModel.

        This function is a wrapper of the `_forward` function. It prepares the infer_state
        and calls the `_forward` function.

        This function is intended to be called by the server.
        """
        
        input_ids_list = [
            req.prompt_token_ids if req.is_prefill_stage() else [req.output_token_ids[-1]]
            for req in cur_batch
        ]
        
        seq_ids_list = [req.request_id for req in cur_batch]
        decoding_seq_lens_list = [
            req.prompt_len + req.get_cur_output_len()
            for req in cur_batch
            if not req.is_prefill_stage()
        ]

        num_prefill_seqs = len(input_ids_list) - len(decoding_seq_lens_list)
        flattened_input_ids = list(itertools.chain(*input_ids_list))
        seq_lengths_list = [len(seq) for seq in input_ids_list[:num_prefill_seqs]] + decoding_seq_lens_list

        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = torch.tensor(seq_lengths_list, dtype=torch.int32, device="cuda")

        batch_size = len(input_ids_list)
        num_tokens = len(flattened_input_ids)

        prefill_seq_lens_list = seq_lengths_list[:num_prefill_seqs]
        prefill_seq_lens = torch.tensor(prefill_seq_lens_list, dtype=torch.int32, device="cuda")
        prefill_start_locs = torch.cumsum(prefill_seq_lens, dim=0, dtype=torch.int32) - prefill_seq_lens
        max_prefill_len = max(prefill_seq_lens_list) if prefill_seq_lens_list else 0

        decoding_seq_lens = torch.tensor(decoding_seq_lens_list, dtype=torch.int32, device="cuda")
        max_decoding_len = max(decoding_seq_lens_list) if decoding_seq_lens_list else 0

        position_indices = torch.cat((
            torch.concat([
                torch.arange(
                    0,
                    prefill_seq_len,
                    device="cuda",
                    dtype=torch.int32
                )
                for prefill_seq_len in prefill_seq_lens_list
            ]) if prefill_seq_lens_list else torch.empty(0, device="cuda", dtype=torch.int32),
            decoding_seq_lens - 1
        ), dim=0)

        if not ignore_kvcache:
            self.gpu_block_manager.allocate_blocks_for_seqs(
                seq_ids,
                seq_lengths
            )

        # Select the seq_block_size
        #
        # Here we use a simple heuristic:
        #
        # In paged attention phase 1, the grid shape is (num_decoding_seqs, num_kv_heads, cdiv(max_decoding_len, seq_block_size))
        # and among these blocks, num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) blocks are useful.
        # Thus we set seq_block_size to be the largest integer that satisfies
        #      num_kv_heads * sum(cdiv(decoding_seq_lens, seq_block_size)) >= 1024
        # to fully utilize the GPU. Here 1024 is a magic number (since most high-end
        # GPUs have ~128 SMs, so ~512 SMSPs. Since the decoding-stage attention
        # is mostly a memory-bound operation, I think 1024 is a reasonable number.)
        #
        # In practice, we use `decoding_seq_lens_sum/seq_block_size` to approximate
        # sum(cdiv(decoding_seq_lens, seq_block_size))

        seq_block_size = 2048
        decoding_seq_lens_sum = sum(decoding_seq_lens_list)
        while self.model_config.num_kv_heads*(decoding_seq_lens_sum/seq_block_size) < 1024 and seq_block_size//2 >= 64 and \
            max_decoding_len / (seq_block_size//2) <= 128:
            seq_block_size //= 2

        infer_state = LlamaInferState(
            batch_size = batch_size,
            num_tokens = num_tokens,

            seq_ids = seq_ids,
            softmax_scale = self.model_config.head_dim ** -0.5,

            num_prefill_seqs = num_prefill_seqs,
            num_prefill_tokens = num_tokens - (batch_size - num_prefill_seqs),
            prefill_seq_start_locs = prefill_start_locs,
            prefill_seq_start_locs_with_end = torch.cat([
                prefill_start_locs,
                torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            ]),
            prefill_seq_lens = prefill_seq_lens,
            max_prefill_len = max_prefill_len,

            num_decoding_seqs = batch_size - num_prefill_seqs,
            decoding_seq_lens = decoding_seq_lens,
            max_decoding_len = max_decoding_len,

            seq_block_size = seq_block_size,
            num_seq_blocks = (max_decoding_len + seq_block_size-1) // seq_block_size,

            position_cos = self._cos_cached[position_indices],
            position_sin = self._sin_cached[position_indices],

            ignore_kvcache = ignore_kvcache
        )
        
        return self._forward(
            torch.tensor(flattened_input_ids, dtype=torch.int32, device="cuda"),
            infer_state
        ).tolist()

    def _swap(
        self,
        seq_ids_list: list[int],
        is_swap_in: bool
    ):
        src_block_manager = self.cpu_block_manager if is_swap_in else self.gpu_block_manager
        dst_block_manager = self.gpu_block_manager if is_swap_in else self.cpu_block_manager
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        seq_lengths = src_block_manager.get_num_allocated_blocks(seq_ids) * self.args.block_size
        src_block_ids = src_block_manager.gather_allocated_blocks_and_free(seq_ids)
        dst_block_ids = dst_block_manager.allocate_blocks_for_seqs(seq_ids, seq_lengths)
        swiftllm_c.swap_blocks(
            src_block_ids.tolist(),
            dst_block_ids.tolist(),
            is_swap_in,

            self.k_cache, self.v_cache,
            self.k_swap, self.v_swap
        )
        
    @torch.inference_mode()
    def swap_in_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap in (move blocks from CPU to GPU) the specified sequences.
        """
        self._swap(seq_ids_list, True)
    
    @torch.inference_mode()
    def swap_out_seqs(
        self,
        seq_ids_list: list[int]
    ):
        """
        Swap out (move blocks from GPU to CPU) the specified sequences.
        """
        self._swap(seq_ids_list, False)

    @torch.inference_mode()
    def free_seqs_resources(self, seq_ids_list: list[int]):
        """
        Free the resources of the specified sequences.
        """
        seq_ids = torch.tensor(seq_ids_list, dtype=torch.int32, device="cuda")
        self.gpu_block_manager.free_blocks_for_seqs(seq_ids)
        self.cpu_block_manager.free_blocks_for_seqs(seq_ids)
