import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from ..config import Config
from ..engine.sequence import Sequence
from .models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .layers.sampler import Sampler
from ..utils.context import set_context, get_context, reset_context, set_context_field
from ..utils.loader import load_model
from src.services.nanovllm_vl.model_runner.models.mm_utils import init_embedding_cache
from src.services.nanovllm_vl.engine.mm_io_struct import ImageInputs


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)    
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # self.model = Qwen3ForCausalLM(hf_config)
        self.model = Qwen2_5_VLForConditionalGeneration(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # self.warmup_model()
        self.allocate_kv_cache()
        
        init_embedding_cache(getattr(self.config, "embedding_cache_size", 100) * 1024 * 1024)
        
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        if hasattr(hf_config, "head_dim"):
            head_dim = hf_config.head_dim
        else:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables
    
    def _expand_mrope_from_input(
        self,
        image_input: ImageInputs,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if image_input.mrope_position_delta.device.type != device:
            # transfer mrope_position_delta to device when the first running,
            # avoiding successvie host-to-device data transfer
            image_input.mrope_position_delta = image_input.mrope_position_delta.to(
                device, non_blocking=True
            )

        mrope_position_deltas = image_input.mrope_position_delta.flatten()
        mrope_positions = (
            (mrope_position_deltas + seq_len - 1).unsqueeze(0).repeat(3, 1)
        )
        return mrope_positions
    
    def _compute_mrope_positions_prefill(self, image_inputs: list[ImageInputs], seq_lens, extend_seq_lens, extend_prefix_lens):
        # batch_size * [3 * seq_len]
        batch_size = seq_lens.shape[0]
        mrope_positions_list = [[]] * batch_size
        for batch_idx in range(batch_size):
            image_input = image_inputs[batch_idx]
            
            extend_seq_len, extend_prefix_len = (
                extend_seq_lens[batch_idx],
                extend_prefix_lens[batch_idx],
            )
            if image_input is None:
                # text only
                mrope_positions = torch.tensor(
                    [
                        [
                            pos
                            for pos in range(
                                extend_prefix_len,
                                extend_prefix_len + extend_seq_len,
                            )
                        ]
                    ]
                    * 3
                )
            else:
                mrope_positions = image_input.mrope_positions[
                    :,
                    extend_prefix_len : extend_prefix_len + extend_seq_len,
                ]
                if mrope_positions.numel() == 0:
                    mrope_positions = self._expand_mrope_from_input(
                        image_input, seq_lens[batch_idx], self.device
                    )
            mrope_positions_list[batch_idx] = mrope_positions

        mrope_positions = torch.cat(
            [pos.to(device=self.device) for pos in mrope_positions_list],
            dim=1,
        ).to(dtype=torch.int64, device=self.device)
        
        return mrope_positions
    
    def _compute_mrope_positions_decode(self, image_inputs: list[ImageInputs], seq_lens):
        # batch_size * [3 * seq_len]
        batch_size = seq_lens.shape[0]
        mrope_positions_list = [[]] * batch_size
        for batch_idx in range(batch_size):
            image_input = image_inputs[batch_idx]
            # 3 * N
            if image_input is None:
                mrope_positions_list[batch_idx] = torch.full(
                    (3, 1),
                    seq_lens[batch_idx] - 1,
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                mrope_positions = self._expand_mrope_from_input(
                    image_input, seq_lens[batch_idx], self.device
                )
                mrope_positions_list[batch_idx] = mrope_positions

        mrope_positions = torch.cat(
            [pos.to(device=self.device) for pos in mrope_positions_list],
            dim=1,
        ).to(dtype=torch.int64, device=self.device)
        
        return mrope_positions
    
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        
        seq_lens = []
        prefix_lens = []
        extend_seq_lens = []
        
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        image_inputs = []

        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            
            # new metadata added here
            image_inputs.append(seq.image_inputs)
            seq_lens.append(seqlen)
            prefix_lens.append(seq.num_cached_tokens)
            extend_seq_lens.append(seqlen_q)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        
        for image_input in image_inputs:
            if image_input is not None:
                for image_item in image_input.image_items:
                    image_item.feature = getattr(image_item, "feature", None).to(self.device, non_blocking=True)
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # what to add in context?
        # prefix_lens, extend_seq_lens, image_inputs
        
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        extend_seq_lens = torch.tensor(extend_seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mrope_positions = self._compute_mrope_positions_prefill(image_inputs, seq_lens, extend_seq_lens, prefix_lens)

        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        
        set_context_field("image_inputs", image_inputs)
        set_context_field("mrope_positions", mrope_positions)
        set_context_field("extend_prefix_lens", prefix_lens)
        set_context_field("extend_seq_lens", extend_seq_lens)
        set_context_field("seq_lens", seq_lens)
        
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        seq_lens = []
        image_inputs = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            seq_lens.append(len(seq))
            image_inputs.append(seq.image_inputs)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        mrope_positions = self._compute_mrope_positions_decode(image_inputs, seq_lens)
        set_context(False, slot_mapping=slot_mapping, context_lens=seq_lens, block_tables=block_tables)
        set_context_field("mrope_positions", mrope_positions)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
