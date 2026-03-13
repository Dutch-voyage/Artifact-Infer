from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention import Attention, store_kvcache, read_kvcache
from src.services.nanovllm_v3.engine.sequence import Sequence
import torch

import itertools

# all implemntation here


class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerLayerwise"

    """
    This version of implementation only 
    """

    def __init__(self, attention_backend: Artifact, config):
        super().__init__()

        attention_backend.register(self)

        self.num_layers = config.num_hidden_layers

        self.seq_to_layer_block_table = {}

        self.per_layer_page_indices = {}
        self.per_layer_seq_lens = {}

        self.cu_seqs: list[Sequence]

    def init_block_table_after_prefill(self, seqs: list[Sequence]):
        self.cu_seqs = seqs
        for seq in seqs:
            self.seq_to_layer_block_table[seq.seq_id] = {}
            for layer_id in range(self.num_layers):
                self.seq_to_layer_block_table[seq.seq_id][
                    layer_id
                ] = seq.block_table.copy()

    def prepare_indices_flashinfer(self):
        # move to model runner before capturing cuda graph
        for layer_id in range(self.num_layers):
            cu_page_indices = torch.tensor(
                list(
                    itertools.chain(
                        *[
                            self.seq_to_layer_block_table[seq.seq_id][layer_id]
                            for seq in self.cu_seqs
                        ]
                    )
                ),
                device="cuda",
            ).to(torch.int32)
            seq_lens = torch.tensor(
                [
                    len(self.seq_to_layer_block_table[seq.seq_id][layer_id])
                    for seq in self.cu_seqs
                ],
                device="cuda",
            )
            self.per_layer_page_indices[layer_id] = cu_page_indices
            self.per_layer_seq_lens[layer_id] = seq_lens
            # self.init_forward_metadata_capture_cuda_graph(bs, seq_lens[:bs], cu_page_indices)

    def update_indices_per_layer_capture(self, bs: int):
        for layer_id in range(self.num_layers):
            self.init_forward_metadata_capture_cuda_graph(
                bs,
                self.per_layer_seq_lens[layer_id][:bs],
                self.per_layer_page_indices[layer_id],
                layer_id, 
            )
        # layer_id = 0
        # self.init_forward_metadata_capture_cuda_graph(
        #     bs,
        #     self.per_layer_seq_lens[layer_id][:bs],
        #     self.per_layer_page_indices[layer_id],
        #     layer_id, 
        # )

    def update_indices_per_layer_replay(self, bs: int):
        # for layer_id in range(self.num_layers):
        #     self.init_forward_metadata_replay_cuda_graph(
        #         bs,
        #         self.per_layer_seq_lens[layer_id][:bs],
        #         self.per_layer_page_indices[layer_id],
        #         layer_id, 
        #     )
        layer_id = 0
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.per_layer_seq_lens[layer_id][:bs],
            self.per_layer_page_indices[layer_id],
            layer_id, 
        )

    def read_and_write(self, k_cache, v_cache, layer_id: int):
        """
        option 1: per-sequence handling

        option 2: like flashinfer's layout, handling with packed indices,
        """
        slot_mappings = []
        for seq in self.cu_seqs:
            slot_mappings.extend(self.seq_to_layer_block_table[seq.seq_id][layer_id])

        slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
            torch.int32
        )

        key, value = read_kvcache(
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )

        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )
