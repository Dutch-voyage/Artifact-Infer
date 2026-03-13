from src.core.artifact_base import Artifact
from src.core.service_base import BaseService

from ..attention.flashinfer_attention import (
    Attention,
    store_kvcache,
    read_kvcache,
    read_q_cache,
)
from src.services.nanovllm_v4.engine.sequence import Sequence
import torch

import itertools

from src.services.nanovllm_v4.utils.logging import get_log, set_log

# all implemntation here


class CacheManager(BaseService):
    @property
    def name(self):
        return "CacheManagerLayerwise"

    """
    This version of implementation only 
    """

    def __init__(self, attention_backend: Artifact, config, compressor=None):
        super().__init__()

        attention_backend.register(self)

        self.num_layers = config.hf_config.num_hidden_layers

        self.seq_to_layer_block_table = {}

        self.per_layer_page_indices = {}
        self.per_layer_seq_lens = {}

        self.cu_seqs: list[Sequence]

        self.compressor = compressor

    def init_block_table_after_prefill(self, seqs: list[Sequence]):
        self.cu_seqs = seqs
        for seq in seqs:
            self.seq_to_layer_block_table[seq.seq_id] = {}
            for layer_id in range(-1, self.num_layers):
                self.seq_to_layer_block_table[seq.seq_id][
                    layer_id
                ] = seq.block_table.copy()
            # self.seq_to_layer_block_table[seq.seq_id][-1] = seq.block_table.copy()

    def prepare_indices_flashinfer(self, seqs):
        # move to model runner before capturing cuda graph
        self.cu_seqs = seqs
        occupied_pages = 0
        cu_page_indices = torch.tensor(
            list(itertools.chain(*[self.seq_to_layer_block_table[seq.seq_id][-1] for seq in self.cu_seqs]))
        )
        
        occupied_pages += cu_page_indices.shape[0]
        seq_lens = torch.tensor(
            [len(self.seq_to_layer_block_table[seq.seq_id][-1]) for seq in self.cu_seqs]
        )
        self.per_layer_page_indices[-1] = cu_page_indices.to(torch.int32)
        self.per_layer_seq_lens[-1] = seq_lens.to(torch.int32)
        
        # for layer_id in range(self.num_layers):
        #     cu_page_indices = torch.tensor(
        #         list(
        #             itertools.chain(
        #                 *[
        #                     self.seq_to_layer_block_table[seq.seq_id][layer_id]
        #                     for seq in self.cu_seqs
        #                 ]
        #             )
        #         ),
        #         device="cuda",
        #     ).to(torch.int32)
        #     occupied_pages += cu_page_indices.shape[0]
        #     seq_lens = torch.tensor(
        #         [
        #             len(self.seq_to_layer_block_table[seq.seq_id][layer_id])
        #             for seq in self.cu_seqs
        #         ],
        #         device="cuda",
        #     )
        #     self.per_layer_page_indices[layer_id] = cu_page_indices
        #     self.per_layer_seq_lens[layer_id] = seq_lens
        #     # self.init_forward_metadata_capture_cuda_graph(bs, seq_lens[:bs], cu_page_indices)

        log = get_log()
        log.occupied_pages = occupied_pages
        set_log(log)
        
    def update_indices_per_layer(self):
        self.prepare_metadata_for_attn(
            self.per_layer_seq_lens[-1],
            self.per_layer_page_indices[-1],
            -1,
        )
        # for layer_id in range(self.num_layers):
        #     self.prepare_metadata_for_attn(
        #         self.per_layer_seq_lens[layer_id],
        #         self.per_layer_page_indices[layer_id],
        #         layer_id,
        #     )

    def update_indices_per_layer_capture(self, bs: int):
        self.init_forward_metadata_capture_cuda_graph(
            bs,
            self.per_layer_seq_lens[-1][:bs],
            self.per_layer_page_indices[-1],
            -1,
        )
        # for layer_id in range(self.num_layers):
        #     self.init_forward_metadata_capture_cuda_graph(
        #         bs,
        #         self.per_layer_seq_lens[layer_id][:bs],
        #         self.per_layer_page_indices[layer_id],
        #         layer_id,
        #     )

    def update_indices_per_layer_replay(self, bs: int):
        self.init_forward_metadata_replay_cuda_graph(
            bs,
            self.per_layer_seq_lens[-1][:bs],
            self.per_layer_page_indices[-1],
            -1,
        )
        
        # for layer_id in range(self.num_layers):
        #     self.init_forward_metadata_replay_cuda_graph(
        #         bs,
        #         self.per_layer_seq_lens[layer_id][:bs],
        #         self.per_layer_page_indices[layer_id],
        #         layer_id,
        #     )

    def read_and_store_cache(self, q_cache, k_cache, v_cache, layer_id: int):
        """
        option 1: per-sequence handling

        option 2: like flashinfer's layout, handling with packed indices,
        """
        slot_mappings = []
        for seq in self.cu_seqs:
            slot_mappings.extend(self.seq_to_layer_block_table[seq.seq_id][layer_id])

        assert len(self.cu_seqs) == 1, "Currently only support single request"

        slot_mappings_tensor = torch.tensor(slot_mappings, device="cuda").to(
            torch.int32
        )

        query_slot_mapping = [seq.query_block_id for seq in self.cu_seqs]

        query_slot_mapping_tensor = torch.tensor(query_slot_mapping, device="cuda").to(
            torch.int32
        )

        query = read_q_cache(
            q_cache=q_cache,
            query_slot_mapping=query_slot_mapping_tensor,
        )

        key, value = read_kvcache(
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )

        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        updated_k, updated_v = self.compressor.update_kv(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )

        key = updated_k.transpose(1, 2).squeeze(0).contiguous()
        value = updated_v.transpose(1, 2).squeeze(0).contiguous()

        # for single request only
        slot_mappings_list = slot_mappings_tensor.tolist()

        # for pruned_block_id in slot_mappings_list[key.shape[0]:]:
        #     self.blocks[pruned_block_id].ref_count -= 1
        #     if self.blocks[pruned_block_id].ref_count == 0:
        #         self._deallocate_block(pruned_block_id)

        slot_mappings_tensor = slot_mappings_tensor[: key.shape[0]]
        
        self.seq_to_layer_block_table[self.cu_seqs[0].seq_id][layer_id] = (
            slot_mappings_list[: key.shape[0]]
        )

        store_kvcache(
            key=key,
            value=value,
            k_cache=k_cache,
            v_cache=v_cache,
            slot_mapping=slot_mappings_tensor,
        )
