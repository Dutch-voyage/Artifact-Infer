# This version of block manager does not implement with prefix caching
# only support block_size == 1 in this version as well
from collections import deque
import xxhash
import numpy as np

from src.services.nanovllm_v8.engine.sequence import Sequence, torch_rotl_uint8

from src.core.service_base import BaseService

import torch

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.token_ids = []

    def update(self, token_ids: list[int]):
        self.token_ids = token_ids

    def reset(self):
        self.token_ids = []


class BlockManager(BaseService):
    num_kv_heads = 8
    @property
    def name(self):
        return "BlockManager"

    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, if_fake_compress=True, *args, **kwargs):
        super().__init__()
        assert num_blocks > 0
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.if_fake_compress = if_fake_compress
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # during the engine running, some block will be released head by head, when all heads are released, the block can be reused
        self.released_block_ids: torch.Tensor = torch.zeros((num_blocks,), dtype=torch.uint8, device="cpu")
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        self.free_block_ids.append(block_id)
        self.released_block_ids[block_id] = 0
        
    def update_blocks_post_compression(self, seq: Sequence, budget: int):
        for block_id in reversed(seq.block_table[budget:]):
            self._deallocate_block(block_id)
        seq.block_table = seq.block_table[:budget]  
        
        # NOTE need further design
        
        seq.headwise_mask_layer_transpose = seq.headwise_mask_layer_transpose[..., :(seq.num_blocks_max_heads + 7) // 8]
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks_max_heads

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for i in range(seq.num_blocks_max_heads):
            token_ids = seq.block(i)
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            block.update(token_ids)
            seq.block_table.append(block_id)
            seq.block_id_to_count[block_id] = 0
            if i % 8 == 0 and i != 0:
                seq.headwise_mask_layer_transpose = torch.cat(
                    [seq.headwise_mask_layer_transpose, torch.zeros((Sequence.num_layers, Sequence.num_kv_heads, 1), device="cuda", dtype=torch.uint8)], dim=2
                )
            seq.headwise_mask_layer_transpose[:, :, i // 8] += seq.next_mask.to(seq.headwise_mask_layer_transpose.device)
            seq.next_mask = torch_rotl_uint8(seq.next_mask, 1)
            
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # print([self.blocks[index].hash for index in block_table]) 
        # NOTE when the block == 1, the handling logic is different 
        assert self.block_size == 1
        block_id = self.free_block_ids[0]
        seq.num_blocks_head += 1
        self._allocate_block(block_id)
        seq.block_table.append(block_id)
        seq.block_id_to_count[block_id] = 0
        # there must be at least one token after prefilling (allocate)
        if (seq.num_blocks_max_heads - 1) // 8 >= seq.headwise_mask_layer_transpose.shape[2]:
            seq.headwise_mask_layer_transpose = torch.cat(
                [seq.headwise_mask_layer_transpose, torch.zeros((Sequence.num_layers, Sequence.num_kv_heads, 1), device="cuda", dtype=torch.uint8)], dim=2
            )
        # print(seq.headwise_mask_layer_transpose[0])
        if self.if_fake_compress:
            seq.headwise_mask_layer_transpose[:, torch.arange(0, self.num_kv_heads), (seq.num_blocks_head - 1) // 8] += seq.next_mask.to(seq.headwise_mask_layer_transpose.device)
        else:
            seq.headwise_mask_layer_transpose[:, torch.arange(0, self.num_kv_heads), (seq.num_blocks_max_heads - 1) // 8] += seq.next_mask.to(seq.headwise_mask_layer_transpose.device)

        seq.next_mask = torch_rotl_uint8(seq.next_mask, 1)
