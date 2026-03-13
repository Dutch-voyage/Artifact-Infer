from collections import deque
import xxhash
import numpy as np

from src.services.nanovllm_v4.engine.sequence import Sequence

class QueryBlock:
    def __init__(self, block_id, block_size):
        self.block_id = block_id
        self.token_ids = [-1] * block_size
        self.last_token_index = -1
        self.block_size = block_size

    def initialize(self, token_ids: list[int]):
        assert len(token_ids) <= self.block_size
        self.token_ids[:len(token_ids)] = token_ids
        self.last_token_index = len(token_ids)
        self.last_token_index %= self.block_size
    
    def update(self, token_id: int):
        self.token_ids[self.last_token_index] = token_id
        self.last_token_index += 1
        self.last_token_index %= self.block_size

    def reset(self):
        self.token_ids = [-1] * self.block_size
        self.last_token_index = -1


class QueryBlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[QueryBlock] = [QueryBlock(i, block_size) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self, block_id: int) -> QueryBlock:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> QueryBlock:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self) -> bool:
        return len(self.free_block_ids) >= 1

    def allocate(self, seq: Sequence):
        assert not seq.query_block_id >= 0
        token_ids = seq.block(-1)
        block_id = self.free_block_ids[0]
        block = self._allocate_block(block_id)
        block.initialize(token_ids)
        seq.query_block_id = block_id
        seq.last_query_window_index = block.last_token_index

    def deallocate(self, seq: Sequence):
        self._deallocate_block(seq.query_block_id)
        seq.query_block_id = -1

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block = self.blocks[seq.query_block_id]
        block.update(seq.last_token)
        seq.last_query_window_index = block.last_token_index
        