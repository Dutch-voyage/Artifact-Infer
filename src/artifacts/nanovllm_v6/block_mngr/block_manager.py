# This version of block manager does not implement with prefix caching
# only support block_size == 1 in this version as well
from collections import deque
import xxhash
import numpy as np

from src.services.nanovllm_v6.engine.sequence import Sequence

from src.core.service_base import BaseService

class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.token_ids = []

    def update(self, token_ids: list[int]):
        self.token_ids = token_ids

    def reset(self):
        self.token_ids = []


class BlockManager(BaseService):
    @property
    def name(self):
        return "BlockManager"

    def __init__(self, num_blocks: int, block_size: int):
        super().__init__()
        assert num_blocks > 0
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)

            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            block.update(token_ids)
            seq.block_table.append(block_id)
    
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._deallocate_block(block_id)
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        # print([self.blocks[index].hash for index in block_table])
        # NOTE when the block == 1, the handling logic is different 
        assert self.block_size == 1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)