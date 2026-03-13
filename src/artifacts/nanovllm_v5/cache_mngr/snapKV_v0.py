import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import merge_kv
from einops import rearrange


class SnapKVCluster:
    def __init__(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        merge=None,
        recent_size=32,
        ratio=0.4,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio

    def reset(
        self,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling="avgpool",
        merge=None,
        recent_size=32,
        ratio=0.4,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    def compute_attn_cache(self, query_states, key_states, value_states):
        bsz, num_q_heads, q_len, head_dim = query_states.shape
        num_kv_heads = key_states.shape[1]
        attn_weights = torch.matmul(
            query_states[..., -self.window_size :, :],
            key_states.repeat_interleave(num_q_heads // num_kv_heads, -3).transpose(
                2, 3
            ),
        ) / math.sqrt(head_dim)
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size :, -self.window_size :] += attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        attn_weights_sum = attn_weights[
            :, :, -self.window_size :, : -self.window_size
        ].sum(dim=-2)

        attn_weights_sum = rearrange(
            attn_weights_sum,
            "bsz (h g) m-> bsz g h m",
            g=num_q_heads // num_kv_heads,
            h=num_kv_heads,
        ).mean(dim=1)

        if self.pooling == "avgpool":
            attn_cache = F.avg_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        elif self.pooling == "maxpool":
            attn_cache = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1,
            )
        else:
            raise ValueError("Pooling method not supported")

        return attn_cache

    # NOTE: here update_kv meaning head-wise selection and rearrangement
    def update_kv(
        self,
        query_states,
        key_states,
        value_states,
    ):
        # check if prefix phase
        # assert key_states.shape[-2] == query_states.shape[-2]

        _, _, kv_len, _ = key_states.shape
        bsz, num_q_heads, q_len, head_dim = query_states.shape

        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")

        if kv_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_cache = self.compute_attn_cache(query_states, key_states, value_states)
            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1
            ).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            if self.merge is not None:
                key_states, value_states = merge_kv(
                    key_states, value_states, indices, self.window_size, self.merge
                )
                return key_states, value_states

            k_past_compress = key_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            v_past_compress = value_states[:, :, : -self.window_size, :].gather(
                dim=2, index=indices
            )
            k_cur = key_states[:, :, -self.window_size :, :]
            v_cur = value_states[:, :, -self.window_size :, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states

    # NOTE: when updating kv indices only, without changing the actual kv states
    def update_kv_indices(
        self,
        query_states,
        key_states,
        value_states,
    ):
        # check if prefix phase
        # assert key_states.shape[-2] == query_states.shape[-2]

        bsz, num_q_heads, q_len, head_dim = query_states.shape
        num_kv_heads = key_states.shape[1]
        seq_len = key_states.shape[2]

        # print(f"SnapKV max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return torch.arange(
                0, seq_len, device="cuda", dtype=torch.int32
            )
        else:
            # average on the heads
            attn_cache = self.compute_attn_cache(
                query_states, key_states, value_states
            ).mean(dim=1)

            indices = attn_cache.topk(
                self.max_capacity_prompt - self.window_size, dim=-1
            ).indices

            indices = torch.cat(
                [
                    indices,
                    torch.arange(
                        seq_len - self.window_size,
                        seq_len,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    .unsqueeze(0)
                    .expand(bsz, -1),
                ],
                dim=-1,
            )

        return indices
