from functools import lru_cache
import torch
from torch import nn

from typing import List, Optional, Tuple

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb_native(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1, 
):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    # embedding is performed in float
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        dtype: torch.dtype = torch.float, 
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=dtype)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    elif "mrope_section" in rope_scaling:

        rotary_emb = MRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            dtype=torch.float32,
            mrope_section=rope_scaling["mrope_section"],
            mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
        )
        
    return rotary_emb

class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, dtype
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Auto-correct by scaling the mrope_section proportionally
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor))
                        for section in self.mrope_section
                    ]
                    # Ensure the sum exactly matches by adjusting the last element
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    # If all sections are 0, create a default distribution
                    self.mrope_section = [
                        expected_sum // len(self.mrope_section)
                    ] * len(self.mrope_section)
                    # Handle remainder
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1

                print(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if (
            self.cos_sin_cache.device != query.device
            or self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = apply_rotary_emb(key_rot, cos, sin)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    # Copied from https://github.com/huggingface/transformers/blob/c8e0e603de9b3d49161a15fe6e8ea84badfb5d02/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1439
    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: Optional[int] = None,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mrope_position_deltas = []
        if input_ids is not None and image_grid_thw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index = 0
            for i, input_ids in enumerate(total_input_ids):
                image_nums = 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images = image_nums
                for _ in range(image_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    if model_type == "qwen2_5_vl":
                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(
                            -1, llm_grid_h * llm_grid_w
                        )

                        time_tensor = (
                            expanded_range * second_per_grid_t * tokens_per_second
                        )

                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()
                    elif model_type in (
                        "qwen2_vl",
                        "qwen3_vl",
                        "qwen3_vl_moe",
                    ):
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                    else:
                        raise RuntimeError(f"Unimplemented model type: {model_type}")
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            s = input_ids.shape[1]
            position_ids = torch.arange(s)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - s
            return position_ids, mrope_position_deltas