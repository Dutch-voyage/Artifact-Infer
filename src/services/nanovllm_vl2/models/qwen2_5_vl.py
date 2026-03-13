from functools import partial
from typing import List, TypedDict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed,
    Qwen2_5_VisionRotaryEmbedding,
)

from nanovllm.engine.mm_io_struct import (
    MultimodalDataItem,
    MultimodalInputs,
)
from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from nanovllm.layers.vision_attention import VisionAttention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.models.qwen2 import Qwen2Model
from nanovllm.models.utils import permute_inv
from nanovllm.models.registry import register_model
from nanovllm.models.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens, general_mm_embed_routine
from nanovllm.utils.context import get_context_field


class Qwen2VLVideoInputs(TypedDict):
    pixel_values_videos: torch.Tensor
    """Shape:
    `(num_patches,
      num_channels * temporal_patch_size * patch_size * patch_size)`
    """

    video_grid_thw: torch.Tensor
    """Shape: `(num_videos, 3)`

    This should be in `(grid_t, grid_h, grid_w)` format.
    """
    
class Qwen2_5_VLMLP(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        bias: bool = True,
        hidden_act="silu",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,  # [gate_proj, up_proj]
            bias=bias,
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
        )
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act(gate) * up
        x_down = self.down_proj(x)
        return x_down
    
class Qwen2_5_VisionBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        hidden_act="silu",
        norm_layer: Type[nn.Module] = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)

        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            rotary_embed="normal",
            proj_bias=True,
            qkv_bias=True, 
        )
        self.mlp = Qwen2_5_VLMLP(
            dim,
            intermediate_dim,
            hidden_act=hidden_act,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x

class Qwen2_5_VisionPatchMerger(nn.Module):

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: [S, B, context_dim]
        S, B, D = x.shape
        x2d = x.reshape(-1, D)
        x2d = self.ln_q(x2d)  # RMSNorm expects 2D
        x2d = x2d.view(-1, self.hidden_size)  # group into spatial_merge_unit
        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel = mlp_fc1(x2d)
        x_parallel = mlp_act(x_parallel)
        out = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionTransformer(nn.Module):

    def __init__(
        self,
        vision_config: Qwen2_5_VLVisionConfig,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        patch_size: int = vision_config.patch_size
        temporal_patch_size: int = vision_config.temporal_patch_size
        spatial_merge_size: int = vision_config.spatial_merge_size
        self.spatial_merge_size = spatial_merge_size
        self.spatial_merge_unit: int = spatial_merge_size * spatial_merge_size
        in_channels: int = vision_config.in_channels
        hidden_size: int = vision_config.hidden_size
        depth: int = vision_config.depth
        num_heads: int = vision_config.num_heads
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        mlp_hidden_size: int = ((vision_config.intermediate_size + 7) // 8) * 8
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=hidden_size,
                    intermediate_dim=mlp_hidden_size,
                    num_heads=num_heads,
                    hidden_act=vision_config.hidden_act,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            dim=vision_config.out_hidden_size,
            context_dim=hidden_size,
            spatial_merge_size=spatial_merge_size,
        )

    def get_window_index(self, grid_thw):
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )
        window_index: list = []
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for i in range(grid_thw.size(0)):
            t, h, w = grid_thw[i].tolist()
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)

            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()

            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
    
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=x.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        # Move window_index to the same device as x before using it to index x
        window_index = window_index.to(device=x.device)
        reverse_indices = permute_inv(window_index)

        # Ensure rotary_pos_emb is on the same device/dtype as x
        rotary_pos_emb = rotary_pos_emb.to(device=x.device, dtype=x.dtype)

        seq_len, _ = x.size()

        x = x.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        x = x[window_index, :, :]
        x = x.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        # After building position_embeddings, make sure both cos and sin are on the same device/dtype as the attention input
        position_embeddings = (
            position_embeddings[0].to(x.device, x.dtype),
            position_embeddings[1].to(x.device, x.dtype),
        )
        
        # compute cu_seqlens - move cu_seqlens to GPU and make it int32
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], device=x.device, dtype=torch.int32),
                (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2])
                .cumsum(dim=0)
                .to(device=x.device, dtype=torch.int32),
            ]
        )
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        # transformers
        x = x.unsqueeze(1)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            x = blk(
                x, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings
            )

        # adapter
        x = self.merger(x)
        x = x[reverse_indices, :]

        return x
    

@register_model("qwen2_5_vl")
class Qwen2_5_VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
    
    @property
    def name(self):
        return "qwen2_5_vl"
    
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
    ) -> None:
        super().__init__()
        
        self.config = config
        self.visual = Qwen2_5_VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
        )

        self.model = Qwen2Model(
            config,
        )

        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        """Run forward pass for Qwen2_5-VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen2-VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
                (Use input_metadata.mrope_positions to replace it)
        """
        positions = get_context_field("mrope_positions")
        mm_inputs = get_context_field("mm_inputs")
        
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            mm_inputs=mm_inputs,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        return hidden_states

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()
        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # in qwen-vl, last dim is the same
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert video_grid_thw.dim() == 2, video_grid_thw.dim()
        video_embeds = self.visual(pixel_values, grid_thw=video_grid_thw)
        return video_embeds
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
    
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight