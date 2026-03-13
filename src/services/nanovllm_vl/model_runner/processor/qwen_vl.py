import asyncio
import math
import os
import re
import time
from typing import List, Union

import numpy as np
import torch
import torchvision
from PIL import Image

from ..layers.rotary_embedding import MRotaryEmbedding
from src.services.nanovllm_vl.model_runner.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from .base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from .base_processor import MultimodalSpecialTokens

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200
RESIZE_RESAMPLE = None
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    width, height = image.size
    min_pixels = min_pixels
    max_pixels = max_pixels
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height), resample=RESIZE_RESAMPLE)
    return image


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


async def resize_image_async(
    image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    size_factor: int = IMAGE_FACTOR,
):
    return resize_image(image, min_pixels, max_pixels, size_factor)

# Compatible with Qwen-VL & Qwen-Omni Series
class QwenVLImageProcessor(SGLangBaseProcessor):
    models = [
        Qwen2_5_VLForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.model_type = hf_config.model_type

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IM_START_TOKEN_ID = hf_config.vision_start_token_id
        self.IM_END_TOKEN_ID = hf_config.vision_end_token_id
        self.vision_start_token_id = hf_config.vision_start_token_id
        self.vision_end_token_id = hf_config.vision_end_token_id

        self.NUM_TOKEN_PER_FRAME = 770
        self.IMAGE_FACTOR = 28
        self.MIN_PIXELS = 4 * 28 * 28
        self.MAX_PIXELS = 16384 * 28 * 28
        self.MAX_RATIO = 200
        pattern= re.compile(r"<\|vision_start\|>(?:<\|image_pad\|>)+<\|vision_end\|>")
        
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            image_token_id=hf_config.image_token_id,
            # The regex that matches expanded image tokens.
            image_token_regex=pattern, 
        ).build(_processor)

        self.process = self.process_mm_data_sync

    def process_mm_data_sync(
        self, 
        image_data: List[Union[str, bytes]],
        input_text, 
        *args, 
        **kwargs
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
                
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            base_output.images = [resize_image(image) for image in base_output.images]
        
        image_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        
        input_ids = input_ids.flatten()

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "image_items": image_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
            
        
    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        # Qwen-specific: resize images if they are raw Image objects
        if base_output.images and isinstance(base_output.images[0], Image.Image):
            resize_tasks = [resize_image_async(image) for image in base_output.images]
            base_output.images = await asyncio.gather(*resize_tasks)
        
        image_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids = input_ids.flatten()

        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self.hf_config.vision_config.spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            vision_start_token_id=self.vision_start_token_id,
            model_type=self.model_type,
            tokens_per_second=getattr(
                self.hf_config.vision_config, "tokens_per_second", None
            ),
            input_ids=input_ids.unsqueeze(0),
            image_grid_thw=getattr(ret, "image_grid_thw", None),
        )
        mrope_positions = mrope_positions.squeeze(1)

        return {
            "input_ids": input_ids.tolist(),
            "image_items": image_items,
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.mm_tokens.image_token_id,
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
        }
