import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from ...engine.mm_io_struct import ImageDataItem
from ...engine.mm_utils import load_image

from...engine.mm_io_struct import ImageDataItem


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image, in given order
    images: Optional[list[Union[Image.Image, dict]]] = dataclasses.field(
        default_factory=list
    )

@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[str, List[str]]] = None

    image_token_id: Optional[int] = None
    
    image_token_regex: Optional[re.Pattern] = None

    def build(self, processor):
        self.convert_to_strs(processor)
        self.parse_regex()
        return self

    def parse_regex(self):
        if self.image_token is not None and self.image_token_regex is None: 
            self.image_token_regex = re.compile(re.escape(self.image_token)) 
    
    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def is_image_token(self, token: str) -> bool:
        if token == self.image_token:
            return True
        
        if self.image_token_regex and self.image_token_regex.match(token):
            return True
        
        return False

    def convert_to_strs(self, processor):
        if not self.image_token:
            self.image_token = self.convert_to_str(self.image_token_id, processor)

    def get_token_id(self) -> Optional[int]:
        return self.image_token_id
    
    def get_image_regex(self) -> Optional[re.Pattern]:
        pattern = self.image_token_regex
        if pattern is not None:
            combined = "(" + "|".join([f"(?:{self.image_token_regex.pattern})"]) + ")"
            return re.compile(combined)  
        else:
            return None
           

class BaseMultimodalProcessor(ABC):
    models = []

    def __init__(
        self, hf_config, server_args, _processor, transport_mode, *args, **kwargs
    ):
        self.hf_config = hf_config
        self._processor = _processor
        self.server_args = server_args
        self.transport_mode = transport_mode

        # FIXME: not accurate, model and image specific
        self.NUM_TOKEN_PER_FRAME = 330

        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("SGLANG_IO_WORKERS", 4))
        )
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("fork"),
            max_workers=int(os.environ.get("SGLANG_CPU_WORKERS", os.cpu_count())),
        )

        self.ATTR_NAME_TO_IMAGE = ["pixel_values",
            "image_sizes",
            "image_grid_thw",
            "image_attention_mask",
            "image_emb_mask",
            "images_spatial_crop",
            "images_crop",
            "tgt_size",
            "image_grid_hws",
            "aspect_ratio_ids",
            "aspect_ratio_mask",
            "num_patches",
            "patch_pixel_values",
            "block_sizes",
            
            "precomputed_embeddings",
        ]

        # name of the feature filed
        # TODO: pass from processors
        self.FEATURE_NAMES = [
            "pixel_values",
        ]

    def process_mm_data(
        self, input_text, images=None, **kwargs
    ) -> dict:
        """
        process multimodal data with transformers AutoProcessor
        """
        if images:
            kwargs["images"] = images

        processor = self._processor
        
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        return result

    @abstractmethod
    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        pass


    @staticmethod
    def _load_single_item(
        data,
        discard_alpha_channel: bool = True,
    ):
        """
        Load a single multimodal data.

        If data is precomputed, returns directly.

        Static method that can be pickled for multiprocessing"""
        if isinstance(data, dict):
            return data
        try:
            img, _ = load_image(data)
            if discard_alpha_channel and img.mode != "RGB":
                img = img.convert("RGB")
            return img
           
        except Exception as e:
            raise RuntimeError(f"Error while loading data {data}: {e}")

    def submit_data_loading_tasks(
        self,
        text_parts: List[str],
        multimodal_tokens: MultimodalSpecialTokens,
        image_data_iterator: Optional[Iterator[Any]],
        discard_alpha_channel: bool = True,
    ) -> Tuple[List, List]:
        """
        load multimodal data parallelly using iterators.
        """
        futures = []
        task_info = []
        
        for text_part in text_parts:
            if multimodal_tokens.is_image_token(text_part):
                if image_data_iterator is None:
                    raise ValueError(f"No data iterator found for token: {text_part}")

                try:
                    data = next(image_data_iterator)
                except StopIteration:
                    raise ValueError(
                        f"Mismatch: More '{text_part}' tokens found than corresponding data items provided."
                    )

                futures.append(
                    self.io_executor.submit(
                        BaseMultimodalProcessor._load_single_item,
                        data,
                        discard_alpha_channel,
                    )
                )
                task_info.append(data)

        try:
            next(image_data_iterator)
            print(
                f"Warning: More  data items provided than corresponding tokens found in the prompt."
            )
        except StopIteration:
            pass
        except Exception:
            pass

        return futures, task_info

    def load_mm_data(
        self,
        prompt: str,
        multimodal_tokens: MultimodalSpecialTokens,
        image_data: Optional[list] = None,
        return_text: Optional[bool] = True,
        discard_alpha_channel: bool = True,
    ) -> BaseMultiModalProcessorOutput:
        """
        Each frame of video/image will be replaced by a single image token

        Args:
            multimodal_tokens (list[str]): list of special token which denoting a single multimodal data
                e.g. image token or audio token
            discard_alpha_channel: if True, discards the alpha channel in the returned images

        """
        multimodal_tokens_pattern = multimodal_tokens.get_image_regex()
        if isinstance(prompt, list) and return_text:
            assert len(prompt) and isinstance(prompt[0], int)
            prompt = self._processor.tokenizer.decode(prompt)
        else:
            prompt = prompt

        assert isinstance(prompt, str)
        assert isinstance(image_data, list)
        # split text into list of normal text and special tokens
        text_parts = re.split(multimodal_tokens_pattern, prompt)
        # collect data
        image_data_iterator = None
        if multimodal_tokens.image_token and image_data:
            image_data_iterator = iter(image_data)

        # futures: the futures of loaded data
        # task_info: modality, raw_data, and other metadata of each data
        futures, task_info = self.submit_data_loading_tasks(
            text_parts=text_parts,
            multimodal_tokens=multimodal_tokens,
            image_data_iterator=image_data_iterator,
            discard_alpha_channel=discard_alpha_channel,
        )
        task_info_iter = iter(task_info)
        futures_iter = iter(futures)

        # Process results
        images = []
        new_text_parts = []
        for text_part in text_parts:
            try:
                if multimodal_tokens_pattern.match(text_part):
                    raw_data = next(task_info_iter)
                    is_precomputed = isinstance(raw_data, dict)
                    result = next(futures_iter).result()

                    # If data is already processed it will be a
                    # dictionary(precomputed). In this case we want to keep the
                    # expanded tokens in text_part. Otherwise, we will
                    # call the processor code, so keep only a single image
                    # token.
                    mm_tokens = (
                        text_part
                        if is_precomputed
                        else multimodal_tokens.image_token
                    )
                    result = [result] if not isinstance(result, list) else result
                    images += result
                    new_text_parts += mm_tokens * len(result)
                else:
                    # normal text
                    new_text_parts += [text_part]

            except Exception as e:
                raise RuntimeError(
                    f"An exception occurred while loading multimodal data: {e}"
                )
        return BaseMultiModalProcessorOutput(
            images=images,
            input_text="".join(new_text_parts),
        )

    @staticmethod
    def get_mm_items_offset(
        input_ids: torch.Tensor, mm_token_id: int
    ) -> List[Tuple[int, int]]:
        """
        Get a set of range for mm_items from input_ids
        Example:
            input_ids = [1, 2, 3, 3, 3, 4, 3, 3]
            mm_token_id = 3
            return result = [(2,4),(6,7)]
        """
        mask = input_ids == mm_token_id
        start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]

        return list(zip(start_positions.tolist(), end_positions.tolist()))

    @staticmethod
    def get_mm_items_offset_by_pair(
        input_ids: torch.Tensor, mm_start_id: int, mm_end_id: int
    ) -> List[Tuple[int, int]]:
        indices_start = (input_ids == mm_start_id).nonzero(as_tuple=True)[0] + 1
        indices_end = (input_ids == mm_end_id).nonzero(as_tuple=True)[0] - 1

        return list(zip(indices_start.tolist(), indices_end.tolist()))

    def collect_mm_items_from_processor_output(
        self, data_dict: dict
    ) -> ImageDataItem:
        """Create mm_items directly from processor output."""
        image_item: ImageDataItem = None
        for attr_name, value in data_dict.items():
            if attr_name == "input_ids":
                continue
            if attr_name in self.ATTR_NAME_TO_IMAGE:
                if image_item is None:
                    image_item = ImageDataItem()
                if attr_name in self.FEATURE_NAMES:
                    attr_name = "feature"

                setattr(image_item, attr_name, value)

        return [image_item]

    def _process_and_collect_mm_items(
        self, input_text: str, images=None, **kwargs
    ) -> Tuple[List[ImageDataItem], torch.Tensor, dict]:
        """
        Helper method to process multimodal data and create mm_items in one step.

        Returns:
            Tuple of (created mm_items, input_ids)
        """
        ret = self.process_mm_data(
            input_text=input_text, images=images, **kwargs
        )

        input_ids = ret["input_ids"].flatten()
        collected_items = self.collect_mm_items_from_processor_output(ret)

        return collected_items, input_ids, ret

    def process_and_combine_mm_data(
        self,
        base_output: BaseMultiModalProcessorOutput,
        mm_tokens: MultimodalSpecialTokens,
        **kwargs,
    ) -> Tuple[List[ImageDataItem], torch.Tensor, dict]:
        """
        Process multimodal data and return the combined multimodal items and input_ids.
        Supports mixed modalities (images and audio in the same request).

        Returns:
            Tuple of (list of mm_items, input_ids)
        """
        # Handle text-only case
        if not base_output.images:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()
            return [], input_ids, {}

        dict_items, raw_images = [], []
        
        for item in base_output.images:
            if isinstance(item, dict):
                dict_items.append(item)
            else:
                raw_images.append(item)
            
        # Process items and get input_ids
        all_collected_items: list[ImageDataItem] = []
        input_ids = None

        collected_items, input_ids, ret = self._process_and_collect_mm_items(
            input_text=base_output.input_text,
            images=base_output.images,
            **kwargs,
        )
        all_collected_items = collected_items
        
        # Handle dict items (already processed)
        for dict_item in dict_items:
            all_collected_items.extend(
                self.collect_mm_items_from_processor_output(dict_item)
            )

        # Fallback tokenization if no raw items were processed
        if input_ids is None:
            input_ids = self._processor.tokenizer(
                base_output.input_text,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.flatten()

        # Add offsets to all items
        for mm_item in all_collected_items:
            mm_token_id = mm_tokens.get_token_id()
            mm_item.offsets = self.get_mm_items_offset(
                input_ids=input_ids,
                mm_token_id=mm_token_id,
            )

        return all_collected_items, input_ids, ret
