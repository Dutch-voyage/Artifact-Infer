from PIL.Image import Image
from dataclasses import dataclass

from typing import List, Optional, Union
import torch
import numpy as np

from .mm_utils import hash_feature  


@dataclass
class ImageDataItem:
    hash: int = None
    pad_value: int = None   
    offsets: Optional[List] = None
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None
    
    feature: Union[torch.Tensor, np.ndarray] = None # pixel_values for image
        
    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = self.hash % (1 << 30)
    
    def merge(self, other):
        self.feature += other.feature
        self.offsets += other.offsets
        self.hash = hash((self.hash, other.hash))
        self.set_pad_value()
        

@dataclass
class ImageInputs:
    image_items: List[ImageDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None
    
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None
    
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(obj: dict):
        ret = ImageInputs(
            image_items=obj["image_items"], 
        )
        
        assert isinstance(ret.image_items, list)
        
        for item in ret.image_items:
            assert isinstance(item, ImageDataItem)
            item.set_pad_value()
        
        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])
        
        return ret
    
    def merge(self, other: "ImageInputs"):
        
        # args to be merged
        optional_args = [
            "image_items", 
            "image_pad_len", 
        ]
        
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))

        