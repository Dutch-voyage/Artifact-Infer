import dataclasses
from enum import Enum, auto
from typing import List, Optional, Union, Any

import torch
import numpy as np


def flatten_nested_list(nested_list):
    if isinstance(nested_list, list):
        return [
            item for sublist in nested_list for item in flatten_nested_list(sublist)
        ]
    else:
        return [nested_list]

class Modality(Enum):
    IMAGE = auto()
    MULTI_IMAGES = auto()
    VIDEO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO]


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem contains all inputs for one modality.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    # the raw features returned by processor, e.g. pixel_values
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        from nanovllm.engine.mm_utils import hash_feature

        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = self.hash % (1 << 30)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_image(self):
        return self.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video()

    def validate(self):
        ...
        # TODO

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def merge(self, other):
        self.feature += other.feature
        self.offsets += other.offsets
        self.hash = hash((self.hash, other.hash))
        self.set_pad_value()


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None
    
    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(obj: dict):
        ret = MultimodalInputs(
            mm_items=obj["mm_items"],
        )

        assert isinstance(ret.mm_items, list)
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]
        for item in ret.mm_items:
            item.set_pad_value()

        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
        ]
        for arg in optional_args:
            if arg in obj:
                setattr(ret, arg, obj[arg])

        return ret