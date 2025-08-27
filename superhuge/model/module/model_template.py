from typing import Sequence
from pydantic import BaseModel
import torch


class ModelInputArgs(BaseModel):
    fs: int
    window_length: int
    num_channels: int | None = None
    num_audio_features: int | None = None

    model_config = {"extra": "allow"}


class LossArgs(BaseModel):
    weight: Sequence[float | torch.Tensor] | None = None

    model_config = {"extra": "allow"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.weight is not None:
            self.weight = torch.Tensor(self.weight)  # type: ignore

    # write core schema for torch.Tensor validation and conversion
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            cls.__validate_tensor,
        )

    # try convert to torch.Tensor if possible.
    @classmethod
    def __validate_tensor(cls, value):
        if value is not None:
            try:
                return torch.Tensor(value)  # type: ignore
            except Exception:
                raise TypeError(
                    f"Cannot convert to torch.Tensor from type {type(value)}"
                )
        else:
            return None
