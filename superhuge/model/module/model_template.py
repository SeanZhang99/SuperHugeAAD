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
