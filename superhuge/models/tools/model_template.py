from dataclasses import dataclass
from typing import TypedDict
from pydantic import BaseModel
from torch.nn import Module


@dataclass
class ModelTemplate:
    fs: int
    window_length: int
    


class ModelInputArgs(BaseModel):
    fs: int
    window_length: int
    num_channels: int | None = None

    model_config = {"extra": "allow"}
