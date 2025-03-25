import numpy as np
import torch
from .abc import Transform


class PadSpeech(Transform):
    def __init__(self, /, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, data: torch.Tensor):
        if data.ndim == 1:
            data = np.stack([data, np.zeros_like(data)], axis=1)
        elif data.ndim == 2 and data.shape[1] == 1:
            data = np.concatenate([data, np.zeros_like(data)], axis=1)
        return data
