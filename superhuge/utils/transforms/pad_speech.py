import torch
from .abc import Transform


class PadSpeech(Transform):
    def __init__(self, /, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, data: torch.Tensor):
        if data.ndim == 1:
            data = torch.stack([data, torch.zeros_like(data)], dim=1)
        elif data.ndim == 2 and data.shape[1] == 1:
            data = torch.cat([data, torch.zeros_like(data)], dim=1)
        return data
