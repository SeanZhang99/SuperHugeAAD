import numpy as np

from .abc import Transform


class PadSpeech(Transform):
    def __init__(self, /, **kwargs) -> None:
        super().__init__(**kwargs)

    def __call__(self, x: np.ndarray, /, *args, **kwargs):
        if x.ndim == 1:
            x = np.stack([x, np.zeros_like(x)], axis=-1)
        elif x.shape[-1] == 1:
            x = np.concatenate([x, np.zeros_like(x)], axis=-1)
        return x
