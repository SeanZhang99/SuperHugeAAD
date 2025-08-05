from typing import Any
import numpy as np

from .abc import Transform


class ChannelMask(Transform):
    """Randomly masks EEG channels with probability `p_mask`."""

    def __init__(self, /, *, p_mask: float = 0.2, **kwargs) -> None:
        if not (0 <= p_mask <= 1):
            raise ValueError("p_mask should be between 0 and 1.")
        super().__init__(**kwargs)
        self.p_mask = p_mask

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x)
        if self.roll():
            x *= self.dice.random(x.shape[-1]) > self.p_mask
        return x, *args
