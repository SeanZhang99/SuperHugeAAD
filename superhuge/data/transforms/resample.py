from typing import Any
import numpy as np
from scipy.signal import resample

from .abc import Transform


class Resample(Transform):
    """Reduces EEG sampling rate to improve efficiency."""

    def __init__(
        self, /, *, old_fs: int | float, new_fs: int | float, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.old_fs = old_fs
        self.new_fs = new_fs

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x, *args)
        num_samples = int(x.shape[0] * self.new_fs / self.old_fs)
        return resample(x, num_samples, axis=0), *args  # type: ignore
