from typing import Any
from warnings import warn
import numpy as np
from scipy.signal import resample

from .abc import Transform


class Resample(Transform):
    """Reduces EEG sampling rate to improve efficiency."""

    def __init__(
        self, /, *, old_fs: int | float, new_fs: int | float, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if kwargs["when"] == "before_returning":
            warn(
                "Resampling before returning is not recommended. Dataset slicing is based on the new sampling rate, and therefore the data slicing will be incorrect if resampling is done after slicing."
            )
        self.old_fs = old_fs
        self.new_fs = new_fs

    def __call__(self, x: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        super().__call__(x)
        num_samples = int(x.shape[0] * self.new_fs / self.old_fs)
        return {"x": resample(x, num_samples, axis=0)}  # type: ignore
