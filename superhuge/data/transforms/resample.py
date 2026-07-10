from typing import Any, Literal
from warnings import warn
import numpy as np
from scipy.signal import resample, decimate, resample_poly

from .abc import Transform


class Resample(Transform):
    """Reduces EEG sampling rate to improve efficiency."""

    def __init__(
        self,
        /,
        *,
        old_fs: int | float,
        new_fs: int | float,
        method: Literal["resample", "decimate", "resample_poly"] = "resample_poly",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if kwargs["when"] == "before_returning":
            warn(
                "Resampling before returning is not recommended. Dataset slicing is based on the new sampling rate, and therefore the data slicing will be incorrect if resampling is done after slicing."
            )
        self.old_fs = old_fs
        self.new_fs = new_fs
        self.method = method

    def __call__(self, x: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        super().__call__(x)
        if self.method == "resample":
            num_samples = int(x.shape[0] * self.new_fs / self.old_fs)
            return {"x": resample(x, num_samples, axis=0)}  # type: ignore
        elif self.method == "decimate":
            decimation_factor = int(self.old_fs / self.new_fs)
            assert (
                self.old_fs % self.new_fs == 0
            ), "Old sampling rate must be an integer multiple of new sampling rate for decimation."
            return {"x": decimate(x, decimation_factor, axis=0)}  # type: ignore
        elif self.method == "resample_poly":
            return {"x": resample_poly(x, self.new_fs, self.old_fs, axis=0)}  # type: ignore
        else:
            raise ValueError(
                f"Invalid resampling method: {self.method}. Choose from 'resample', 'decimate', or 'resample_poly'."
            )
