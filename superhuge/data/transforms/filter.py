from typing import Sequence

import numpy as np
from scipy.signal import butter, filtfilt

from .abc import Transform


class Filter(Transform):
    """Applies a filter to EEG data."""

    def __init__(
        self,
        Wn: float | Sequence[float],
        fs: float,
        btype: str,
        order: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(Wn, float):
            self.Wn = Wn / (fs / 2)
        elif isinstance(Wn, Sequence):
            self.Wn = [wn / (fs / 2) for wn in Wn]
        else:
            raise TypeError(
                f"Wn must be a float or a sequence of floats, got {type(Wn)}."
            )
        self.fs = fs
        self.order = order
        self.btype = btype

        result = butter(self.order, self.Wn, btype=self.btype)
        if result is None or len(result) != 2:
            raise ValueError("Butter function did not return expected coefficients.")
        self.b, self.a = result

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> np.ndarray:
        super().__call__(x)
        if self.roll():
            x = filtfilt(self.b, self.a, x, axis=0).copy()
        return x, *args
