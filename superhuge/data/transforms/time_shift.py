from typing import Any
import numpy as np

from .abc import Transform


class TimeShift(Transform):
    """Randomly shifts EEG signals along the time axis."""

    def __init__(
        self, /, *, max_shift: int | float = 200, fs: int | None = None, **kwargs
    ) -> None:
        """
        Args:
            max_shift (int | float): Maximum shift in samples or seconds.
                If float, `fs` must be provided.
            fs (int | None): Sampling frequency. Required if `max_shift` is a float.
            kwargs: Additional arguments for the Transform class.
        """
        kwargs.setdefault("when", "before_slicing")
        super().__init__(**kwargs)
        if isinstance(max_shift, float) and fs is None:
            raise ValueError("Sampling frequency must be provided for float max_shift.")
        self.max_shift = (
            int(max_shift * fs)
            if isinstance(max_shift, float) and fs is not None
            else int(max_shift)
        )

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x, *args, **kwargs)
        if self.roll():
            shift = np.random.randint(-self.max_shift, self.max_shift)
            if shift >= 0:
                return np.pad(x[shift:], ((0, shift), (0, 0))).copy(), *args
            else:
                return np.pad(x[:shift], ((-shift, 0), (0, 0))).copy(), *args
        return x, *args
