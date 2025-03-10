import numpy as np
from .abc import Transform


class TimeShift(Transform):
    """Randomly shifts EEG signals along the time axis."""

    def __init__(
        self, max_shift: int | float = 200, fs: int | None = None, **kwargs
    ) -> None:
        kwargs.setdefault("apply_on", "before_slicing")
        super().__init__(**kwargs)
        if isinstance(max_shift, float) and fs is None:
            raise ValueError("Sampling frequency must be provided for float max_shift.")
        self.max_shift = (
            int(max_shift * fs)
            if isinstance(max_shift, float) and fs is not None
            else int(max_shift)
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = np.random.randint(-self.max_shift, self.max_shift)
        return np.roll(x, shift, axis=0)
