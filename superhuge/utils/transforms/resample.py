import numpy as np
from scipy.signal import resample
from .abc import Transform


class Resample(Transform):
    """Reduces EEG sampling rate to improve efficiency."""

    def __init__(self, /, *, old_fs: int, new_fs: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.old_fs = old_fs
        self.new_fs = new_fs

    def __call__(self, x: np.ndarray) -> np.ndarray:
        num_samples = int(x.shape[0] * self.new_fs / self.old_fs)
        return resample(x, num_samples, axis=0)  # type: ignore
