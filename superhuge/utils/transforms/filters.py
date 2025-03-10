import numpy as np
from scipy.signal import butter, filtfilt
from .abc import Transform


class BandpassFilter(Transform):
    """Applies a bandpass filter to EEG data."""

    def __init__(self, /, *, lowcut=1, highcut=50, fs=250, order=4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        nyquist = 0.5 * self.fs
        result = butter(
            self.order, [self.lowcut / nyquist, self.highcut / nyquist], btype="band"
        )
        if result is None or len(result) != 2:
            raise ValueError("Butter function did not return expected coefficients.")
        self.b, self.a = result

    def __call__(self, x: np.ndarray) -> np.ndarray:
        super().__call__(x)

        return filtfilt(self.b, self.a, x, axis=0)
