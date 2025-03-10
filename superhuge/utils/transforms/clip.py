import numpy as np
from .abc import Transform


class Clipping(Transform):
    """Clips EEG values to remove extreme artifacts."""

    def __init__(self, percentile: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.percentile = percentile

    def __call__(self, x: np.ndarray) -> np.ndarray:
        low_bound = np.percentile(x, self.percentile // 2, axis=0)
        high_bound = np.percentile(x, 100 - self.percentile // 2, axis=0)
        return np.clip(x, low_bound, high_bound)
