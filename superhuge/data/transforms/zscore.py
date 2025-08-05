from typing import Any
import numpy as np
from .abc import Transform


class ZScore(Transform):
    """Z-score the input data."""

    def __init__(self, /, **kwargs) -> None:
        """
        Args:
            **kwargs: Additional parameters for subclasses.
        """

        super().__init__(**kwargs)

    def __call__(
        self, x: np.ndarray, /, *args, eps=1.0e-5, **kwargs
    ) -> tuple[np.ndarray, Any]:
        super().__call__(x)
        x -= x.mean(axis=0, keepdims=True)
        x /= x.std(axis=0, keepdims=True) + eps
        return x, *args
