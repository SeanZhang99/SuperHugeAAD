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

    def __call__(self, x: np.ndarray, **kwargs) -> dict[str, np.ndarray]:
        super().__call__(x, **kwargs)
        eps = kwargs.get("eps", 1.0e-5)
        x -= x.mean(axis=0, keepdims=True)
        x /= x.std(axis=0, keepdims=True) + eps
        return {"x": x}
