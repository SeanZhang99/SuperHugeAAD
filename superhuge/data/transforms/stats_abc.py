from __future__ import annotations
from abc import abstractmethod
from typing import Any, Optional, Sequence
import numpy as np
from .abc import Transform


class StatisticalTransform(Transform):
    """
    Base class for transforms that require accumulation and fitting.
    Provides standardized `update`, `fit`, and `stat` property interfaces.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stat: np.ndarray | Sequence[float] | None
        self._fitted: bool = False

    @abstractmethod
    def update(self, x: np.ndarray) -> None:
        """
        Update internal statistics from a single sample.
        Subclasses should implement their own logic.
        """
        ...

    @abstractmethod
    def fit(self) -> None:
        """
        Finalize statistics using accumulated values.
        """
        ...

    @property
    def stat(self) -> Any:
        """
        Return the current fitted statistic (e.g., mean, cov matrix).
        """
        if not self._fitted or self._stat is None:
            raise RuntimeError("Statistic not fitted yet. Call `fit()` first.")
        return self._stat

    @stat.setter
    def stat(self, value: Any) -> None:
        """
        Manually set the statistic (e.g., from a training run).
        """
        self._stat = value
        self._fitted = True

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Transform not yet fitted. Call `update()` with your samples, and then call `fit()`, or set the `stat` attribute manually."
            )

    def reset(self) -> None:
        """
        Reset the internal state of the transform.
        Useful for reusing the same instance with new data.
        """
        self._stat = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """
        Check if the transform has been fitted.
        """
        return self._fitted

    @abstractmethod
    def __call__(self, x: np.ndarray, /, *args, **kwargs):
        """
        Apply the transform to the input data.
        Subclasses should implement their own logic.
        """
        self._check_fitted()
        return super().__call__(x, *args, **kwargs)
