from typing import Any
from .stats_abc import StatisticalTransform
import numpy as np


class ZScoreAlign(StatisticalTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0
        self._n_channels = None

    def update(self, x: np.ndarray) -> None:
        if x.shape[0] < x.shape[1]:
            x = x.T
        if self._n_channels is None:
            self._n_channels = x.shape[0]
        assert x.shape[0] == self._n_channels
        self._sum += x.sum(axis=1)
        self._sum_sq += (x**2).sum(axis=1)
        self._count += x.shape[1]

    def fit(self):
        mean = self._sum / self._count
        var = (self._sum_sq / self._count) - (mean**2)
        std = np.sqrt(var + 1e-6)
        self._stat = (mean, std)
        self._fitted = True

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x, *args, **kwargs)[0]
        assert self._fitted, "ZScoreAlign must be fitted before calling."
        assert self._stat is not None, "Statistical parameters are not set."
        mean, std = self._stat
        return (x - mean) / std, *args
