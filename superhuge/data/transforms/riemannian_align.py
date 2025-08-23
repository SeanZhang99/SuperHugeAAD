from typing import Any
from .stats_abc import StatisticalTransform
import numpy as np


class RiemannianAlign(StatisticalTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cov_matrices: list[np.ndarray] = []
        self._n_channels: int | None = None

    def update(self, x: np.ndarray) -> None:
        if x.shape[0] < x.shape[1]:
            x = x.T
        # assert time by channel
        assert x.ndim == 2, f"Input must be a 2D ndarray but got {x.ndim}D"
        assert (
            x.shape[0] > x.shape[1]
        ), f"Input must be time first, but seems to be channel first: {x.shape}."
        if self._n_channels is None:
            self._n_channels = x.shape[1]
        assert x.shape[1] == self._n_channels
        cov = np.cov(x, bias=True, rowvar=False) + np.eye(self._n_channels) * 1e-6
        assert cov.shape == (self._n_channels, self._n_channels), (
            f"Covariance matrix shape mismatch: expected ({self._n_channels}, {self._n_channels}), "
            f"but got {cov.shape}."
        )
        self._cov_matrices.append(cov)

    def fit(self):
        from pyriemann.utils.mean import mean_riemann
        from pyriemann.utils.base import invsqrtm

        mean_cov = mean_riemann(np.stack(self._cov_matrices, axis=0))
        self._stat = invsqrtm(mean_cov)
        self._fitted = True

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x, *args)[0]
        return (self._stat @ x.T).T, *args
