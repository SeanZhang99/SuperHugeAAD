from typing import Any
from einops import einsum, rearrange
import numpy as np
import torch
import pydantic

from .lwcov import lwcov_from_cov, lwcov
from .abc import LinearABC
from ...types import EEG_TYPE, AUDIO_TYPE

import gc, psutil, os

from memory_profiler import profile


class WienerFilterConfig(pydantic.BaseModel, extra="allow"):
    pre_lag: float | int
    post_lag: float | int
    l2: float
    fs: int
    window_length: int
    num_channels: int

    @pydantic.field_validator(
        "pre_lag", "post_lag", "l2", "fs", "window_length", "num_channels"
    )
    def positive_float(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @pydantic.computed_field
    @property
    def nlag(self) -> int:
        assert isinstance(self.pre_lag, int) and isinstance(self.post_lag, int)
        return self.pre_lag + self.post_lag + 1

    def model_post_init(self, __context: Any) -> None:
        self.pre_lag = int(self.pre_lag * self.fs)
        self.post_lag = int(self.post_lag * self.fs)
        return super().model_post_init(__context)


class WienerFilter(LinearABC):
    _weights: torch.Tensor
    _sum_xx: torch.Tensor
    _sum_xy: torch.Tensor
    _sum_x: torch.Tensor

    def __init__(
        self,
        /,
        *,
        pre_lag: float,
        post_lag: float,
        l2: float,
        use_lwcov: bool,
        **kwargs,
    ):
        super().__init__()
        self.cfg = WienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            l2=l2,
            **kwargs,
        )

        self._use_lw_cov = use_lwcov

        self.register_buffer(
            "weights", torch.zeros(self.cfg.nlag * self.cfg.num_channels, 1)
        )

        self.register_buffer(
            "_sum_xx",
            torch.zeros(
                self.cfg.nlag * self.cfg.num_channels,
                self.cfg.nlag * self.cfg.num_channels,
            ),
        )
        self.register_buffer(
            "_sum_xy",
            torch.zeros(self.cfg.nlag * self.cfg.num_channels, 1),
        )
        self.register_buffer(
            "_sum_x",
            torch.zeros(self.cfg.nlag * self.cfg.num_channels),
        )
        self.x_list = []

    def update(self, eeg: EEG_TYPE, env: AUDIO_TYPE) -> None:
        """
        Update the model with new data (online accumulation).
        eeg: batch, time, channel
        env: batch, time, feature, speaker
        """
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env)
        super().update(eeg, env)
        x_lag = self.lag_and_flatten(
            eeg,
            "batch time lag channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )

        y = rearrange(
            env[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )
        # Online accumulation — avoid storing all samples
        self._sum_xx += x_lag.mT @ x_lag  # (p, p)
        self._sum_xy += x_lag.mT @ y  # (p, 1)
        self._sum_x += x_lag.sum(dim=0)  # (p,)
        self.x_list.append(x_lag)
        del x_lag, y

        gc.collect()

    def fit(self):
        """
        Fit the model from accumulated running sums (online covariance).
        """
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."

        n = self._n_samples

        if self._use_lw_cov:
            assert n > 1, "At least two observations are needed for LW covariance"
            # Compute sample covariance with mean subtraction
            mu = self._sum_x / n  # (p,)
            S = (self._sum_xx - n * torch.outer(mu, mu)) / (n - 1)
            Rxx = lwcov_from_cov(S, n)  # type: ignore
        else:
            Rxx = self._sum_xx
            Rxx += self.cfg.l2 * torch.eye(
                self.cfg.nlag * self.cfg.num_channels, device=Rxx.device
            )

        rxy = self._sum_xy
        self._weights = torch.linalg.solve(Rxx, rxy).detach()
        self._fitted = True

        # Release running-sum memory now that fitting is complete
        self._sum_xx.zero_()
        self._sum_xy.zero_()
        self._sum_x.zero_()

    def predict(self, eeg: EEG_TYPE, env: AUDIO_TYPE) -> tuple[EEG_TYPE, AUDIO_TYPE]:
        """
        Predict the output based on the input data.
        """
        assert self._fitted, "Model is not fitted yet."
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env)
        x_lag = self.lag_and_flatten(
            eeg,
            "batch time lag channel -> batch time (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y_pred = x_lag @ self._weights
        return y_pred, env

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: torch.Tensor):
        self._weights = value


def _main():
    wf = WienerFilter(
        pre_lag=0.25,
        post_lag=0.4,
        l2=1.0,
        fs=64,
        window_length=60,
        num_channels=32,
        use_lwcov=True,
    )
    x = torch.randn(1, 128, 2)
    x_lagged = wf.get_lag_mtx(x, 16, 0)
    print(x_lagged)
    np.save("x_lagged.npy", x_lagged.numpy())
