from dataclasses import dataclass
from typing import Any

import numpy as np
import pydantic
import torch
from einops import rearrange

from .abc import LinearABC
from ...types import EEG_TYPE, AUDIO_TYPE


class WienerFilterConfig(pydantic.BaseModel, extra="allow"):
    pre_lag: float | int | None = None
    post_lag: float | int | None = None
    lags: list | None = None
    l2: float
    fs: int
    window_length: int
    num_channels: int

    @pydantic.field_validator(
        "pre_lag", "post_lag", "l2", "fs", "window_length", "num_channels"
    )
    def positive_float(cls, v):
        if v is None:
            return v
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @pydantic.model_validator(mode="after")
    def check_lag_mode(self):
        has_pre = self.pre_lag is not None or self.post_lag is not None
        has_lags = self.lags is not None
        if has_pre and has_lags:
            raise ValueError(
                "pre_lag/post_lag and lags are mutually exclusive: choose one lag mode"
            )
        if has_pre and (self.pre_lag is None or self.post_lag is None):
            raise ValueError("pre_lag and post_lag must be given together")
        if not has_pre and not has_lags:
            raise ValueError("must specify either pre_lag/post_lag or lags")
        if has_lags:
            if (
                not isinstance(self.lags, (list, tuple))
                or len(self.lags) != 2
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in self.lags
                )
            ):
                raise ValueError(
                    "lags must be a [start, end] pair of seconds values (start may be negative)"
                )
            if not self.lags[1] > self.lags[0]:
                raise ValueError("lags end must be greater than start")
        return self

    @property
    def lag_start_samples(self) -> int:
        """Left edge of the lag window in samples (negative = past, i.e. causal)."""
        if self.lags is not None:
            return int(self.lags[0] * self.fs)
        return -self.pre_lag  # pre_lag already converted to samples in model_post_init

    @property
    def lag_end_samples(self) -> int:
        """Right edge of the lag window in samples (positive = future, non-causal)."""
        if self.lags is not None:
            return int(self.lags[1] * self.fs)
        return self.post_lag  # post_lag already converted to samples in model_post_init

    @pydantic.computed_field
    @property
    def nlag(self) -> int:
        return self.lag_end_samples - self.lag_start_samples + 1

    def model_post_init(self, __context: Any) -> None:
        # pydantic v2 runs model_post_init BEFORE model_validator(after), so the
        # lag-mode checks must be enforced here explicitly (validator stays as a
        # guard for rebuild/copy paths).
        self.check_lag_mode()
        # Legacy mode: convert pre_lag/post_lag seconds -> samples in place,
        # preserving the historic behaviour of these fields. lags mode leaves
        # them None (samples are derived via lag_start_samples/lag_end_samples).
        if self.lags is None:
            self.pre_lag = int(self.pre_lag * self.fs)
            self.post_lag = int(self.post_lag * self.fs)
        return super().model_post_init(__context)


@dataclass
class WienerFilterState:
    """Running sufficient statistics for online Wiener filter estimation.

    Accumulates:
        sum_xx     : Σ x_k x_k^T        (p, p)
        sum_xy     : Σ x_k y_k          (p, 1)
        sum_x      : Σ x_k              (p,)
        sum_normsq : Σ ||x_k||^2        (scalar)
        sum_x4     : Σ ||x_k||^4        (scalar)
        sum_wx     : Σ ||x_k||^2 x_k    (p,)
        Rxx:    Covariance matrix       (p, p)
        rxy:    Cross-covariance vector (p, 1)
    """

    sum_xx: torch.Tensor
    sum_xy: torch.Tensor
    sum_x: torch.Tensor
    sum_normsq: torch.Tensor
    sum_x4: torch.Tensor
    sum_wx: torch.Tensor
    Rxx: torch.Tensor
    rxy: torch.Tensor

    def __init__(self, p: int):
        self.sum_xx = torch.zeros(p, p)
        self.sum_xy = torch.zeros(p, 1)
        self.sum_x = torch.zeros(p)
        self.sum_normsq = torch.tensor(0.0)
        self.sum_x4 = torch.tensor(0.0)
        self.sum_wx = torch.zeros(p)
        self.Rxx = torch.zeros(p, p)
        self.rxy = torch.zeros(p, 1)

    def zero_(self) -> "WienerFilterState":
        self.sum_xx.zero_()
        self.sum_xy.zero_()
        self.sum_x.zero_()
        self.sum_normsq.zero_()
        self.sum_x4.zero_()
        self.sum_wx.zero_()
        self.Rxx.zero_()
        self.rxy.zero_()
        return self


class WienerFilter(LinearABC):
    weights: torch.Tensor
    _state: WienerFilterState
    _use_lw_cov: bool

    def __init__(
        self,
        /,
        *,
        pre_lag: float | None = None,
        post_lag: float | None = None,
        lags: list | None = None,
        l2: float,
        use_lwcov: bool,
        **kwargs,
    ):
        super().__init__()
        self.cfg = WienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            lags=lags,
            l2=l2,
            **kwargs,
        )
        self._use_lw_cov = use_lwcov

        p = self.cfg.nlag * self.cfg.num_channels

        self.register_buffer("weights", torch.zeros(p, 1))

        state = WienerFilterState(p)
        for name, tensor in state.__dict__.items():
            self.register_buffer(f"_{name}", tensor)
        self._state = state

    def update(self, eeg: EEG_TYPE, env: AUDIO_TYPE, **kwargs) -> None:
        """Update running statistics with a new batch."""
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env)
        super().update(eeg, env, **kwargs)

        x_lag = self.lag_and_flatten(
            eeg,
            "batch time lag channel -> (batch time) (lag channel)",
            self.cfg.lag_start_samples,  # type: ignore
            self.cfg.lag_end_samples,  # type: ignore
        )
        y = rearrange(
            env[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )

        self._state.sum_xx += x_lag.mT @ x_lag
        self._state.sum_xy += x_lag.mT @ y
        self._state.sum_x += x_lag.sum(dim=0)

        # Fourth-moment accumulators for sample-based Ledoit-Wolf shrinkage.
        norms2 = (x_lag**2).sum(dim=1)
        self._state.sum_normsq += norms2.sum()
        self._state.sum_x4 += (norms2**2).sum()
        self._state.sum_wx += (x_lag * norms2.unsqueeze(1)).sum(dim=0)

    def fit(self) -> None:
        """Fit weights from running statistics."""
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."

        n = self._n_samples
        p = self._state.sum_x.shape[0]

        if self._use_lw_cov:
            assert n > 1, "At least two observations are needed for LW covariance"
            mu = self._state.sum_x / n
            S = (self._state.sum_xx - n * torch.outer(mu, mu)) / (n - 1)

            mu_norm_sq = torch.dot(mu, mu)
            sum_z4 = (
                self._state.sum_x4
                + 4.0 * (mu @ (self._state.sum_xx @ mu))
                - 4.0 * torch.dot(self._state.sum_wx, mu)
                + 2.0 * mu_norm_sq * self._state.sum_normsq
                - 3.0 * n * mu_norm_sq**2
            )

            tr_S = torch.trace(S)
            m = tr_S / p
            eye = torch.eye(p, device=S.device, dtype=S.dtype)
            d2 = torch.norm(S - m * eye, p="fro") ** 2 / p

            tr_S2 = torch.trace(S @ S)
            bbar2 = (sum_z4 + (2.0 - n) * tr_S2) / (p * n * n)
            b2 = torch.minimum(bbar2, d2)
            a2 = d2 - b2

            self._state.Rxx = (b2 / d2) * m * eye + (a2 / d2) * S
        else:
            self._state.Rxx = self._state.sum_xx / n
            self._state.Rxx += self.cfg.l2 * torch.eye(p, device=self._state.Rxx.device)

        self.weights = torch.linalg.solve(
            self._state.Rxx, self._state.sum_xy / n
        ).detach()
        self._fitted = True

        # Release running-sum memory now that fitting is complete.
        # self._state.zero_()

    def predict(self, eeg: EEG_TYPE, env: AUDIO_TYPE) -> tuple[EEG_TYPE, AUDIO_TYPE]:
        """Predict the output based on the input data."""
        assert self._fitted, "Model is not fitted yet."
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env)
        x_lag = self.lag_and_flatten(
            eeg,
            "batch time lag channel -> batch time (lag channel)",
            self.cfg.lag_start_samples,  # type: ignore
            self.cfg.lag_end_samples,  # type: ignore
        )
        y_pred = x_lag @ self.weights
        return y_pred, env
