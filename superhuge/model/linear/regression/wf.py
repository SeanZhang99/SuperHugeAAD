from typing import Any
from einops import rearrange
import numpy as np
import torch
import pydantic

from .lwcov import lwcov
from .abc import LinearABC
from ...types import EEG_TYPE, AUDIO_TYPE


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
    Rxx: torch.Tensor
    rxy: torch.Tensor
    _weights: torch.Tensor

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
            "Rxx",
            torch.zeros(
                self.cfg.nlag * self.cfg.num_channels,
                self.cfg.nlag * self.cfg.num_channels,
            ),
        )

        self.register_buffer(
            "rxy",
            torch.zeros(self.cfg.nlag * self.cfg.num_channels, 1),
        )

        self.x_list = []
        self.y_list = []

    def update(self, eeg: EEG_TYPE, env: AUDIO_TYPE) -> None:
        """
        Update the model with new data.
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
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y = rearrange(
            env[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )
        self.x_list.append(x_lag)
        self.y_list.append(y)

    def fit(self):
        """
        Fit the model to the data.
        """
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."
        x_all = torch.cat(self.x_list, dim=0)
        if self._use_lw_cov:
            self.Rxx = lwcov(x_all, detect_orientation=True) / self._n_samples  # type: ignore
        else:
            self.Rxx = (x_all.mT @ x_all) / self._n_samples
            self.Rxx += self.cfg.l2 * torch.eye(
                self.cfg.nlag * self.cfg.num_channels, device=self.Rxx.device
            )
        self.rxy = x_all.mT @ torch.cat(self.y_list, dim=0) / self._n_samples
        self._weights = torch.linalg.solve(self.Rxx, self.rxy).detach()
        self._fitted = True
        self.x_list.clear()
        self.y_list.clear()

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
            "batch lag time channel -> batch time (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y_pred = x_lag @ self.weights
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
