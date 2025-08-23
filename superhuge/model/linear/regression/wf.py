from typing import Any
from einops import rearrange
import torch
import pydantic
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
    weights: torch.Tensor

    def __init__(self, /, *, pre_lag: float, post_lag: float, l2: float, **kwargs):
        super().__init__()
        self.cfg = WienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            l2=l2,
            **kwargs,
        )

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

    def update(self, eeg: EEG_TYPE, audio: AUDIO_TYPE) -> None:
        """
        Update the model with new data.
        """
        super().update(eeg, audio)
        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y = rearrange(
            audio[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )
        self.Rxx += x_lag.T @ x_lag
        self.rxy += x_lag.T @ y

    def fit(self):
        """
        Fit the model to the data.
        """
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."
        self.Rxx /= self._n_samples
        self.rxy /= self._n_samples
        self.Rxx += self.cfg.l2 * torch.eye(
            self.cfg.nlag * self.cfg.num_channels, device=self.Rxx.device
        )
        self.weights = torch.linalg.solve(self.Rxx, self.rxy).detach()
        self._fitted = True

    def predict(self, eeg: EEG_TYPE, audio: AUDIO_TYPE) -> tuple[EEG_TYPE, AUDIO_TYPE]:
        """
        Predict the output based on the input data.
        """
        assert self._fitted, "Model is not fitted yet."
        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> batch time (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y_pred = x_lag @ self.weights
        return y_pred, audio
