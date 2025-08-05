from einops import rearrange
import einops
import torch
import pydantic
from .abc import LinearABC


class WienerFilterConfig(pydantic.BaseModel):
    pre_lag: float | int
    post_lag: float | int
    l2: float
    fs: int
    num_features: int
    num_channels: int

    @pydantic.field_validator("pre_lag", "post_lag", "l2", "fs", "num_channels")
    def positive_float(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @pydantic.computed_field
    @property
    def nlag(self) -> int:
        assert isinstance(self.pre_lag, int) and isinstance(self.post_lag, int)
        return self.pre_lag + self.post_lag + 1

    def model_post_init(self, __context):
        self.pre_lag = int(self.pre_lag * self.fs)
        self.post_lag = int(self.post_lag * self.fs)


class WienerFilter(LinearABC):
    Rxx: torch.Tensor
    Rxy: torch.Tensor
    weights: torch.Tensor

    def __init__(self, /, *, pre_lag: float, post_lag: float, l2: float, **kwargs):
        super().__init__()
        self.cfg = WienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            l2=l2,
            fs=kwargs["fs"],
            num_channels=kwargs["num_channels"],
            num_features=kwargs.get("num_features", 1),
        )

        self.register_buffer(
            "weights", torch.zeros(self.cfg.nlag * self.cfg.num_features, 1)
        )

        self.register_buffer(
            "Rxx",
            torch.zeros(
                self.cfg.nlag * self.cfg.num_features,
                self.cfg.nlag * self.cfg.num_features,
            ),
        )

        self.register_buffer(
            "Rxy",
            torch.zeros(self.cfg.nlag * self.cfg.num_features, self.cfg.num_channels),
        )

    def update(self, eeg: torch.Tensor, audio: torch.Tensor) -> None:
        """
        Update the model with new data.
        """
        super().update(eeg, audio)
        x_lag = self.lag_and_flatten(
            audio[..., 0],
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y = rearrange(
            eeg,
            "batch time num_channels -> (batch time) num_channels",
            num_channels=self.cfg.num_channels,
        )
        self.Rxx += x_lag.T @ x_lag
        self.Rxy += x_lag.T @ y

    def fit(self):
        """
        Fit the model to the data.
        """
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."
        self.Rxx /= self._n_samples
        self.Rxy /= self._n_samples
        self.Rxx += self.cfg.l2 * torch.eye(
            self.cfg.nlag * self.cfg.num_channels, device=self.Rxx.device
        )
        self.weights = torch.linalg.solve(self.Rxx, self.Rxy).detach()
        self._fitted = True

    def predict(self, eeg, audio):
        """
        Predict the output based on the input data.
        """
        assert self._fitted, "Model is not fitted yet."
        x_lag = self.lag_and_flatten(
            audio,
            "batch lag time feature spekaer -> batch time (lag channel) speaker",
            self.cfg.pre_lag,  # type: ignore
            self.cfg.post_lag,  # type: ignore
        )
        y_pred: torch.Tensor = einops.einsum(
            "batch time (lag channel) speaker, (lag channel) channel -> batch time channel speaker",
            x_lag,
            self.weights,  # type: ignore
        )
        return eeg, y_pred
