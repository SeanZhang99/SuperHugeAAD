from einops import rearrange
import einops
import torch
import pydantic
from .abc import LinearABC


class WienerFilterConfig(pydantic.BaseModel):
    pre_lag: float | int | None = None
    post_lag: float | int | None = None
    lags: list | None = None
    l2: float
    fs: int
    num_features: int
    num_channels: int

    @pydantic.field_validator("pre_lag", "post_lag", "l2", "fs", "num_channels")
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
        if self.lags is not None:
            return int(self.lags[0] * self.fs)
        return -self.pre_lag  # pre_lag already converted to samples in model_post_init

    @property
    def lag_end_samples(self) -> int:
        if self.lags is not None:
            return int(self.lags[1] * self.fs)
        return self.post_lag  # post_lag already converted to samples in model_post_init

    @pydantic.computed_field
    @property
    def nlag(self) -> int:
        return self.lag_end_samples - self.lag_start_samples + 1

    def model_post_init(self, __context):
        # pydantic v2 runs model_post_init BEFORE model_validator(after), so the
        # lag-mode checks must be enforced here explicitly.
        self.check_lag_mode()
        if self.lags is None:
            self.pre_lag = int(self.pre_lag * self.fs)
            self.post_lag = int(self.post_lag * self.fs)


class WienerFilter(LinearABC):
    Rxx: torch.Tensor
    Rxy: torch.Tensor
    weights: torch.Tensor

    def __init__(self, /, *, pre_lag: float | None = None,
                 post_lag: float | None = None, lags: list | None = None,
                 l2: float, **kwargs):
        super().__init__()
        self.cfg = WienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            lags=lags,
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

    def update(self, eeg: torch.Tensor, env: torch.Tensor) -> None:
        """
        Update the model with new data.
        """
        super().update(eeg, env)
        x_lag = self.lag_and_flatten(
            env[..., 0],
            "batch time lag channel -> (batch time) (lag channel)",
            self.cfg.lag_start_samples,  # type: ignore
            self.cfg.lag_end_samples,  # type: ignore
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

    def predict(self, eeg, env):
        """
        Predict the output based on the input data.
        """
        assert self._fitted, "Model is not fitted yet."
        x_lag = self.lag_and_flatten(
            env,
            "batch time lag feature spekaer -> batch time (lag channel) speaker",
            self.cfg.lag_start_samples,  # type: ignore
            self.cfg.lag_end_samples,  # type: ignore
        )
        y_pred: torch.Tensor = einops.einsum(
            "batch time (lag channel) speaker, (lag channel) channel -> batch time channel speaker",
            x_lag,
            self.weights,  # type: ignore
        )
        return eeg, y_pred
