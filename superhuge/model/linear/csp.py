import numpy as np
import scipy
import torch
from einops import rearrange
from pydantic import BaseModel, field_validator
from .abc import LinearABC


class CommonSpatialMappingFilterConfig(BaseModel):
    pre_lag: float
    post_lag: float
    fs: int
    num_channels: int
    nlag: int | None = None

    @field_validator("pre_lag", "post_lag", "fs", "num_channels")
    def non_negative(cls, v):
        if v < 0:
            raise ValueError("All parameters must be non-negative")
        return v

    def model_post_init(self, __context):
        self.pre_lag = int(self.pre_lag * self.fs)
        self.post_lag = int(self.post_lag * self.fs)
        self.nlag = self.pre_lag + self.post_lag + 1


class CommonSpatialMappingFilter(LinearABC):
    def __init__(
        self,
        /,
        *,
        pre_lag: float,
        post_lag: float,
        fs: int,
        **kwargs,
    ):
        super().__init__()
        self.cfg = CommonSpatialMappingFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            fs=fs,
            num_channels=kwargs["num_channels"],
        )

        feature_dim = self.cfg.nlag * self.cfg.num_channels

        self.register_buffer("Rxx", torch.zeros(feature_dim, feature_dim))
        self.register_buffer("rxd", torch.zeros(feature_dim, 1))  # Δr = E[x(t) d(t)]
        self.register_buffer("weights", torch.zeros(feature_dim, 1))

    def update(self, eeg: torch.Tensor, audio: torch.Tensor) -> None:
        """
        audio: [batch, time, feature=1, speaker=N]
               where audio[..., 0] is attended, audio[..., 1:] are distractors
        """
        super().update(eeg, audio)

        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,
            self.cfg.post_lag,
        )

        # Assumption: feature = 1
        s_att = audio[..., 0]  # [batch, time, 1]
        s_unatt = audio[..., 1:]  # [batch, time, 1, K]
        s_unatt_mean = s_unatt.mean(dim=-1)  # [batch, time, 1]

        d = s_att - s_unatt_mean  # [batch, time, 1]
        d = rearrange(d, "b t 1 -> (b t) 1")  # flatten to [BT, 1]

        self.Rxx += x_lag.T @ x_lag
        self.rxd += x_lag.T @ d

    def fit(self):
        assert not self._fitted, "Model already fitted."
        assert self._n_samples > 0, "No training data."

        self.Rxx /= self._n_samples - 1
        self.rxd /= self._n_samples - 1

        self.Rxx += torch.eye(self.Rxx.shape[0], device=self.Rxx.device) * 1e-3

        # Construct Rdd = Δr Δrᵀ
        Rdd = self.rxd @ self.rxd.T

        if self.Rxx.device.type == "cuda":
            Rdd_np = Rdd.detach().cpu().numpy()
            Rxx_np = self.Rxx.detach().cpu().numpy()
        else:
            Rdd_np = Rdd.numpy()
            Rxx_np = self.Rxx.numpy()

        # Solve generalized eigenvalue problem: Rdd w = λ Rxx w
        eigvals, eigvecs = scipy.linalg.eigh(Rdd_np, Rxx_np)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        max_idx = np.argmax(eigvals)
        w = eigvecs[:, max_idx]
        w = (
            torch.from_numpy(w)
            .to(dtype=torch.float32, device=self.Rxx.device)
            .unsqueeze(1)
        )

        # Normalize: wᵀ Rxx w = 1
        w_norm = w / torch.sqrt(w.T @ self.Rxx @ w + 1e-6)

        assert not torch.isnan(w_norm).any(), "Weights contain NaN values."

        self.weights = w_norm.detach()
        self._fitted = True

    def predict(self, eeg: torch.Tensor, audio: torch.Tensor):
        assert self._fitted, "Model is not fitted yet."

        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> batch time (lag channel)",
            self.cfg.pre_lag,
            self.cfg.post_lag,
        )
        y_pred = x_lag @ self.weights
        assert not torch.isnan(y_pred).any(), "Predictions contain NaN values."
        assert not torch.isinf(audio).any(), "Predictions contain infinite values."
        return y_pred, audio
