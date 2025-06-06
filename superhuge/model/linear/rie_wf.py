from collections.abc import Sequence
from einops import rearrange
import torch
import pydantic
from .abc import LinearABC


class RiemannianWienerFilterConfig(pydantic.BaseModel):
    pre_lag: float = pydantic.Field(..., gt=0)
    post_lag: float = pydantic.Field(..., gt=0)
    l2: float = pydantic.Field(0.0, ge=0)
    fs: int = pydantic.Field(..., gt=0)
    num_channels: int = pydantic.Field(..., gt=0)

    pre_lag_samples: int | None = None
    post_lag_samples: int | None = None
    nlag: int | None = None

    @pydantic.model_validator(mode="after")
    def compute_lag_samples(self) -> "RiemannianWienerFilterConfig":
        self.pre_lag_samples = int(self.pre_lag * self.fs)
        self.post_lag_samples = int(self.post_lag * self.fs)
        self.nlag = self.pre_lag_samples + self.post_lag_samples + 1
        return self


class RiemannianWienerFilter(LinearABC):
    log_cov_sum: torch.Tensor
    rxy_accum: torch.Tensor
    weights: torch.Tensor
    mean_cov: torch.Tensor
    whitening_matrix: torch.Tensor
    _n_samples: int = 0

    def __init__(self, *, pre_lag: float, post_lag: float, l2: float, **kwargs):
        super().__init__()
        self.cfg = RiemannianWienerFilterConfig(
            pre_lag=pre_lag,
            post_lag=post_lag,
            l2=l2,
            fs=kwargs["fs"],
            num_channels=kwargs["num_channels"],
        )

        d = self.cfg.nlag * self.cfg.num_channels

        self.register_buffer("log_cov_sum", torch.zeros(d, d))
        self.register_buffer("rxy_accum", torch.zeros(d, 1))
        self.register_buffer("weights", torch.zeros(d, 1))
        self.register_buffer("mean_cov", torch.eye(d))
        self.register_buffer("whitening_matrix", torch.eye(d))

    def sym_logm(self, M: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals = torch.clamp(eigvals, min=1e-6)
        log_eigvals = torch.log(eigvals)
        return eigvecs @ torch.diag(log_eigvals) @ eigvecs.T

    def sym_expm(self, M: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(M)
        exp_eigvals = torch.exp(eigvals)
        return eigvecs @ torch.diag(exp_eigvals) @ eigvecs.T

    def inv_sqrtm(self, cov: torch.Tensor) -> torch.Tensor:
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inv_sqrt_eigvals = 1.0 / torch.sqrt(torch.clamp(eigvals, min=1e-6))
        return eigvecs @ torch.diag(inv_sqrt_eigvals) @ eigvecs.T

    def update(self, eeg: torch.Tensor, audio: torch.Tensor) -> None:
        super().update(eeg, audio)

        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.pre_lag_samples,
            self.cfg.post_lag_samples,
        )

        self._n_samples += x_lag.shape[0]

        cov = (x_lag.T @ x_lag) / x_lag.shape[0]
        log_cov = self.sym_logm(cov)
        self.log_cov_sum += log_cov

        y = rearrange(audio[..., 0], "batch time features -> (batch time) features")
        self.rxy_accum += x_lag.T @ y

    def fit(self) -> None:
        assert not self._fitted, "Model already fitted"
        assert self._n_samples > 0, "No data for fitting"

        log_mean_cov = self.log_cov_sum / self._n_samples
        self.mean_cov = self.sym_expm(log_mean_cov)
        self.whitening_matrix = self.inv_sqrtm(self.mean_cov)

        R_reg = self.mean_cov + self.cfg.l2 * torch.eye(
            self.mean_cov.shape[0], device=self.mean_cov.device
        )
        R_reg = self.whitening_matrix @ R_reg @ self.whitening_matrix.T
        tangent_space = self.sym_logm(R_reg)

        rxy_normalized = self.whitening_matrix @ self.rxy_accum / self._n_samples
        self.weights = torch.linalg.solve(tangent_space, rxy_normalized).detach()
        self._fitted = True

    def predict(self, eeg: torch.Tensor, audio: torch.Tensor) -> Sequence[torch.Tensor]:
        assert self._fitted, "Model not fitted yet."

        x_lag = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> batch time (lag channel)",
            self.cfg.pre_lag_samples,
            self.cfg.post_lag_samples,
        )

        y_pred = x_lag @ self.whitening_matrix @ self.weights
        return y_pred, audio
