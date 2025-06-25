import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import scipy.signal
from pydantic import BaseModel, Field

from .abc import LinearABC


# === Helper: FIR Filterbank ===
def make_dyadic_fir_filterbank(
    fs: int, n_filters: int, min_len: int = 2, max_len: int = 128
):
    lengths = np.geomspace(min_len, max_len, num=n_filters).astype(int)
    filters = []
    for L in lengths:
        nyq = fs / 2
        low = 0.5 / (L / fs)
        high = nyq * 0.9
        b = scipy.signal.firwin(L, [low, high], pass_zero=False, fs=fs)
        filters.append(torch.tensor(b, dtype=torch.float32))
    return filters


class FIRFilterbank(nn.Module):
    def __init__(self, filters: list[torch.Tensor]):
        super().__init__()
        self.n_filters = len(filters)
        for i, filt in enumerate(filters):
            self.register_buffer(f"filter_{i}", filt.view(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b t c -> b c t")
        outputs = []
        for i in range(self.n_filters):
            filt: torch.Tensor = getattr(self, f"filter_{i}")
            xi = F.conv1d(
                x, filt.expand(x.size(1), 1, -1), padding="same", groups=x.size(1)
            )
            outputs.append(xi)
        out = torch.cat(outputs, dim=1)  # [b, c*f, t]
        return rearrange(out, "b c t -> b t c")


# === PCA and CCA utilities ===
def eigen_decomposition(cov_matrix: torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    idx = torch.argsort(eigvals, descending=True)
    return eigvals[idx], eigvecs[:, idx]


def compute_sphering_matrix(cov_matrix: torch.Tensor, reg_strength: float = 1e-12):
    eigvals, eigvecs = eigen_decomposition(cov_matrix)
    max_eigval = eigvals[0]
    keep = eigvals / max_eigval > reg_strength
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    return eigvecs @ torch.diag(torch.sqrt(1.0 / eigvals))


def canonical_correlation_analysis(
    cov_matrix: torch.Tensor,
    num_features_x: int,
    reg_strength: float,
    num_components: int,
):
    d_x = num_features_x
    cov_xx = cov_matrix[:d_x, :d_x]
    cov_yy = cov_matrix[d_x:, d_x:]
    cov_xy = cov_matrix[:d_x, d_x:]

    w_x = compute_sphering_matrix(cov_xx, reg_strength)
    w_y = compute_sphering_matrix(cov_yy, reg_strength)

    T = w_x.T @ cov_xy @ w_y
    u, s, vh = torch.linalg.svd(T, full_matrices=False)
    k = min(num_components, u.shape[1], vh.shape[0])

    weights_x = w_x @ u[:, :k] * torch.sqrt(torch.tensor(2.0))
    weights_y = w_y @ vh[:k, :].T * torch.sqrt(torch.tensor(2.0))

    return weights_x, weights_y


# === Config ===
class FilterbankCCAConfig(BaseModel):
    fs: int = Field(..., gt=0)
    impulse_response_lengths: list[int]
    num_components: int
    num_channels: int
    num_audio_features: int


# === Main model ===
class FilterbankCCA(LinearABC):
    Rxyxy: torch.Tensor
    weight_x: torch.Tensor
    weight_y: torch.Tensor

    def __init__(
        self,
        *,
        fs,
        impulse_response_lengths,
        num_components,
        num_channels,
        num_audio_features,
    ):
        super().__init__()
        self.cfg = FilterbankCCAConfig(
            fs=fs,
            impulse_response_lengths=impulse_response_lengths,
            num_components=num_components,
            num_channels=num_channels,
            num_audio_features=num_audio_features,
        )

        filters = make_dyadic_fir_filterbank(
            fs,
            len(impulse_response_lengths),
            min(impulse_response_lengths),
            max(impulse_response_lengths),
        )
        self.filterbank = FIRFilterbank(filters)

        feat_x = num_channels * len(impulse_response_lengths)
        feat_y = num_audio_features * len(impulse_response_lengths)
        self.register_buffer("Rxyxy", torch.zeros((feat_x + feat_y, feat_x + feat_y)))
        self.register_buffer("weight_x", torch.zeros((feat_x, num_components)))
        self.register_buffer("weight_y", torch.zeros((feat_y, num_components)))

    def update(self, eeg: torch.Tensor, audio: torch.Tensor):
        eeg_filt = self.filterbank(eeg)
        audio = rearrange(audio, "b t f s -> b t (f s)")
        audio_filt = self.filterbank(audio)

        x = rearrange(eeg_filt, "b t f -> (b t) f")
        y = rearrange(audio_filt, "b t f -> (b t) f")
        joint = torch.cat([x, y], dim=1)
        self.Rxyxy += joint.T @ joint

    def fit(self):
        assert not self._fitted
        cov = self.Rxyxy / self._n_samples

        d_x = self.cfg.num_channels * len(self.cfg.impulse_response_lengths)
        self.weight_x, self.weight_y = canonical_correlation_analysis(
            cov_matrix=cov,
            num_features_x=d_x,
            reg_strength=1e-5,
            num_components=self.cfg.num_components,
        )

        self._fitted = True

    def predict(self, eeg: torch.Tensor, audio: torch.Tensor):
        assert self._fitted

        eeg_filt = self.filterbank(eeg)
        audio = rearrange(audio, "b t f s -> b t (f s)")
        audio_filt = self.filterbank(audio)

        x = rearrange(eeg_filt, "b t f -> (b t) f")
        y = rearrange(audio_filt, "b t f -> (b t) f")

        x_proj = x @ self.weight_x
        y_proj = y @ self.weight_y

        return x_proj, y_proj
