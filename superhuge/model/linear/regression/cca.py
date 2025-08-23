from einops import rearrange
import torch
import pydantic
from typing import Any
from .abc import LinearABC  # Importing the abstract base class
from ...types import EEG_TYPE, AUDIO_TYPE


class CCAConfig(pydantic.BaseModel):
    x_lag_sec: float
    y_lag_sec: float
    fs: int
    l2: float
    num_features_x: int
    num_features_y: int
    num_components: int

    # These will be calculated internally

    @pydantic.field_validator(
        "x_lag_sec",
        "y_lag_sec",
        "l2",
        "fs",
        "num_features_x",
        "num_features_y",
        "num_components",
    )
    def positive_float(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @pydantic.computed_field
    @property
    def x_lag_samples(self) -> int:
        """Convert x_lag_sec to samples using sampling frequency"""
        return int(self.x_lag_sec * self.fs)

    @pydantic.computed_field
    @property
    def y_lag_samples(self) -> int:
        """Convert y_lag_sec to samples using sampling frequency"""
        return int(self.y_lag_sec * self.fs)


class CCA(LinearABC):
    Rxyxy: torch.Tensor
    weight_x: torch.Tensor
    weight_y: torch.Tensor

    def __init__(
        self,
        /,
        *,
        x_lag_sec: float,
        y_lag_sec: float,
        l2: float = 0.0,
        num_components: int,
        **kwargs,
    ):
        """
        Canonical Correlation Analysis model with temporal filtering

        Args:
            x_lag_sec: Pre-stimulus lag (seconds)
            y_lag_sec: Post-stimulus lag (seconds)
            fs: Sampling frequency (Hz)
            l2: L2 regularization strength
            num_features_x: Number of input features (channels)
            num_features_y: Number of output features
            num_components: Number of CCA components to extract
        """
        super().__init__()

        # Create validated configuration
        self.cfg = CCAConfig(
            x_lag_sec=x_lag_sec,
            y_lag_sec=y_lag_sec,
            fs=kwargs["fs"],
            l2=l2,
            num_features_x=kwargs["num_channels"],
            num_features_y=kwargs["num_audio_features"],
            num_components=num_components,
        )
        self.covar_dim_x = (self.cfg.x_lag_samples * 2 + 1) * self.cfg.num_features_x
        self.covar_dim_y = (self.cfg.y_lag_samples * 2 + 1) * self.cfg.num_features_y

        covar_dim = self.covar_dim_x + self.covar_dim_y
        self.register_buffer("Rxyxy", torch.zeros((covar_dim, covar_dim)))
        self.register_buffer(
            "weight_x", torch.zeros((self.covar_dim_x, self.cfg.num_components))
        )
        self.register_buffer(
            "weight_y", torch.zeros((self.covar_dim_y, self.cfg.num_components))
        )

    def update(self, eeg: torch.Tensor, audio: torch.Tensor) -> None:
        """
        Accumulate cross-correlation statistics

        Args:
            x: Input features [batch, time, features_x]
            y: Target features [batch, time, features_y]
        """
        super().update(eeg, audio)  # Update sample count

        # Create and flatten lagged matrices
        x_lag_flat = self.lag_and_flatten(
            eeg,
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.x_lag_samples,
            self.cfg.x_lag_samples,
        )
        y_lag_flat = self.lag_and_flatten(
            audio[..., 0],
            "batch lag time channel -> (batch time) (lag channel)",
            self.cfg.y_lag_samples,
            self.cfg.y_lag_samples,
        )

        # Combine into joint tensor
        joint_signal = torch.cat([x_lag_flat, y_lag_flat], dim=-1)

        # Update cross-covariance matrix
        self.Rxyxy += joint_signal.T @ joint_signal

    def fit(self) -> None:
        """
        Compute canonical components using accumulated statistics
        """
        assert not self._fitted, "Model is already fitted"
        assert self._n_samples > 0, "No data to fit"

        # Solve CCA on accumulated statistics

        self.Rxyxy += (
            torch.eye(self.Rxyxy.shape[0], device=self.Rxyxy.device) * self.cfg.l2
        )

        self.weight_x, self.weight_y = canonical_correlation_analysis(
            cov_matrix=self.Rxyxy / self._n_samples,
            num_features_x=(self.cfg.x_lag_samples * 2 + 1) * self.cfg.num_features_x,
            reg_strength=1.0e-5,
            num_components=self.cfg.num_components,
        )

        # Mark model as fitted
        self._fitted = True

    def predict(
        self, eeg: EEG_TYPE, audio: AUDIO_TYPE  # Preserve base class interface
    ) -> tuple[EEG_TYPE, AUDIO_TYPE]:
        """
        Project input onto CCA space

        Args:
            x: EEG features [batch, time, features_x]
            audio: [batch, time, features_y, speaker]

        Returns:
            x_proj: Projected input features [batch, time, num_components]
            y_proj: Projected target features [batch, time, num_components, speaker]
        """
        assert self._fitted, "Model not fitted yet"

        # Create and flatten lagged matrices for both inputs
        x_lag_flat = self.lag_and_flatten(
            eeg,
            "batch lag time features_x -> batch time (lag features_x)",
            self.cfg.x_lag_samples,
            self.cfg.x_lag_samples,
        )
        y_lag_flat = self.lag_and_flatten(
            audio,
            "batch lag time features_y speaker -> batch time speaker (lag features_y)",
            self.cfg.y_lag_samples,
            self.cfg.y_lag_samples,
        )

        # Project using learned weights
        x_proj = x_lag_flat @ self.weight_x
        y_proj = y_lag_flat @ self.weight_y

        y_proj = rearrange(
            y_proj, "batch time speaker features_y -> batch time features_y speaker"
        )

        return x_proj, y_proj


# Helper functions (outside class)
def eigen_decomposition(cov_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sorted eigenvalues/vectors of covariance matrix

    Args:
        cov_matrix: [n, n] covariance matrix

    Returns:
        eigenvalues: [n] in descending order
        eigenvectors: [n, n] corresponding eigenvectors
    """
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    descending = torch.argsort(eigvals, descending=True)
    return eigvals[descending], eigvecs[:, descending]


def compute_sphering_matrix(
    cov_matrix: torch.Tensor, reg_strength: float = 1e-12
) -> torch.Tensor:
    """
    Compute whitening transformation for covariance matrix

    Args:
        cov_matrix: [n, n] covariance matrix
        reg_strength: Regularization cutoff threshold

    Returns:
        transform: [n, m] whitening matrix (m <= n)
    """
    eigvals, eigvecs = eigen_decomposition(cov_matrix)

    # Regularization (discard components below threshold)
    max_eigval = eigvals[0]
    keep_mask = eigvals / max_eigval > reg_strength
    eigvals = eigvals[keep_mask]
    eigvecs = eigvecs[:, keep_mask]

    # Whitening transformation (X → Λ^{-1/2}Uᵀ)
    return eigvecs @ torch.diag(torch.sqrt(1.0 / eigvals))


def canonical_correlation_analysis(
    cov_matrix: torch.Tensor,
    num_features_x: int,
    reg_strength: float = 1e-12,
    num_components: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform Canonical Correlation Analysis (CCA)

    Reference: Based on implementation from 10.1016/j.neuroimage.2018.01.033

    Args:
        cov_matrix: [d, d] joint covariance matrix (x and y concatenated)
        num_features_x: Dimension of x features
        reg_strength: Truncation threshold

    Returns:
        weights_x: [d_x, k] projection weights for x
        weights_y: [d_y, k] projection weights for y
    """
    # Split covariance matrix
    d_x = num_features_x

    cov_xx = cov_matrix[:d_x, :d_x]
    cov_yy = cov_matrix[d_x:, d_x:]
    cov_xy = cov_matrix[:d_x, d_x:]

    # Compute whitening transforms
    w_x = compute_sphering_matrix(cov_xx, reg_strength)
    w_y = compute_sphering_matrix(cov_yy, reg_strength)

    # Compute transformed cross-covariance
    transformed_cov = w_x.T @ cov_xy @ w_y

    # SVD of cross-covariance matrix (max k = min(rank_x, rank_y))
    u, s, vh = torch.linalg.svd(transformed_cov, full_matrices=False)
    k = min(u.shape[1], vh.shape[0], s.shape[0])
    assert (
        k >= num_components
    ), f"Number of components {num_components} exceeds rank {k} of covariance matrix"
    k = num_components

    # Compute canonical vectors
    weights_x = w_x @ u[:, :k] * torch.sqrt(torch.tensor(2.0))
    weights_y = w_y @ vh[:k, :].T * torch.sqrt(torch.tensor(2.0))

    return weights_x, weights_y
