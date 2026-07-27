"""
Per-subject Wiener Filter
=========================

Variant of WienerFilter that independently computes Rxx (with optional
Ledoit-Wolf shrinkage or l2 regularization) **per subject**, then averages
the per-subject Rxx matrices before solving for the decoder weights.

Key difference from ``wf.WienerFilter``:
  - Original: a single global Rxx is estimated from all data pooled together.
  - This version: Rxx is estimated separately for each subject, then averaged
    across subjects.  Subjects are identified by a **composite key**
    ``(dataset_id, subject_id)``, so the same subject_id in different
    datasets is treated as distinct entities.

Usage
-----
Identical interface to ``WienerFilter``.  ``update()`` **must** receive
``meta`` containing **both** ``"subject_id"`` and ``"dataset_id"``
(1-D arrays/tensors of length ``batch``).  The composite key
``(dataset_id, subject_id)`` disambiguates subjects across datasets.

    model = PerSubjectWienerFilter(pre_lag=0.1, post_lag=0.9, l2=1.0,
                                   use_lwcov=False, fs=64, num_channels=64,
                                   window_length=512)
    for batch in dataloader:
        model.update(batch.eeg, batch.env, meta=batch.meta)
    model.fit()
    pred = model.predict(batch.eeg, batch.env)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from einops import rearrange

from .abc import LinearABC
from ...types import EEG_TYPE, AUDIO_TYPE

# ---------------------------------------------------------------------------
# Reuse the same config as the original WienerFilter
# ---------------------------------------------------------------------------
from .wf import WienerFilterConfig


class _SubjectStats:
    """Per-subject sufficient statistics needed for Rxx estimation."""

    __slots__ = ("n", "sum_xx", "sum_x", "sum_normsq", "sum_x4", "sum_wx")

    def __init__(self, p: int, device: torch.device, dtype: torch.dtype):
        self.n: int = 0
        self.sum_xx = torch.zeros(p, p, device=device, dtype=dtype)
        self.sum_x = torch.zeros(p, device=device, dtype=dtype)
        self.sum_normsq = torch.tensor(0.0, device=device, dtype=dtype)
        self.sum_x4 = torch.tensor(0.0, device=device, dtype=dtype)
        self.sum_wx = torch.zeros(p, device=device, dtype=dtype)


class PerSubjectWienerFilter(LinearABC):
    """Wiener filter with per-subject Rxx estimation and cross-subject averaging.

    Parameters
    ----------
    pre_lag : float
        Pre-stimulus lag in seconds.
    post_lag : float
        Post-stimulus lag in seconds.
    l2 : float
        l2-regularisation strength (used when ``use_lwcov=False``).
    use_lwcov : bool
        If ``True``, apply Ledoit-Wolf shrinkage per subject instead of
        a fixed l2 penalty.
    **kwargs
        Forwarded to ``WienerFilterConfig`` (e.g. ``fs``, ``num_channels``,
        ``window_length``).
    """

    _weights: torch.Tensor
    _sum_xy: torch.Tensor
    _use_lw_cov: bool
    _per_subject: dict[Any, _SubjectStats]

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

        p = self.cfg.nlag * self.cfg.num_channels
        self.register_buffer("weights", torch.zeros(p, 1))
        self.register_buffer("_sum_xy", torch.zeros(p, 1))

        # Lazy-initialised per-subject statistics
        self._per_subject: dict[Any, _SubjectStats] = {}
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # update()
    # ------------------------------------------------------------------
    def update(self, eeg: EEG_TYPE, env: AUDIO_TYPE, **kwargs) -> None:
        """Accumulate global rxy and per-subject Rxx statistics.

        ``kwargs["meta"]`` **must** contain both ``"subject_id"`` and
        ``"dataset_id"`` — 1-D arrays/tensors of length ``batch``.
        The composite key ``(dataset_id[b], subject_id[b])`` uniquely
        identifies each subject across multiple datasets.
        """
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg)
        if isinstance(env, np.ndarray):
            env = torch.from_numpy(env)
        super().update(eeg, env, **kwargs)

        B, T = eeg.shape[0], eeg.shape[1]
        p = self.cfg.nlag * self.cfg.num_channels

        # ---- build lagged design matrix ---------------------------------
        x_lag = self.lag_and_flatten(
            eeg,
            "batch time lag channel -> (batch time) (lag channel)",
            self.cfg.pre_lag,  # type: ignore[arg-type]
            self.cfg.post_lag,  # type: ignore[arg-type]
        )
        y = rearrange(
            env[..., 0],
            "batch time num_features -> (batch time) num_features",
            num_features=1,
        )

        # Cache device / dtype from the first call
        if self._device is None:
            self._device = x_lag.device
            self._dtype = x_lag.dtype

        # ---- global cross-covariance ------------------------------------
        self._sum_xy += x_lag.mT @ y

        # ---- per-subject auto-covariance statistics --------------------
        meta = kwargs.get("meta", {})
        composite_keys = self._resolve_composite_keys(meta, B)

        for b in range(B):
            key = composite_keys[b]

            # Initialise per-subject stats on first encounter
            if key not in self._per_subject:
                self._per_subject[key] = _SubjectStats(
                    p, self._device, self._dtype  # type: ignore[arg-type]
                )

            start = b * T
            end = (b + 1) * T
            x_b = x_lag[start:end]  # (T, p)

            s = self._per_subject[key]
            s.n += T
            s.sum_xx += x_b.mT @ x_b
            s.sum_x += x_b.sum(dim=0)

            # Fourth-moment accumulators for sample-based Ledoit-Wolf
            norms2 = (x_b**2).sum(dim=1)  # (T,)
            s.sum_normsq += norms2.sum()
            s.sum_x4 += (norms2**2).sum()
            s.sum_wx += (x_b * norms2.unsqueeze(1)).sum(dim=0)

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------
    def fit(self) -> None:
        """Compute per-subject Rxx, average across subjects, then solve."""
        assert not self._fitted, "Model is already fitted."
        assert self._n_samples > 0, "No data to fit the model."
        assert len(self._per_subject) > 0, (
            "No subjects seen during update() — was meta missing subject_id?"
        )

        p = self.cfg.nlag * self.cfg.num_channels
        device = self._device or self._sum_xy.device
        dtype = self._dtype or self._sum_xy.dtype

        # ---- compute Rxx for each subject independently -----------------
        Rxx_list: list[torch.Tensor] = []
        for sid, s in self._per_subject.items():
            n_s = s.n
            if n_s < 1:
                continue

            if self._use_lw_cov:
                Rxx_s = self._compute_lw_cov(s, p, device, dtype)
            else:
                Rxx_s = s.sum_xx / n_s
                Rxx_s = Rxx_s + self.cfg.l2 * torch.eye(
                    p, device=device, dtype=dtype
                )

            Rxx_list.append(Rxx_s)

        if not Rxx_list:
            raise RuntimeError(
                "No subject had enough samples to compute Rxx."
            )

        # ---- average Rxx across subjects --------------------------------
        self._Rxx_avg: torch.Tensor = torch.stack(Rxx_list, dim=0).mean(dim=0)

        # ---- solve for decoder weights ----------------------------------
        n_total = self._n_samples
        self._weights = torch.linalg.solve(
            self._Rxx_avg, self._sum_xy / n_total
        ).detach()
        self._fitted = True

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------
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
            self.cfg.pre_lag,  # type: ignore[arg-type]
            self.cfg.post_lag,  # type: ignore[arg-type]
        )
        y_pred = x_lag @ self._weights
        return y_pred, env

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @weights.setter
    def weights(self, value: torch.Tensor) -> None:
        self._weights = value

    @property
    def Rxx_avg(self) -> torch.Tensor | None:
        """Averaged Rxx across all subjects (available after ``fit()``)."""
        return getattr(self, "_Rxx_avg", None)

    @property
    def num_subjects(self) -> int:
        """Number of unique subjects seen during ``update()``."""
        return len(self._per_subject)

    @property
    def subject_ids(self) -> list[Any]:
        """List of all unique (dataset_id, subject_id) composite keys."""
        return list(self._per_subject.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_composite_keys(
        meta: dict[str, Any], B: int
    ) -> list[Any]:
        """Build composite keys ``(dataset_id, subject_id)`` for each batch element.

        Both ``meta["dataset_id"]`` and ``meta["subject_id"]`` are 1-D
        arrays/tensors of length ``B``.
        """
        ds_raw = meta["dataset_id"]
        sid_raw = meta["subject_id"]

        ds_arr = np.asarray(
            ds_raw.detach().cpu().numpy() if isinstance(ds_raw, torch.Tensor) else ds_raw
        )
        sid_arr = np.asarray(
            sid_raw.detach().cpu().numpy() if isinstance(sid_raw, torch.Tensor) else sid_raw
        )

        assert ds_arr.ndim == 1 and ds_arr.shape[0] == B, (
            f"dataset_id must be 1-D of length {B}, got {ds_arr.shape}"
        )
        assert sid_arr.ndim == 1 and sid_arr.shape[0] == B, (
            f"subject_id must be 1-D of length {B}, got {sid_arr.shape}"
        )

        return [
            (_to_key(ds_arr[b]), _to_key(sid_arr[b])) for b in range(B)
        ]

    @staticmethod
    def _compute_lw_cov(
        s: _SubjectStats, p: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Ledoit-Wolf shrinkage covariance for a single subject."""
        n = s.n
        assert n > 1, (
            f"At least two observations are needed for LW covariance, "
            f"got n={n}"
        )

        mu = s.sum_x / n
        S = (s.sum_xx - n * torch.outer(mu, mu)) / (n - 1)

        mu_norm_sq = torch.dot(mu, mu)
        sum_z4 = (
            s.sum_x4
            + 4.0 * (mu @ (s.sum_xx @ mu))
            - 4.0 * torch.dot(s.sum_wx, mu)
            + 2.0 * mu_norm_sq * s.sum_normsq
            - 3.0 * n * mu_norm_sq**2
        )

        tr_S = torch.trace(S)
        m_val = tr_S / p
        eye = torch.eye(p, device=device, dtype=dtype)
        d2 = torch.norm(S - m_val * eye, p="fro") ** 2 / p

        tr_S2 = torch.trace(S @ S)
        bbar2 = (sum_z4 + (2.0 - n) * tr_S2) / (p * n * n)
        b2 = torch.minimum(bbar2, d2)
        a2 = d2 - b2

        Rxx_s = (b2 / d2) * m_val * eye + (a2 / d2) * S
        return Rxx_s


def _to_key(val: Any) -> Any:
    """Convert a scalar subject identifier to a hashable key."""
    if isinstance(val, np.ndarray):
        return val.item()
    if isinstance(val, torch.Tensor):
        return val.item()
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val
