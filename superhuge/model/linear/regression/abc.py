from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, final
import torch
from einops import rearrange

import superhuge

from memory_profiler import profile


class LinearABC(torch.nn.Module, ABC):
    _fitted: bool = False
    _n_samples: int = 0
    _x_lag_buf: torch.Tensor

    @abstractmethod
    def __init__(self, /, **kwargs):
        super().__init__()

    @abstractmethod
    def update(self, eeg: torch.Tensor, env: torch.Tensor) -> None:
        """
        Update the model with new data.
        """
        self._n_samples += eeg.shape[0] * eeg.shape[1]

    @final
    def forward(
        self, eeg: torch.Tensor, env: "torch.Tensor"
    ) -> tuple["superhuge.model.types.EEG_TYPE", "superhuge.model.types.AUDIO_TYPE"]:
        """
        Forward pass of the model.
        """
        if self._fitted:
            eeg, env = self.predict(eeg, env)
        elif self.training:
            self.update(eeg, env)
        return eeg, env

    @abstractmethod
    def fit(self) -> None:
        """
        Fit the model to the data.
        """
        ...

    @abstractmethod
    def predict(self, eeg: torch.Tensor, env: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Predict the output based on the input data.
        """
        ...

    @final
    def get_lag_mtx(self, x: torch.Tensor, pre_lag: int, post_lag: int):
        """
        Fill a single pre-allocated buffer with lagged copies of x.
        Returns view — zero allocation per call.

        Buffer layout: (batch, time, nlag, ch_flat), contiguous.
        This layout means merge patterns like
            "batch time lag channel -> (batch time) (lag channel)"
        are pure views (both batch-time and lag-channel are adjacent in memory).

        Old layout was (batch, nlag, time, ...) — that layout required
        a copy for every einops rearrange because batch and time were
        separated by the nlag dimension.
        """
        batch, time, *rest = x.shape
        nlag = pre_lag + post_lag + 1

        need_shape = (batch, time, nlag, *rest)
        if not hasattr(self, "_x_lag_buf") or self._x_lag_buf.shape != need_shape:
            self.register_buffer(
                "_x_lag_buf",
                torch.empty(*need_shape, device=x.device),
            )

        buf = self._x_lag_buf
        buf.zero_()

        for lag_idx, shift in enumerate(range(-pre_lag, post_lag + 1)):
            if shift < 0:
                buf[:, -shift:, lag_idx, :] = x[:, : time + shift, ...]
            elif shift > 0:
                buf[:, : time - shift, lag_idx, :] = x[:, shift:, ...]
            else:
                # unresolved problem: memory leak of this line.
                buf[:, :, lag_idx, :].copy_(x.detach())

        return buf  # (batch, time, nlag, ...) — view

    @final
    def get_lag_mtx_old(self, x: torch.Tensor, *lag: int, **kwargs):
        """
        Construct a lagged matrix for the input with given lag.

        Parameters:
        x: torch.Tensor, input tensor. Shape: (batch_size, time_steps, ...). The lag (or advance) is operated along the time dimension, and lagged signals is inseted into the second dimension. Other dimensions remain unchanged.
        pre_lag: int, number of time steps to advance. The mathematical formula is y(t) = sum_n w(n)*x(t-n)
        post_lag: int, number of time steps to lag. The mathematical formula is y(t) = sum_n w(n)*x(t+n)

        Example:
        lag: 10, 10. return: (batch_size, 21, time_steps, ...). 10 means the input is advanced by 10 time steps, 10 means the input is lagged by 10 time steps.
        """
        x_lag = []
        pre_lag, post_lag = lag
        if pre_lag < 0:
            pre_lag = -pre_lag

        assert pre_lag >= 0, "pre_lag must be non-negative"
        assert post_lag >= 0, "post_lag must be non-negative"

        for l in range(-pre_lag, post_lag + 1):
            if l <= 0:
                x_lag.append(
                    torch.cat(
                        [
                            torch.zeros((x.shape[0], -l, *x.shape[2:])).type_as(x),
                            x[:, : x.shape[1] + l],
                        ],
                        dim=1,
                    )
                )
            elif l > 0:
                x_lag.append(
                    torch.cat(
                        [
                            x[:, l:, :],
                            torch.zeros(x.shape[0], l, *x.shape[2:]).type_as(x),
                        ],
                        dim=1,
                    )
                )
        x_lag = torch.stack(
            x_lag,
            dim=1,
        )
        return x_lag

    def lag_and_flatten(
        self, signal: torch.Tensor, pattern: str, *lag_samples: int
    ) -> torch.Tensor:
        """
        Create lagged matrix and flatten it.

        Args:
            signal: Input tensor [batch, time, features]
            lag_samples: Number of lag samples
            num_features: Number of features

        Returns:
            Flattened lagged matrix [batch * time, lag * features]
        """
        lagged_matrix = self.get_lag_mtx(signal, lag_samples[0], lag_samples[1])
        return rearrange(
            lagged_matrix,
            pattern,
        )
