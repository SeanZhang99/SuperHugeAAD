from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, final
import torch
from einops import rearrange

import superhuge



class LinearABC(torch.nn.Module, ABC):
    _fitted: bool = False
    _n_samples: int = 0

    @abstractmethod
    def __init__(self, /, **kwargs):
        super().__init__()

    @abstractmethod
    def update(self, eeg: torch.Tensor, audio: torch.Tensor) -> None:
        """
        Update the model with new data.
        """
        self._n_samples += eeg.shape[0] * eeg.shape[1]

    @final
    def forward(
        self, eeg: torch.Tensor, audio: "torch.Tensor"
    ) -> tuple["superhuge.model.types.EEG_TYPE", "superhuge.model.types.AUDIO_TYPE"]:
        """
        Forward pass of the model.
        """
        if self._fitted:
            eeg, audio = self.predict(eeg, audio)
        elif self.training:
            self.update(eeg, audio)
        return eeg, audio

    @abstractmethod
    def fit(self) -> None:
        """
        Fit the model to the data.
        """
        ...

    @abstractmethod
    def predict(self, eeg: torch.Tensor, audio: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        Predict the output based on the input data.
        """
        ...

    @final
    def get_lag_mtx(self, x: torch.Tensor, *lag: int, **kwargs):
        """
        Construct a lagged matrix for the input with given lag.

        Parameters:
        x: torch.Tensor, input tensor. Shape: (batch_size, time_steps, ...). The lag (or advance) is operated along the time dimension, and lagged signals is inseted into the second dimension. Other dimensions remain unchanged.
        lag: two integers, the lag range. The first element specify where the lag begins, the second element specify where the lag ends. The first lag indicates advance, and the second indicates delay.

        Example:
        lag: 10, 10. return: (batch_size, 21, time_steps, ...). 10 means the input is advanced by 10 time steps, 10 means the input is lagged by 10 time steps.
        """
        x_lag = []
        assert (
            isinstance(lag, Sequence) and len(lag) == 2
        ), "lag must be a sequence of two integers"

        if lag[0] <= 0 and lag[1] >= 0:
            lag = (-lag[0], lag[1])

        for l in range(-lag[0], lag[1] + 1):
            if l >= 0:
                x_lag.append(
                    torch.cat(
                        [
                            torch.zeros((x.shape[0], l, *x.shape[2:])).type_as(x),
                            x[:, : x.shape[1] - l],
                        ],
                        dim=1,
                    )
                )
            elif l < 0:
                x_lag.append(
                    torch.cat(
                        [
                            x[:, -l:, :],
                            torch.zeros(x.shape[0], -l, *x.shape[2:]).type_as(x),
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
            Flattened lagged matrix [batch*time, lag*features]
        """
        lagged_matrix = self.get_lag_mtx(signal, lag_samples[0], lag_samples[1])
        return rearrange(
            lagged_matrix,
            pattern,
        )
