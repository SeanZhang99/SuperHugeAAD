from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np


class Transform(ABC):
    """Abstract base class for EEG transforms in superhuge package."""

    def __init__(self, /, *, apply_on: str | None = None, **kwargs) -> None:
        """
        Args:
            apply_on (str | None, optional): When to apply this transform.
                Options:
                - `before_slicing`: Before segmenting EEG
                - `before_returning`: Before returning EEG from dataset
                Defaults to `before_returning`.
            **kwargs: Additional parameters for subclasses.
        """
        self.apply_on = apply_on
        super().__init__()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply transformation to EEG data.

        Args:
            x (np.ndarray): Input EEG data (shape: T × C).

        Returns:
            np.ndarray: Transformed EEG data.
        """
        assert x.ndim == 2, f"Input must be a 2D ndarray but got {x.ndim}"
        assert (
            x.shape[1] < x.shape[0]
        ), f"Input must be time first, but seems to be channel first: {x.shape}."

    def __repr__(self):
        """Generic representation for all subclasses."""
        params = ", ".join(
            f"{k}={v if not isinstance(v, (Sequence,np.ndarray)) else type(v)}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({params})"
