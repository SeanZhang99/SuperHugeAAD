import numpy as np
from .abc import Transform
from ..channel_enum import CHANNEL1D_ENUM


class ChannelRearrangeTransform(Transform):
    """
    A transform class to rearrange EEG channels based on CHANNEL1D_ENUM.
    """

    def __init__(
        self, /, *, apply_on: str | None = "before_returning", **kwargs
    ) -> None:
        """
        Args:
            apply_on (str | None, optional): When to apply this transform.
                Defaults to `before_returning`.
            **kwargs: Additional parameters for subclasses.
        """
        super().__init__(apply_on=apply_on, **kwargs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Rearrange EEG channels based on CHANNEL1D_ENUM.

        Args:
            x (np.ndarray): Input EEG data (shape: T × C).

        Returns:
            np.ndarray: Rearranged EEG data (shape: T × len(CHANNEL1D_ENUM)).
        """
        super().__call__(x)  # Validate input shape and type

        # Initialize output array with zeros
        y = np.zeros((x.shape[0], len(CHANNEL1D_ENUM)), dtype=x.dtype)

        # Rearrange channels based on CHANNEL1D_ENUM

        rearranged_idx = []
        original_idx = []
        for i, channel_name in enumerate(CHANNEL1D_ENUM):
            if channel_name.name in CHANNEL1D_ENUM:
                rearranged_idx.append(channel_name.value)
                original_idx.append(i)
        y[:, rearranged_idx] = x[:, original_idx]

        return y
