from typing import Any
import numpy as np

from ..metadata_processing.data import MetadataElement
from .abc import Transform


class ChannelSelection(Transform):
    """Selects a subset of EEG channels based on provided indices."""

    def __init__(
        self,
        /,
        *,
        channel_indices: list[int] | None = None,
        channel_names: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.channel_indices = channel_indices
        self.channel_names = channel_names

        assert (self.channel_indices is not None) ^ (
            self.channel_names is not None
        ), "Either channel_indices or channel_names must be provided, but not both."

    def __call__(self, x: np.ndarray, /, **kwargs) -> dict[str, np.ndarray]:
        super().__call__(x)
        if self.channel_indices is None:
            meta: MetadataElement = kwargs.get("meta", {})
            channel_infos = getattr(meta, "channel_infos", {})
            assert (
                isinstance(channel_infos, dict) and channel_infos
            ), "Channel information is required in metadata."
            for idx, name_dict in channel_infos.items():
                if name_dict.get("name") in self.channel_names:
                    if self.channel_indices is None:
                        self.channel_indices = []
                    self.channel_indices.append(idx - 1)
        assert self.channel_indices, "No valid channel indices were found."
        return {"x": x[..., self.channel_indices]}

    @property
    def num_channels(self) -> int:
        """Number of selected channels."""
        if self.channel_indices is not None:
            return len(self.channel_indices)
        elif self.channel_names is not None:
            return len(self.channel_names)
        else:
            raise ValueError("Channel indices or names must be provided.")
