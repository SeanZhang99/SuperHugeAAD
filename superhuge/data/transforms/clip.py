from typing import Any
import numpy as np

from .abc import Transform


class Clipper(Transform):
    """Clips EEG values to remove extreme artifacts."""

    def __init__(
        self,
        /,
        *,
        bound: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert bound is not None or (
            lower_bound is not None and upper_bound is not None
        ), "Either `bound` or (`lower_bound` and `upper_bound`) must be provided."
        if bound is not None:
            self.lower_bound = -bound
            self.upper_bound = bound
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    def __call__(self, x: np.ndarray, /, *args, **kwargs) -> tuple[np.ndarray, Any]:
        super().__call__(x)
        if self.roll():
            x = np.clip(x, a_min=self.lower_bound, a_max=self.upper_bound)
        return x, *args
