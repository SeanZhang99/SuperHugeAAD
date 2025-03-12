from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Sequence


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
        assert apply_on in [
            None,
            "before_slicing",
            "before_returning",
        ], f"apply_on must be one of `before_slicing`, `before_returning` or `None`, but got {apply_on}."
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
            f"{k}={v if not isinstance(v, (list,tuple,dict,np.ndarray)) else type(v)}"
            for k, v in self.__dict__.items()
        )
        return f"{self.__class__.__name__}({params})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: object) -> "Transform":
        if not isinstance(value, Transform):
            raise TypeError(
                f"Expected an instance of Transform, got {type(value).__name__}"
            )
        return value
