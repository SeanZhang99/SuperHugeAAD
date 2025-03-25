from pydantic import GetCoreSchemaHandler, BaseModel, field_validator, validator
from pydantic_core import core_schema
from abc import ABC, abstractmethod
import numpy as np
from collections.abc import Sequence


class TransformConfig(BaseModel):
    when: str = "before_returning"
    whom: str | Sequence[str] = "eeg"

    @field_validator("when")
    def validate_when(cls, value):
        valid_options = ["before_slicing", "before_returning"]
        if value not in valid_options:
            raise ValueError(f"when must be one of {valid_options}, but got {value}.")
        return value

    @field_validator("whom")
    def validate_whom(cls, value):
        valid_options = ["eeg", "audio", "label", "all"]
        if isinstance(value, str):
            if value not in valid_options:
                raise ValueError(
                    f"whom must be one of {valid_options}, but got {value}."
                )
            value = [value]
        elif isinstance(value, Sequence):
            for v in value:
                if v not in valid_options or v == "all":
                    raise ValueError(
                        f"whom must be one of {valid_options.copy().remove("all")}, but got {v}."
                    )
        else:
            raise ValueError(
                f"whom must be a string or a sequence, but got {type(value).__name__}."
            )
        return value


class Transform(ABC):
    """Abstract base class for EEG transforms in superhuge package."""

    def __init__(
        self,
        /,
        *,
        when: str = "before_returning",
        whom: str | Sequence[str] = "eeg",
        **kwargs,
    ) -> None:
        """
        Args:
            when (str | None, optional): When to apply this transform.
                Options:
                - `before_slicing`: Before segmenting EEG
                - `before_returning`: Before returning EEG from dataset
                Defaults to `before_returning`.
            whom (str | Sequence[str] | None, optional): whom this transform is applied to.
                Options:
                - "eeg": Apply to EEG data
                - "audio": Apply to audio data
                - "label": Apply to label
                - A sequence of combinatin of above options
                - "all": Apply to all


            **kwargs: Additional parameters for subclasses.
        """
        config = TransformConfig(when=when, whom=whom)
        self.when = config.when
        self.whom = config.whom
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
