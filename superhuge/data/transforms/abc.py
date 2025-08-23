from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema


class TransformConfig(BaseModel):
    seed: int = 42
    apply_prob: float = 0.5
    when: Literal["before_slicing", "before_returning"] = "before_returning"
    whom: (
        Literal["eeg", "audio", "label", "all"]
        | Sequence[Literal["eeg", "audio", "label", "all"]]
    ) = "eeg"

    @field_validator("seed")
    def validate_seed(cls, value):
        """Validate the seed value.
        It should be a positive integer.
        """
        assert isinstance(value, int), "seed must be an integer."
        if value < 0:
            raise ValueError("seed must be a positive integer.")
        return value

    @field_validator("apply_prob")
    def validate_apply_prob(cls, value):
        """Validate the apply_prob value.
        It should be between 0 and 1.
        """
        assert isinstance(value, (float, int)), "apply_prob must be a float or int."
        if not (0 <= value <= 1):
            raise ValueError("apply_prob must be between 0 and 1.")
        return value

    @field_validator("when")
    def validate_when(cls, value) -> Literal["before_slicing", "before_returning"]:
        valid_options = ["before_slicing", "before_returning"]
        if value not in valid_options:
            raise ValueError(f"when must be one of {valid_options}, but got {value}.")
        return value

    @field_validator("whom")
    def validate_whom(
        cls, value
    ) -> (
        Literal["eeg", "audio", "label", "all"]
        | Sequence[Literal["eeg", "audio", "label", "all"]]
    ):
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
                        f"whom must be one of {valid_options}, but got {v}."
                    )
        else:
            raise ValueError(
                f"whom must be a string or a sequence, but got {type(value).__name__}."
            )
        return value


class Transform(ABC):
    """Abstract base class for EEG transforms in superhuge package."""

    @abstractmethod
    def __init__(
        self,
        /,
        *,
        seed: int = 42,
        apply_prob: float = 1.0,
        when: Literal["before_slicing", "before_returning"] = "before_returning",
        whom: (
            Literal["eeg", "audio", "label", "all"]
            | Sequence[Literal["eeg", "audio", "label", "all"]]
        ) = "eeg",
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
        self._cfg = TransformConfig(
            when=when,
            whom=whom if isinstance(whom, Sequence) else (whom,),
            apply_prob=apply_prob,
        )
        self.dice = np.random.Generator(np.random.PCG64(seed))
        super().__init__()

    @property
    def apply_prob(self) -> float:
        """Probability of applying this transform."""
        return self._cfg.apply_prob

    @property
    def when(self) -> Literal["before_slicing", "before_returning"]:
        """When to apply this transform."""
        return self._cfg.when

    @property
    def whom(self) -> str | Sequence[str]:
        """Whom this transform is applied to."""
        return self._cfg.whom

    @apply_prob.setter
    def apply_prob(self, value: float) -> None:
        """Set the probability of applying this transform."""
        self._cfg.validate_apply_prob(value)
        self._cfg.apply_prob = value

    @when.setter
    def when(self, value: str) -> None:
        """Set when to apply this transform."""
        self._cfg.when = self._cfg.validate_when(value)

    @whom.setter
    def whom(self, value: str | Sequence[str]) -> None:
        """Set whom this transform is applied to."""
        self._cfg.whom = self._cfg.validate_whom(value)

    @abstractmethod
    def __call__(
        self, x: np.ndarray | int | str, /, *args: Any, **kwargs
    ) -> tuple[np.ndarray | int | str, Any]:
        """Apply transformation to EEG data.

        Args:
            x (np.ndarray): Input EEG data (shape: T × C).

        Returns:
            np.ndarray: Transformed EEG data.
        """
        if isinstance(x, int) or isinstance(x, str):
            # If x is an int or str, we assume it's a label or metadata.
            # We return it as is, since transforms are not applied to labels.
            return x, *args
        elif isinstance(x, np.ndarray):
            assert x.ndim in (2, 3), f"Input must be a 2D/3D ndarray but got {x.ndim}D"
            assert (
                x.shape[1] < x.shape[0]
            ), f"Input must be time first, but seems to be channel first: {x.shape}."
            return x, *args
        else:
            raise TypeError(
                f"Input must be a np.ndarray, int, or str but got {type(x).__name__}."
            )

    def roll(self) -> bool:
        return self.dice.random() < self.apply_prob

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
