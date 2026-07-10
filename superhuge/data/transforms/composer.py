from typing import Any, Literal, Sequence
import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core.core_schema import CoreSchema, no_info_plain_validator_function

from ..metadata_processing.data import MetadataElement
from .abc import Transform


class TransformComposer:
    """Base class for metadata filters."""

    def __init__(
        self,
        *transforms: Transform,
        **kwargs,
    ) -> None:
        self._transforms = list(transforms)
        self._when_options: list[Literal["before_slicing", "before_returning"]] = [
            "before_slicing",
            "before_returning",
        ]

    def __call__(
        self,
        x: np.ndarray,
        /,
        *args: Any,
        meta: MetadataElement,
        when: Literal["before_slicing", "before_returning"],
        whom: str,
        **kwargs,
    ) -> tuple[np.ndarray, MetadataElement]:
        """Filter metadata element."""
        assert (
            when in self._when_options
        ), f"Invalid when option: {when}. Expected one of {self._when_options}."
        for transform in self._transforms:
            if transform.when == when and (
                whom in transform.whom or "all" in transform.whom
            ):
                output = transform(x, meta=meta)
                assert isinstance(
                    output["x"], np.ndarray
                ), f"Transform {transform} did not return expected ndarray for x."
                x = output["x"]
                if "meta" in output:
                    meta = output["meta"]  # type: ignore

        return x, meta

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: object) -> "TransformComposer":
        if not isinstance(value, TransformComposer):
            raise TypeError(
                f"Expected an instance of TransformComposer, got {type(value).__name__}"
            )
        else:
            for transform in value.transforms:
                assert isinstance(
                    transform, Transform
                ), f"Transform {transform} is not a Transform."
                assert callable(transform), f"Transform {transform} is not callable."
        return value

    @property
    def transforms(self) -> Sequence[Transform]:
        """Get transforms."""
        return self._transforms

    @property
    def when_options(self) -> Sequence[Literal["before_slicing", "before_returning"]]:
        """Get when options."""
        return self._when_options
