from __future__ import annotations
from copy import deepcopy
from pydantic import GetCoreSchemaHandler
from pydantic_core.core_schema import CoreSchema, no_info_plain_validator_function

from ..metadata_processing.data import Metadata, MetadataElement
from .general import MetadataFilter
import collections.abc
import inspect
from typing import Callable, get_origin

# Types that indicate the filter expects the entire (batched) metadata dict.
_COLLECTION_TYPES = (collections.abc.Mapping, collections.abc.Sequence)

# Sequences that are NOT treated as batch containers (e.g. str is technically a Sequence).
_ATOMIC_SEQUENCE_TYPES = (str, bytes, bytearray)


def _is_batch_filter(func: Callable) -> bool:
    """Return True if the first parameter's annotation is a collection type (batch mode)."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        return False

    annotation = params[0].annotation
    if annotation is inspect.Parameter.empty:
        # No annotation → default to per-element mode for backward compatibility.
        return False

    # Unwrap generic aliases (e.g. Dict[K, V] → dict, Mapping[K, V] → Mapping).
    origin = get_origin(annotation)
    check_type = origin if origin is not None else annotation

    # Reject atomic sequence types (str, bytes, etc.) that technically are Sequences.
    if isinstance(check_type, type) and issubclass(check_type, _ATOMIC_SEQUENCE_TYPES):
        return False

    # Check if the resolved type is a subclass of Mapping or Sequence.
    if isinstance(check_type, type) and issubclass(check_type, _COLLECTION_TYPES):
        return True

    return False


class MetadataFilterComposer:
    """Base class for metadata filters."""

    def __init__(
        self,
        *filters: MetadataFilter,
    ) -> None:
        for filter in filters:
            assert isinstance(
                filter, MetadataFilter
            ), f"Filter {filter} is not a MetadataFilter."
        self._filters = list(filters)

    # def __call__(
    #     self, metadata_element: MetadataElement | None
    # ) -> MetadataElement | None:
    #     """Filter metadata element."""
    #     for filter_func in self._filters:
    #         metadata_element = filter_func(metadata_element)
    #         if metadata_element is None:
    #             return None
    #     return metadata_element

    def __call__(self, metadata: Metadata) -> Metadata:
        """Filter metadata — apply all filters in sequence.

        Uses ``filter.apply(dict)`` as the universal batch entry point,
        which works for both legacy and new-style filters.
        """
        out: dict[str, MetadataElement] = deepcopy(metadata)  # type: ignore
        for filter_func in self._filters:
            out = filter_func.apply(out)
        return out

    @property
    def filters(self) -> list[MetadataFilter]:
        """Get filters."""
        return self._filters

    @filters.setter
    def filters(self, new_filters: list[MetadataFilter]) -> None:
        """Set filters."""
        for filter in new_filters:
            assert isinstance(
                filter, MetadataFilter
            ), f"Filter {filter} is not a MetadataFilter."
        self.filters = new_filters

    @filters.deleter
    def filters(self) -> None:
        """Delete all filters."""
        self.filters = []

    def add_filters(self, *new_filters: MetadataFilter) -> None:
        """Add some filters."""
        for filter in new_filters:
            assert isinstance(
                filter, MetadataFilter
            ), f"Filter {filter} is not a MetadataFilter."
        self.filters.extend(new_filters)

    def pop_filters(self, index) -> None:
        """Remove some filters."""
        self.filters.pop(index)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return no_info_plain_validator_function(cls._validate)

    @classmethod
    def _validate(cls, value: object) -> MetadataFilterComposer:
        if not isinstance(value, MetadataFilterComposer):
            raise TypeError(
                f"Expected an instance of MetadataFilterComposer, got {type(value).__name__}"
            )
        else:
            for filter in value.filters:
                assert isinstance(
                    filter, MetadataFilter
                ), f"Filter {filter} is not a MetadataFilter."
                assert callable(filter), f"Filter {filter} is not callable."
        return value
