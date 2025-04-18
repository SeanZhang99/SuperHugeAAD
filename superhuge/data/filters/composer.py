from pydantic import GetCoreSchemaHandler
from pydantic_core.core_schema import CoreSchema, no_info_plain_validator_function

from ..metadata_processing.data import MetaDataElement
from .general import MetadataFilter


class MetaDataFilterComposer:
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

    def __call__(
        self, metadata_element: MetaDataElement | None
    ) -> MetaDataElement | None:
        """Filter metadata element."""
        for filter_func in self._filters:
            metadata_element = filter_func(metadata_element)
            if metadata_element is None:
                return None
        return metadata_element

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
    def _validate(cls, value: object) -> "MetadataFilter":
        if not isinstance(value, MetaDataFilterComposer):
            raise TypeError(
                f"Expected an instance of Transform, got {type(value).__name__}"
            )
        else:
            for filter in value.filters:
                assert isinstance(
                    filter, MetadataFilter
                ), f"Transform {filter} is not a Transform."
                assert callable(filter), f"Transform {filter} is not callable."
        return value
