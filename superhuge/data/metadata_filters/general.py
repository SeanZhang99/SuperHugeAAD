from typing import Any, Sequence

from ..metadata_processing.data import MetadataElement
from .abc import MetadataFilter


class MetadataValueSelector(MetadataFilter):
    def __init__(
        self, /, attribute_name: str, attribute_value: Any | Sequence[Any]
    ) -> None:
        self.attribute_name = attribute_name
        if not isinstance(attribute_value, Sequence):
            self.attribute_value = [attribute_value]
        else:
            self.attribute_value = list(attribute_value)

    def __call__(self, metadata_element: MetadataElement | None):
        if (
            isinstance(metadata_element, MetadataElement)
            and hasattr(metadata_element, self.attribute_name)
            and getattr(metadata_element, self.attribute_name) in self.attribute_value
        ):
            return metadata_element
        return None
