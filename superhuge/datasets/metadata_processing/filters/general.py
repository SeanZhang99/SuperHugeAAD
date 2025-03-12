from typing import Sequence
from ..data import MetaDataElement
from .abc import MetadataFilter


class DatasetSelector(MetadataFilter):
    def __init__(self, dataset_name: str | Sequence[str]) -> None:
        if isinstance(dataset_name, str):
            self.dataset_name = [dataset_name]
        else:
            self.dataset_name = list(dataset_name)

    def __call__(
        self,
        metadata_element: MetaDataElement | None,
    ) -> MetaDataElement | None:
        if metadata_element is None:
            return None
        elif "dataset_name" not in metadata_element.model_fields:
            return None
        elif (
            hasattr(metadata_element, "dataset_name")
            and metadata_element.dataset_name in self.dataset_name  # type: ignore
        ):
            return metadata_element
        return None
