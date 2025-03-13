from typing import Sequence
from ..data import MetaDataElement
from .abc import MetadataFilter


class DatasetSelector(MetadataFilter):
    def __init__(self,/,*, dataset_name: str | Sequence[str]) -> None:
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


class SubjectSelector(MetadataFilter):
    def __init__(self, /,*,subject_id: int | Sequence[int]) -> None:
        if isinstance(subject_id, int):
            self.subject_id = [subject_id]
        else:
            self.subject_id = list(subject_id)

    def __call__(self, metadata_element: MetaDataElement | None) -> MetaDataElement | None:
        if metadata_element is None:
            return None
        elif "subject_id" not in metadata_element.model_fields:
            return None
        elif (
            hasattr(metadata_element, "subject_id")
            and metadata_element.subject_id in self.subject_id  # type: ignore
        ):
            return metadata_element
        return None
