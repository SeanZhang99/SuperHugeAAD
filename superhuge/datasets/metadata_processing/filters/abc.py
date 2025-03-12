from abc import ABC, abstractmethod
from ..data import MetaDataElement


class MetadataFilter(ABC):
    @abstractmethod
    def filter_metadata_element(
        self,
        metadata_element: MetaDataElement | None,
    ) -> MetaDataElement | None:
        pass
