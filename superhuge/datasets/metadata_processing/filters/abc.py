from abc import ABC, abstractmethod
from ..data import MetaDataElement


class MetadataFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        metadata_element: MetaDataElement | None,
    ) -> MetaDataElement | None:
        pass
