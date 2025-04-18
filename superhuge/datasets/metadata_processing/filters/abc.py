from abc import ABC, abstractmethod
from ..data import MetaDataElement


class MetadataFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        metadata_element: MetaDataElement | None,
    ) -> MetaDataElement | None:
        pass


class ClassifyMetadataFilter(MetadataFilter):
    """
    Base class for classification metadata filters.
    Does nothing, just for identification.
    """

    pass


class RegressionMetadataFilter(MetadataFilter):
    """
    Base class for regression metadata filters.
    Does nothing, just for identification.
    """

    pass
