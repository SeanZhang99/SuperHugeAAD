from abc import ABC, abstractmethod

from ..metadata_processing.data import MetaDataElement


class MetadataFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        metadata_element: MetaDataElement | None,
    ) -> MetaDataElement | None:
        """
        __call__ call method of a metadata filter.

        :param metadata_element: metadata_element to be filtered. Should be an instance of MetaDataElement.
        :type metadata_element: MetaDataElement | None
        :return: fitlered metadata_element (can be modified according to the rule of the filter). Or `None`, if
        to discard this metadata_element.
        :rtype: MetaDataElement | None
        """
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
