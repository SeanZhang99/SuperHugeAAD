from abc import ABC, abstractmethod

from ..metadata_processing.data import MetadataElement


class MetadataFilter(ABC):
    @abstractmethod
    def __call__(
        self,
        metadata_element: MetadataElement | None,
    ) -> MetadataElement | None:
        """
        __call__ call method of a metadata filter.

        :param metadata_element: metadata_element to be filtered. Should be an instance of MetadataElement.
        :type metadata_element: MetadataElement | None
        :return: fitlered metadata_element (can be modified according to the rule of the filter). Or `None`, if
        to discard this metadata_element.
        :rtype: MetadataElement | None
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
