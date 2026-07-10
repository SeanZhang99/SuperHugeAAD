from ..metadata_processing.data import RegressionMetadataElement
from .abc import MetadataFilter

__all__ = ["get_regression_filter"]


class RegressionMetadataFilter(MetadataFilter):
    """Base class for regression metadata filters.

    Does nothing, just for identification.
    """

    _filter_marker = True


class EnvFilter(RegressionMetadataFilter):
    def __call__(
        self, metadata_element: RegressionMetadataElement | None
    ) -> RegressionMetadataElement | None:
        """
        This function filters the metadata elements based on the presence of the 'env' attribute.
        If the 'env' attribute is present, the function returns the metadata element.
        Otherwise, it returns None.
        """
        if hasattr(metadata_element, "env") and getattr(metadata_element, "env"):
            return metadata_element
        return None


class MelFilter(RegressionMetadataFilter):
    def __call__(
        self, metadata_element: RegressionMetadataElement | None
    ) -> RegressionMetadataElement | None:
        """
        This function filters the metadata elements based on the presence of the 'mel' attribute.
        If the 'mel' attribute is present, the function returns the metadata element.
        Otherwise, it returns None.
        """
        if hasattr(metadata_element, "mel") and getattr(metadata_element, "mel"):
            return metadata_element
        return None


class AudioFilter(RegressionMetadataFilter):
    def __init__(self, /, *, field_name: str, **kwargs):
        super().__init__(**kwargs)
        self.field_name = field_name

    def __call__(
        self, metadata_element: RegressionMetadataElement | None
    ) -> RegressionMetadataElement | None:
        """
        This function filters the metadata elements based on the presence of a specified audio attribute.
        If the specified audio attribute is present, the function returns the metadata element.
        Otherwise, it returns None.
        """
        if hasattr(metadata_element, self.field_name) and getattr(
            metadata_element, self.field_name
        ):
            return metadata_element
        return None

    def __repr__(self) -> str:
        return f"AudioFilter(field_name={self.field_name})"


def get_regression_filter(speech_feature: str) -> RegressionMetadataFilter:
    return AudioFilter(field_name=speech_feature)
