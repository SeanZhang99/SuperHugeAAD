from ..metadata_processing.data import RegressionMetadataElement
from .abc import RegressionMetadataFilter

__all__ = ["get_regression_filter"]

ALLOWED_SPEECH_FEATURES = ["env", "mel"]


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


def get_regression_filter(
    speech_feature: str, *args, **kwargs
) -> RegressionMetadataFilter:
    if speech_feature == "env":
        return EnvFilter()
    elif speech_feature == "mel":
        return MelFilter()
    else:
        raise ValueError(
            f"REGRESSION_FILTER:GET_REGRESSION_FILTER:VALUE_ERROR: Invalid speech feature name: {speech_feature}"
        )
