from collections.abc import Callable
from .metadata_processing import RegressionMetaDataElement

__all__ = ["get_regression_filter"]

ALLOWED_SPEECH_FEATURES = ["env", "mel"]


def env_filter(
    metadata_element: RegressionMetaDataElement,
) -> RegressionMetaDataElement | None:
    """
    This function filters the metadata elements based on the presence of the 'env' attribute.
    If the 'env' attribute is present, the function returns the metadata element.
    Otherwise, it returns None.
    """
    if hasattr(metadata_element, "env"):
        return metadata_element
    return None


def mel_filter(
    metadata_element: RegressionMetaDataElement,
) -> RegressionMetaDataElement | None:
    """
    This function filters the metadata elements based on the presence of the 'mel' attribute.
    If the 'mel' attribute is present, the function returns the metadata element.
    Otherwise, it returns None.
    """
    if hasattr(metadata_element, "mel"):
        return metadata_element
    return None


def get_regression_filter(
    speech_feature: str,
) -> Callable[[RegressionMetaDataElement], RegressionMetaDataElement | None]:
    """
    This function returns a filter function based on the filter name.
    The filter function is selected based on the filter name.
    """
    filter_func = globals().get(f"{speech_feature}_filter")
    if filter_func is None:
        raise ValueError(
            f"REGRESSION_FILTER:GET_REGRESSION_FILTER:VALUE_ERROR: Invalid speech feature name: {speech_feature}"
        )
    return filter_func
