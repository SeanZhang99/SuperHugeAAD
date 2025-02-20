from collections.abc import Callable

from ..utils.metadata_processing.data import ClassifyMetaDataElement

__all__ = ["get_classification_filter"]

# Define the allowed number of class strings
ALLOWED_NUM_CLASS_STRING = [
    "binary_leftright",
    "binary_frontrear",
    "four-class",
    "eight-class",
]
ALLOWED_NUM_CLASS_INT = [2, 4, 8]


def angle_wrapper(label: int) -> int:
    """
    Convert -180 to 0 degree to 180-360.
    """
    if -180 <= label < 0:
        return label + 360
    return label


def binary_leftright_filter(
    metadata_element: ClassifyMetaDataElement,
) -> ClassifyMetaDataElement | None:
    """
    This function filters the metadata elements based on the label value.
    If the label value is a string, it should be either "left" or "right" (case insensitive).
    If the label value is an integer, it should be between 0 and 180 or between 180 and 360.
    If the label value is in the range of 0 to 180, the label value is set to "right".
    If the label value is in the range of 180 to 360, the label value is set to "left".
    If the label value is not in the specified ranges, the function returns None.
    """
    result = None
    label = metadata_element.label
    if isinstance(label, str):
        if label.lower() in ["left", "right"]:
            metadata_element.label = label.lower()
            result = metadata_element
    elif isinstance(label, int):
        label = angle_wrapper(label)
        if 180 < label < 360:
            metadata_element.label = "left"
            result = metadata_element
        elif 0 < label < 180:
            metadata_element.label = "right"
            result = metadata_element
    return result


def binary_frontrear_filter(
    metadata_element: ClassifyMetaDataElement,
) -> ClassifyMetaDataElement | None:
    """
    This function filters the metadata elements based on the label value.
    If the label value is a string, it should be either "front" or "rear" (case insensitive).
    If the label value is an integer, it should be between 0 and 180 or between 180 and 360.
    If the label value is in the range of 0 to 180, the label value is set to "front".
    If the label value is in the range of 180 to 360, the label value is set to "rear".
    If the label value is not in the specified ranges, the function returns None.
    """
    result = None
    label = metadata_element.label
    if isinstance(label, str):
        if label.lower() in ["front", "rear"]:
            metadata_element.label = label.lower()
            result = metadata_element
    elif isinstance(label, int):
        label = angle_wrapper(label)
        if 180 < label < 360:
            metadata_element.label = "rear"
            result = metadata_element
        elif 0 < label < 180:
            metadata_element.label = "front"
            result = metadata_element
    return result


def four_class_filter(
    metadata_element: ClassifyMetaDataElement,
) -> ClassifyMetaDataElement | None:
    """
    This function filters the metadata elements into four classes based on the label value.
    The classes are: -45-45, 45-135, 135-225, 225-315.
    If the label value is an integer, it should be between 0 and 360.
    The label value is set to one of the four classes based on its range.
    If the label value is not in the specified ranges, the function returns None.
    """
    result = None
    label = metadata_element.label
    if isinstance(label, int):
        label = angle_wrapper(label)
        if 0 <= label < 45 or 315 <= label < 360:
            metadata_element.label = "Front-Right"
            result = metadata_element
        elif 45 <= label < 135:
            metadata_element.label = "Front-Left"
            result = metadata_element
        elif 135 <= label < 225:
            metadata_element.label = "Rear-Left"
            result = metadata_element
        elif 225 <= label < 315:
            metadata_element.label = "Rear-Right"
            result = metadata_element
    return result


def eight_class_filter(
    metadata_element: ClassifyMetaDataElement,
) -> ClassifyMetaDataElement | None:
    """
    This function filters the metadata elements into eight classes based on the label value.
    The classes are: -22.5 to 22.5, 22.5 to 67.5, 67.5 to 112.5, 112.5 to 157.5,
    157.5 to 202.5, 202.5 to 247.5, 247.5 to 292.5, 292.5 to 337.5, 337.5 to 360.
    If the label value is an integer, it should be between 0 and 360.
    The label value is set to one of the eight classes based on its range.
    If the label value is not in the specified ranges, the function returns None.
    """
    result = None
    label = metadata_element.label
    if isinstance(label, int):
        label = angle_wrapper(label)
        if 337.5 <= label < 360 or 0 <= label < 22.5:
            metadata_element.label = "North"
            result = metadata_element
        elif 22.5 <= label < 67.5:
            metadata_element.label = "North-East"
            result = metadata_element
        elif 67.5 <= label < 112.5:
            metadata_element.label = "East"
            result = metadata_element
        elif 112.5 <= label < 157.5:
            metadata_element.label = "South-East"
            result = metadata_element
        elif 157.5 <= label < 202.5:
            metadata_element.label = "South"
            result = metadata_element
        elif 202.5 <= label < 247.5:
            metadata_element.label = "South-West"
            result = metadata_element
        elif 247.5 <= label < 292.5:
            metadata_element.label = "West"
            result = metadata_element
        elif 292.5 <= label < 337.5:
            metadata_element.label = "North-West"
            result = metadata_element
    return result


def get_classify_filter(
    num_class: int | str,
) -> Callable[[ClassifyMetaDataElement], ClassifyMetaDataElement | None]:
    """
    This function returns a filter function based on the number of classes.
    The filter function is selected based on the number of classes.
    The number of classes can be 2, 4, or 8.
    If the number of classes is 2, the filter function is binary_leftright_filter.
    If the number of classes is 4, the filter function is four_class_filter.
    If the number of classes is 8, the filter function is eight_class_filter.
    """
    if isinstance(num_class, str):
        if num_class == "binary_leftright":
            filter_func = binary_leftright_filter
        elif num_class == "binary_frontrear":
            filter_func = binary_frontrear_filter
        elif num_class == "four_class":
            filter_func = four_class_filter
        elif num_class == "eight_class":
            filter_func = eight_class_filter
        else:
            raise ValueError(
                f"CLASSIFIER_FILTER:GET_CLASSIFICATION:FILTER:STR_INPUT:VALUE_ERROR: Invalid number of classes. The number of classes can be {ALLOWED_NUM_CLASS_STRING}."
            )
    elif isinstance(num_class, int):
        if num_class == 2:
            filter_func = binary_leftright_filter
        elif num_class == 4:
            filter_func = four_class_filter
        elif num_class == 8:
            filter_func = eight_class_filter
        else:
            raise ValueError(
                f"CLASSIFIER_FILTER:GET_CLASSIFICATION_FILTER:INT_INPUT:VALUE_ERROR: nvalid number of classes. The number of classes can be {ALLOWED_NUM_CLASS_INT}."
            )
    else:
        raise TypeError(
            f"CLASSIFER_FILTER:GET_CLASSIFICATION_FILTER:{str(type(input)).upper()}_INPUT:TYPE_ERROR: Invalid type for num_class. The number of classes can be an integer or a string."
        )
    return filter_func
