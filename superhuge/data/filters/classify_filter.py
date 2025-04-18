from ..metadata_processing.data import ClassifyMetaDataElement
from .abc import ClassifyMetadataFilter

__all__ = ["get_classify_filter"]

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


class BinaryLeftRightFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetaDataElement | None
    ) -> ClassifyMetaDataElement | None:
        """
        This function filters the metadata elements based on the label value.
        If the label value is a string, it should be either "left" or "right" (case insensitive).
        If the label value is an integer, it should be between 0 and 180 or between 180 and 360.
        If the label value is in the range of 0 to 180, the label value is set to "right".
        If the label value is in the range of 180 to 360, the label value is set to "left".
        For `left`, convert to `0`, for `right`, convert to `1`.
        If the label value is not in the specified ranges, the function returns None.
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, str):
            if label.lower() in ["left", "right"]:
                metadata_element.label = label.lower()
                result = metadata_element
            else:
                try:
                    label = int(label)
                    metadata_element.label = angle_wrapper(label)
                    result = self(metadata_element)
                except ValueError:
                    result = None
        elif isinstance(label, int):
            label = angle_wrapper(label)
            if 180 < label < 360:
                metadata_element.label = "left"
            elif 0 < label < 180:
                metadata_element.label = "right"
            result = metadata_element
        if result is not None:
            result.label = 0 if result.label == "left" else 1
        return result


class BinaryFrontRearFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetaDataElement | None
    ) -> ClassifyMetaDataElement | None:
        """
        This function filters the metadata elements based on the label value.
        If the label value is a string, it should be either "front" or "rear" (case insensitive).
        If the label value is an integer, it should be between 0 and 180 or between 180 and 360.
        If the label value is in the range of 0 to 180, the label value is set to "front".
        If the label value is in the range of 180 to 360, the label value is set to "rear".
        If the label value is not in the specified ranges, the function returns None.
        Labels will be converted to int: `front`->`0`, `rear`->`1`
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, str):
            if label.lower() in ["front", "rear"]:
                metadata_element.label = label.lower()
                result = metadata_element
            else:
                try:
                    label = int(label)
                    metadata_element.label = angle_wrapper(label)
                    result = self(metadata_element)
                except ValueError:
                    pass
        elif isinstance(label, int):
            label = angle_wrapper(label)
            if 180 < label < 360:
                metadata_element.label = "rear"
            elif 0 < label < 180:
                metadata_element.label = "front"
            result = metadata_element
        if result is not None:
            result.label = 0 if result.label == "front" else 1
        return result


class FourClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetaDataElement | None
    ) -> ClassifyMetaDataElement | None:
        """
        This function filters the metadata elements into four classes based on the label value.
        The classes are: -45-45, 45-135, 135-225, 225-315.
        If the label value is an integer, it should be between 0 and 360.
        The label value is set to one of the four classes based on its range.
        If the label value is not in the specified ranges, the function returns None.
        Labels will be converted to int. `fr`->0,`fl`->1, `rl`->2, `rr`->3
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, int):
            label = angle_wrapper(label)
            if 0 <= label < 45 or 315 <= label < 360:
                metadata_element.label = "Front-Right"
            elif 45 <= label < 135:
                metadata_element.label = "Front-Left"
            elif 135 <= label < 225:
                metadata_element.label = "Rear-Left"
            elif 225 <= label < 315:
                metadata_element.label = "Rear-Right"
            result = metadata_element
        elif isinstance(label, str):
            try:
                label = int(label)
                metadata_element.label = angle_wrapper(label)
                result = self(metadata_element)
            except ValueError:
                pass

        if result is not None:
            if result.label == "Front-Right":
                result.label = 0
            elif result.label == "Front-Left":
                result.label = 1
            elif result.label == "Rear-Left":
                result.label = 2
            elif result.label == "Rear-Right":
                result.label = 3
            else:
                raise ValueError(
                    f"CLASSIFIER_FILTER:FOUR_CLASS_FILTER:VALUE_ERROR: Invalid label value. The label value can only be 'Front-Right', 'Front-Left', 'Rear-Left', or 'Rear-Right'."
                )
        return result


class EightClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetaDataElement | None
    ) -> ClassifyMetaDataElement | None:
        """
        This function filters the metadata elements into eight classes based on the label value.
        The classes are: -22.5 to 22.5, 22.5 to 67.5, 67.5 to 112.5, 112.5 to 157.5,
        157.5 to 202.5, 202.5 to 247.5, 247.5 to 292.5, 292.5 to 337.5, 337.5 to 360.
        If the label value is an integer, it should be between 0 and 360.
        The label value is set to one of the eight classes based on its range.
        If the label value is not in the specified ranges, the function returns None.
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, int):
            label = angle_wrapper(label)
            if 337.5 <= label < 360 or 0 <= label < 22.5:
                metadata_element.label = "North"
            elif 22.5 <= label < 67.5:
                metadata_element.label = "North-East"
            elif 67.5 <= label < 112.5:
                metadata_element.label = "East"
            elif 112.5 <= label < 157.5:
                metadata_element.label = "South-East"
            elif 157.5 <= label < 202.5:
                metadata_element.label = "South"
            elif 202.5 <= label < 247.5:
                metadata_element.label = "South-West"
            elif 247.5 <= label < 292.5:
                metadata_element.label = "West"
            elif 292.5 <= label < 337.5:
                metadata_element.label = "North-West"
            result = metadata_element
        elif isinstance(label, str):
            try:
                label = int(label)
                metadata_element.label = angle_wrapper(label)
                result = self(metadata_element)
            except ValueError:
                pass

        if result is not None:
            if result.label == "North":
                result.label = 0
            elif result.label == "North-East":
                result.label = 1
            elif result.label == "East":
                result.label = 2
            elif result.label == "South-East":
                result.label = 3
            elif result.label == "South":
                result.label = 4
            elif result.label == "South-West":
                result.label = 5
            elif result.label == "West":
                result.label = 6
            elif result.label == "North-West":
                result.label = 7
            else:
                raise ValueError(
                    f"CLASSIFIER_FILTER:EIGHT_CLASS_FILTER:VALUE_ERROR: Invalid label value. The label value can only be 'North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', or 'North-West'."
                )
        return result


def get_classify_filter(
    num_class: int | str,
) -> ClassifyMetadataFilter:
    """
    This function returns a filter class based on the number of classes.
    The filter class is selected based on the number of classes.
    The number of classes can be 2, 4, or 8.
    If the number of classes is 2, the filter class is BinaryLeftRightFilter.
    If the number of classes is 4, the filter class is FourClassFilter.
    If the number of classes is 8, the filter class is EightClassFilter.
    """
    if isinstance(num_class, str):
        if num_class == "binary_leftright":
            return BinaryLeftRightFilter()
        elif num_class == "binary_frontrear":
            return BinaryFrontRearFilter()
        elif num_class == "four_class":
            return FourClassFilter()
        elif num_class == "eight_class":
            return EightClassFilter()
        else:
            raise ValueError(
                f"CLASSIFIER_FILTER:GET_CLASSIFICATION:FILTER:STR_INPUT:VALUE_ERROR: Invalid number of classes. The number of classes can be {ALLOWED_NUM_CLASS_STRING}."
            )
    elif isinstance(num_class, int):
        if num_class == 2:
            return BinaryLeftRightFilter()
        elif num_class == 4:
            return FourClassFilter()
        elif num_class == 8:
            return EightClassFilter()
        else:
            raise ValueError(
                f"CLASSIFIER_FILTER:GET_CLASSIFICATION_FILTER:INT_INPUT:VALUE_ERROR: Invalid number of classes. The number of classes can be {ALLOWED_NUM_CLASS_INT}."
            )
    else:
        raise TypeError(
            f"CLASSIFER_FILTER:GET_CLASSIFICATION_FILTER:{str(type(num_class)).upper()}_INPUT:TYPE_ERROR: Invalid type for num_class. The number of classes can be an integer or a string."
        )
