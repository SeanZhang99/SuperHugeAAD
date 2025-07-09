from ..metadata_processing.data import ClassifyMetadataElement
from .abc import ClassifyMetadataFilter

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
# from ... import ...

__all__ = ["get_classify_filter"]

ALLOWED_NUM_CLASS_STRING = [
    "binary_leftright",
    "binary_frontrear",
    "four-class",
    "eight-class",
]
ALLOWED_NUM_CLASS_INT = [2, 4, 8]


def angle_wrapper(label: "int") -> int:
    """
    Convert -180 to 0 degree to 180-360.
    """
    assert (
        -360 <= label < 360
    ), f"ANGLE_WRAPPER:VALUE_ERROR: Invalid label value. The label value must be between -360 and 360. Got {label}."
    if -360 <= label < 0:
        return label + 360
    return label


class BinaryLeftRightFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
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
                result.label = 0 if result.label == "left" else 1
            else:
                try:
                    metadata_element.label = int(label)
                    result = self(metadata_element)
                except ValueError:
                    result = None
        elif isinstance(label, int):
            label = angle_wrapper(label)
            if 180 < label < 360:
                metadata_element.label = 0
            elif 0 < label < 180:
                metadata_element.label = 1
            result = metadata_element
        return result


class BinaryFrontRearFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        """
        This function filters the metadata elements based on the label value.
        If the label value is a string, it should be either "front" or "rear" (case insensitive).
        If the label value is an integer, it should be between 0 and 360.
        For `front` (270~90), convert to `0`, for `rear` (90~270), convert to `1`.
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, str):
            if label.lower() in ["front", "rear"]:
                metadata_element.label = label.lower()
                result = metadata_element
                result.label = 0 if result.label == "front" else 1
            else:
                try:
                    metadata_element.label = int(label)
                    result = self(metadata_element)
                except ValueError:
                    result = None
        elif isinstance(label, int):
            label = angle_wrapper(label)
            if 90 <= label < 270:
                metadata_element.label = 1  # Rear
            elif 270 <= label < 360 or 0 <= label < 90:
                metadata_element.label = 0  # Front
            result = metadata_element
        return result


class FourClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        """
        This function filters the metadata elements into four classes based on the label value.
        The classes are: -45-45, 45-135, 135-225, 225-315.
        Labels will be converted to int: `fr`->0, `fl`->1, `rl`->2, `rr`->3.
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, int):
            label = angle_wrapper(label)
            if 0 <= label < 45 or 315 <= label < 360:
                metadata_element.label = 0
            elif 45 <= label < 135:
                metadata_element.label = 1
            elif 135 <= label < 225:
                metadata_element.label = 2
            elif 225 <= label < 315:
                metadata_element.label = 3
            result = metadata_element
        elif isinstance(label, str):
            try:
                metadata_element.label = int(label)
                result = self(metadata_element)
            except ValueError:
                result = None
        return result


class EightClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        """
        This function filters the metadata elements into eight classes based on the label value.
        The classes are: -22.5 to 22.5, 22.5 to 67.5, 67.5 to 112.5, 112.5 to 157.5,
        157.5 to 202.5, 202.5 to 247.5, 247.5 to 292.5, 292.5 to 337.5, 337.5 to 360.
        Labels will be converted to int: `north`->0, `north-east`->1, ..., `north-west`->7.
        """
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, int):
            label = angle_wrapper(label)
            if 337.5 <= label < 360 or 0 <= label < 22.5:
                metadata_element.label = 0
            elif 22.5 <= label < 67.5:
                metadata_element.label = 1
            elif 67.5 <= label < 112.5:
                metadata_element.label = 2
            elif 112.5 <= label < 157.5:
                metadata_element.label = 3
            elif 157.5 <= label < 202.5:
                metadata_element.label = 4
            elif 202.5 <= label < 247.5:
                metadata_element.label = 5
            elif 247.5 <= label < 292.5:
                metadata_element.label = 6
            elif 292.5 <= label < 337.5:
                metadata_element.label = 7
            result = metadata_element
        elif isinstance(label, str):
            try:
                metadata_element.label = int(label)
                result = self(metadata_element)
            except ValueError:
                result = None
        return result


class AsIsClassifyFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        """
        This function returns the metadata element as is.
        """
        if hasattr(metadata_element, "label"):
            if isinstance(metadata_element.label, str):
                metadata_element.label = int(metadata_element.label.lower())
            elif isinstance(metadata_element.label, int):
                metadata_element.label = angle_wrapper(metadata_element.label)
            else:
                raise TypeError(
                    f"CLASSIFIER_FILTER:AS_IS_CLASSIFY_FILTER:LABEL_TYPE_ERROR: Invalid type for label. The label must be an integer or a string. Got {str(type(metadata_element.label)).upper()}."
                )
            return metadata_element
        else:
            return None


class cEEGridThreeClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, str):
            label = int(label)
        assert isinstance(label, int)
        if label < 0:
            metadata_element.label = 0
        elif label == 0:
            metadata_element.label = 1
        elif label > 0:
            metadata_element.label = 2
        result = metadata_element
        return result


class cEEGridBinaryFilter(cEEGridThreeClassFilter):
    def __call__(self, metadata_element):
        result = super().__call__(metadata_element)
        if result is not None and result.label == 1:
            result = None

        return result


label_hstb = {-60: 0, -120: 1, 0: 2, 60: 3, 120: 4}


class cEEGridFiveClassFilter(ClassifyMetadataFilter):
    def __call__(
        self, metadata_element: ClassifyMetadataElement | None
    ) -> ClassifyMetadataElement | None:
        if metadata_element is None:
            return None
        result = None
        label = metadata_element.label
        if isinstance(label, str):
            label = int(label)
        assert isinstance(label, int)
        if label in label_hstb:
            metadata_element.label = label_hstb[label]
            result = metadata_element
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
        elif num_class == "as_is":
            return AsIsClassifyFilter()
        elif num_class == "cEEGrid_three_class":
            return cEEGridThreeClassFilter()
        elif num_class == "cEEGrid_five_class":
            return cEEGridFiveClassFilter()
        elif num_class == "cEEGrid_binary_leftright":
            return cEEGridBinaryFilter()
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
