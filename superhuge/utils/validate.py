from collections.abc import Iterable


def validate_kwargs(kwargs: Iterable, required_keys: Iterable):
    """
    验证 kwargs 是否包含所有必需的键。
    """
    for key in required_keys:
        if key not in kwargs:
            raise KeyError(
                f"EEG_DATASET:VALIDATE_KWARGS:KEY_ERROR: Missing required key '{key}' in kwargs {kwargs}. You should go back to the caller function and check the kwargs to be validated"
            )
