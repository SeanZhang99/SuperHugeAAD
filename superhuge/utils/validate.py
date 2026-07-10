from collections.abc import Callable, Sequence
import inspect


def validate_kwargs(kwargs: Sequence | dict, required_keys: Sequence | dict):
    """
    验证 kwargs 是否包含所有必需的键。
    """
    if isinstance(required_keys, dict):
        for key in required_keys.keys():
            if key not in kwargs:
                raise KeyError(
                    f"EEG_DATASET:VALIDATE_KWARGS:KEY_ERROR: Missing required key '{key}' in kwargs {kwargs}. You should go back to the caller function and check the kwargs to be validated"
                )
    else:
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    f"EEG_DATASET:VALIDATE_KWARGS:KEY_ERROR: Missing required key '{key}' in kwargs {kwargs}. You should go back to the caller function and check the kwargs to be validated"
                )


def has_required_params(func: Callable, param_dict: dict):
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty and name not in param_dict:
            return False
    return True
