from collections.abc import Callable

from ..utils.commons.eeg_dataset import EegDataset
from ..utils.metadata_processing.data import ClassifyMetaDataElement
from .classify_filter import (
    ALLOWED_NUM_CLASS_INT,
    ALLOWED_NUM_CLASS_STRING,
    get_classify_filter,
)


class EegClassifyBaseDataset(EegDataset):
    metadata_cls = ClassifyMetaDataElement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        required_meta_fields = ["label"]
        self._validate_kwargs(kwargs["metadata_fields"], required_meta_fields)

    def __getitem__(self, idx):
        meta, exg = super().__getitem__(idx).values()
        label = meta["label"]
        return {"meta": meta, "exg": exg, "label": label}

    @classmethod
    def meta_filter_func_parser(
        cls,
        meta_filter_func: (
            Callable[
                [
                    ClassifyMetaDataElement,
                ],
                ClassifyMetaDataElement | None,
            ]
            | None
        ),
        *args,
        **kwargs,
    ):
        if meta_filter_func is None:
            assert (
                len(args) > 0 or "target" in kwargs
            ), "EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:INPUT_ARGUMENT_ERROR: target must be specified at the first positional argument or as a keyword argument"
            target = args[0] if len(args) > 0 else kwargs["target"]

            # Validate target is a valid argument
            assert (isinstance(target, int) and target in ALLOWED_NUM_CLASS_INT) or (
                isinstance(target, str) and target in ALLOWED_NUM_CLASS_STRING
            ), f"EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:VALUE_ERROR: target must be a valid integer from {ALLOWED_NUM_CLASS_INT} or a valid string from {ALLOWED_NUM_CLASS_STRING}"
            return get_classify_filter(target)
        elif isinstance(meta_filter_func, Callable):
            return super().meta_filter_func_parser(meta_filter_func, *args, **kwargs)
        else:
            raise TypeError(
                f"EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:TYPE_ERROR: meta_filter_func must be a Callable or None, got {type(meta_filter_func)}"
            )
