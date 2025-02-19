from collections.abc import Callable

from .classfy_filter import (
    get_classification_filter,
    ALLOWED_NUM_CLASS_INT,
    ALLOWED_NUM_CLASS_STRING,
)
from .eeg_dataset import EegDataset
from .metadata_processing import Callable, ClassifyMetaDataElement


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
        assert (
            len(args) > 0 or "target" in kwargs
        ), "EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:INPUT_ARGUMENT_ERROR: target must be specified at the first positional argument or as a keyword argument"
        target = args[0] if len(args) > 0 else kwargs["target"]

        # Validate target is a valid argument
        assert (isinstance(target, int) and target in ALLOWED_NUM_CLASS_INT) or (
            isinstance(target, str) and target in ALLOWED_NUM_CLASS_STRING
        ), f"EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:VALUE_ERROR: target must be a valid integer from {ALLOWED_NUM_CLASS_INT} or a valid string from {ALLOWED_NUM_CLASS_STRING}"
        assert isinstance(
            target, (int, str)
        ), "EEG_CLASSIFY_BASE_DATASET:META_FILTER_FUNC_PARSER:ASSERTION:TYPE_ERROR: target must be either an integer or a string"

        if meta_filter_func is None:
            meta_filter_func = get_classification_filter(target)
        return meta_filter_func
