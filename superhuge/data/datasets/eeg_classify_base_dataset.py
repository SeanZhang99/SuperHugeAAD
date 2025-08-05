from collections.abc import Mapping
from typing import Literal, override

import numpy as np

from ..metadata_processing.data import ClassifyMetadataElement
from .eeg_dataset import EegDataset
from .eeg_regression_base_dataset import EegRegressionBaseDataset

# from .typing import DATASet


class EegClassifyBaseDataset(EegDataset):
    metadata_cls = ClassifyMetadataElement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        required_meta_fields = ["label"]
        self._validate_kwargs(kwargs["metadata_fields"], required_meta_fields)

    # python require mapping key to be invariant (it does not allow extending or narrowing the `Literal`).
    # Declare a type ignore to suppress warning, but make the return type more clear for users.
    @override
    def load_data(self, idx) -> Mapping[  # type: ignore
        Literal["meta", "eeg", "label"],
        np.ndarray | ClassifyMetadataElement,
    ]:
        item = super().load_data(idx)
        meta: ClassifyMetadataElement = item["meta"]  # type: ignore
        eeg: np.ndarray | np.memmap = item["eeg"]  # type: ignore

        # in normal conditions, `label` is in `meta`.
        # If a diamond inheritage is used (e.g. class(EegClassifyBaseDataset, EegRegressionBaseDataset)),
        # `label` may be delted during regression.__getitem__
        # in this case, label is None.
        label: int | str = meta.label  # type: ignore

        assert isinstance(label, (int, str)) or isinstance(
            super(), EegRegressionBaseDataset
        ), f"EEG_CLASSIFY_BASE_DATASET:load_data:ASSERTION:VALUE_ERROR: label must be an integer, got {type(label)}"

        if self.transform:
            label: int | str = self.transform(
                label, meta=meta, whom="label", when="before_returning"
            )[
                0
            ]  # type: ignore

        meta.label = label

        return {"meta": meta, "eeg": eeg, "label": label}  # type: ignore

    def __getitem__(  # type: ignore
        self, idx
    ) -> Mapping[Literal["meta", "eeg", "label"], np.ndarray | dict]:
        item = self.load_data(idx)

        meta: dict = item["meta"].model_dump()  # type: ignore
        eeg: np.ndarray = item["eeg"]  # type: ignore
        label: np.ndarray = item["label"]  # type: ignore

        # Remove unnecessary fields
        for field in ["env", "mel", "wav"]:
            if field in meta:
                del meta[field]
            if f"{field}_fs" in meta:
                del meta[f"{field}_fs"]

        return {"meta": meta, "eeg": eeg, "label": label}
