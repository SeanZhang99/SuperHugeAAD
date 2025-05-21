import numpy as np
from .eeg_regression_base_dataset import EegRegressionBaseDataset
from . import EegDataset
from ..metadata_processing.data import ClassifyMetaDataElement


class EegClassifyBaseDataset(EegDataset):
    metadata_cls = ClassifyMetaDataElement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        required_meta_fields = ["label"]
        self._validate_kwargs(kwargs["metadata_fields"], required_meta_fields)

    def load_data(self, idx):
        item = super().load_data(idx)
        meta: dict = item["meta"]
        eeg: np.ndarray | np.memmap = item["eeg"]

        # in normal conditions, `label` is in `meta`.
        # If a diamond inheritage is used (e.g. class(EegClassifyBaseDataset, EegRegressionBaseDataset)),
        # `label` may be delted during regression.__getitem__
        # in this case, label is None.
        label: int | None = meta.get("label", None)

        assert isinstance(label, (int)) or isinstance(
            super(), EegRegressionBaseDataset
        ), f"EEG_CLASSIFY_BASE_DATASET:load_data:ASSERTION:VALUE_ERROR: label must be an integer, got {type(label)}"

        if self.transform:
            label = self.transform(
                label, meta=meta, whom="label", when="before_returning"
            )[0]

        meta["label"] = label

        return {"meta": meta, "eeg": eeg, "label": label}

    def __getitem__(self, idx):
        item = self.load_data(idx)

        meta = item["meta"]
        eeg = item["eeg"]
        label = item["label"]

        # Remove unnecessary fields
        for field in ["env", "mel", "wav"]:
            if field in meta:
                del meta[field]
            if f"{field}_fs" in meta:
                del meta[f"{field}_fs"]

        return {"meta": meta, "eeg": eeg, "label": label}

        # def __getitem__(self,idx):
        # .....

        # def load_data(self,idx):
        #     meta, eeg = super().load_data(idx).values()
        #     ....

        # def __getitem__(self,idx):
        #     ...
