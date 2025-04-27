from . import EegDataset
from ..metadata_processing.data import ClassifyMetaDataElement


class EegClassifyBaseDataset(EegDataset):
    metadata_cls = ClassifyMetaDataElement

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        required_meta_fields = ["label"]
        self._validate_kwargs(kwargs["metadata_fields"], required_meta_fields)

    def getitem(self, idx):
        return self.__getitem__(idx)

    def __getitem__(self, idx):
        meta, eeg = super().__getitem__(idx).values()
        label: str | int = meta["label"]

        assert isinstance(
            label, int
        ), f"EEG_CLASSIFY_BASE_DATASET:GETITEM:ASSERTION:VALUE_ERROR: label must be an integer, got {type(label)}"

        if self.transform:
            label = self.transform(label, meta, whom="label", when="before_returning")

        meta["label"] = label

        # Remove unnecessary fields
        for field in ["env", "mel", "wav"]:
            if field in meta:
                del meta[field]
            if f"{field}_fs" in meta:
                del meta[f"{field}_fs"]

        return {"meta": meta, "eeg": eeg, "label": label}
