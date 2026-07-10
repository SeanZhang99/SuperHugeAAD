from collections.abc import Mapping
from typing import Literal

import numpy as np

from .eeg_classify_base_dataset import EegClassifyBaseDataset


class EegClassifyDatasetWithSpectrum(EegClassifyBaseDataset):
    def __init__(self, **kwargs):
        """
        Args:
            spectrum_key (str): 提取谱图的元数据键，支持 'Pssl' 或 'Pmvdr'。
        """
        super().__init__(**kwargs)
        required_keys = ["spectrum_key"]
        self._validate_kwargs(list(kwargs.keys()), required_keys)
        self.spectrum_key = kwargs["spectrum_key"]
        assert (
            self.spectrum_key == "Pssl" or self.spectrum_key == "Pmvdr"
        ), f"{self.spectrum_key} is not supported currently."

    def __getitem__(  # type: ignore
        self, idx
    ) -> Mapping[Literal["meta", "eeg", "label", "spectrum"], np.ndarray | dict]:
        """
        加载样本数据，并返回元数据、信号段、标签和谱图。
        """
        item = super().load_data(idx)
        meta: dict = item["meta"].model_dump()  # type: ignore
        eeg: np.ndarray = item["eeg"]  # type: ignore
        label: np.ndarray = item["label"]  # type: ignore

        # 提取谱图
        spectrum = getattr(meta, self.spectrum_key, None)
        if spectrum is None:
            raise AttributeError(
                f"EEG_CLASSIFY_DATASET_WITH_SPECTRUM:__GETITEM__:GETATTR:ATTRIBUTE_ERROIR: Spectrum key '{self.spectrum_key}' not found in metadata for file {self.files[idx]}."
            )

        return {"meta": meta, "eeg": eeg, "label": label, "spectrum": spectrum}
