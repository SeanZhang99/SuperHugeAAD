import os
from collections.abc import Mapping
from typing import Literal

import numpy as np

from ..metadata_processing.data import RegressionMetadataElement
from .eeg_dataset import EegDataset

ENV_ALIASE = ["env", "envelope", "env_path"]
MEL_ALIASE = ["mel", "mel spectrum", "mfcc", "mel_path"]


class EegRegressionBaseDataset(EegDataset):
    metadata_cls = RegressionMetadataElement

    def __init__(self, /, **kwargs):
        """ """
        for metadata_field in kwargs["metadata_fields"]:
            # 处理支持的语音特征别名
            if metadata_field in ENV_ALIASE:
                self.speech_feature_type = "env"
                break
            elif metadata_field in MEL_ALIASE:
                self.speech_feature_type = "mel"
                break
        else:
            raise ValueError(
                f"EEG_REGRESSION_BASE_DATASET:__INIT__:VALUE_ERROR: Supported speech feature not found. Supported values are {ENV_ALIASE+MEL_ALIASE}."
            )

        super().__init__(**kwargs)

        self.speech_feature_path = os.path.join(
            self.eeg_path.replace("eeg", "stimuli"), self.speech_feature_type
        )

    def load_data(self, idx) -> Mapping[  # type: ignore
        Literal["meta", "eeg", "audio"],
        np.ndarray | np.memmap | RegressionMetadataElement,
    ]:
        """
        加载样本数据，并返回元数据、EEG段、语音特征段和标签。
        """
        item = super().load_data(idx)
        meta: RegressionMetadataElement = item["meta"]  # type: ignore
        eeg: np.ndarray | np.memmap = item["eeg"]  # type: ignore

        assert isinstance(
            meta, RegressionMetadataElement
        ), f"EEG_REGRESSION_BASE_DATASET:load_data:ASSERTION:TYPE_ERROR: meta must be a RegressionMetadataElement, got {type(meta)}"

        entry = meta.entry
        # 加载语音特征
        speech_feature: np.ndarray = np.load(
            os.path.join(
                self.speech_feature_path,
                f"{entry}_{self.speech_feature_type}.npy",
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        _, start_idx = self._map_idx_to_file_and_segment(idx)

        meta.__setattr__("speech_feature_type", self.speech_feature_type)

        if self.transform:
            speech_feature = self.transform(
                speech_feature, meta=meta, whom="audio", when="before_slicing"
            )[
                0
            ]  # type: ignore

        # 根据 segment_length 和 overlap 截取语音特征段
        stride = self.segment_length // self.overlap
        speech_segment: np.ndarray = speech_feature[
            start_idx : start_idx + self.segment_length
        ].copy()

        if self.transform:
            speech_segment = self.transform(
                speech_segment, meta=meta, whom="audio", when="before_returning"
            )[
                0
            ]  # type: ignore

        return {"meta": meta, "eeg": eeg, "audio": speech_segment.astype(np.float32)}

    def __getitem__(  # type: ignore
        self, idx
    ) -> Mapping[Literal["meta", "eeg", "audio"], np.ndarray | dict]:
        item = self.load_data(idx)
        meta: dict = item["meta"].model_dump()  # type: ignore
        eeg: np.ndarray | np.memmap = item["eeg"]  # type: ignore
        speech_segment: np.ndarray | np.memmap = item["audio"]  # type: ignore
        if "label" in meta.keys():
            del meta["label"]

        for field in ["env", "mel", "wav"]:
            if field != self.speech_feature_type:
                if field in meta:
                    del meta[field]
                if f"{field}_fs" in meta:
                    del meta[f"{field}_fs"]

        assert (
            eeg.shape[0] == speech_segment.shape[0]
        ), f"EEG and speech segment lengths do not match: {eeg.shape[0]} vs {speech_segment.shape[0]} in {meta['entry']}."

        return {"meta": meta, "eeg": eeg, "audio": speech_segment}
