import os
from pydantic import BaseModel
import numpy as np

from ..commons.eeg_dataset import EegDataset
from ..metadata_processing.data import RegressionMetaDataElement


ENV_ALIASE = ["env", "envelope", "env_path"]
MEL_ALIASE = ["mel", "mel spectrum", "mfcc", "mel_path"]


class EEGDatasetWithSpeechFeatureCreationConfig(BaseModel, extra="allow"):
    speech_feature_key: str


class EegRegressionBaseDataset(EegDataset):
    metadata_cls = RegressionMetaDataElement

    def __init__(self, /, **kwargs):
        """
        Args:
            speech_feature_key (str): 元数据中存储语音特征文件名的键。
        """
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

    def __getitem__(self, idx):
        """
        加载样本数据，并返回元数据、EEG段、语音特征段和标签。
        """
        meta, eeg = super().__getitem__(idx).values()

        entry = meta["entry"]
        # 加载语音特征
        speech_feature: np.ndarray = np.load(
            os.path.join(
                self.speech_feature_path,
                f"{entry}_{self.speech_feature_type}.npy",
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        _, segment_idx = self._map_idx_to_file_and_segment(idx)

        if self.transform:
            speech_feature = self.transform(
                speech_feature, meta, whom="audio", when="before_slicing"
            )

        # 根据 segment_length 和 overlap 截取语音特征段
        stride = self.segment_length // self.overlap
        start_idx = segment_idx * stride
        speech_segment = speech_feature[start_idx : start_idx + self.segment_length]
        if speech_segment.ndim == 1:
            speech_segment = speech_segment[:, np.newaxis]

        if self.transform:
            speech_segment = self.transform(
                speech_segment, meta, whom="audio", when="before_returning"
            )

        if "label" in meta:
            del meta["label"]

        for field in ["env", "mel", "wav"]:
            if field != self.speech_feature_type:
                if field in meta:
                    del meta[field]
                if f"{field}_fs" in meta:
                    del meta[f"{field}_fs"]

        return {"meta": meta, "eeg": eeg, "audio": speech_segment.astype(np.float32)}
