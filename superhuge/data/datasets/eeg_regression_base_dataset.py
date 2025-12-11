import os
from collections.abc import Mapping
from typing import Literal

import numpy as np

from ..metadata_processing.data import RegressionMetadataElement
from .eeg_dataset import EegDataset

ENV_ALIASE = ["env", "envelope"]
MEL_ALIASE = ["mel", "mel spectrum", "mfcc"]
WAVEFORM_ALIASE = ["wav", "waveform", "audio", "wav_path", "raw"]


class EegRegressionBaseDataset(EegDataset):
    metadata_cls = RegressionMetadataElement
    speech_feature_types: list[str] = []
    speech_feature_paths: list[str] = []

    def __init__(self, /, **kwargs):
        """ """
        super().__init__(**kwargs)
        for metadata_field in kwargs["metadata_fields"]:
            # 处理支持的语音特征别名
            if metadata_field in ENV_ALIASE:
                self.speech_feature_types.append("env")
            elif metadata_field in MEL_ALIASE:
                self.speech_feature_types.append("mel")
            elif metadata_field in WAVEFORM_ALIASE:
                self.speech_feature_types.append("wav")
            elif metadata_field.startswith("wav2vec"):
                self.speech_feature_types.append(metadata_field)
        self.speech_feature_types = type(self.speech_feature_types)(
            set(self.speech_feature_types)
        )
        if len(self.speech_feature_types) == 0:
            raise ValueError(
                "EEG_REGRESSION_BASE_DATASET:INIT:VALUE_ERROR: No valid speech feature type found in metadata_fields."
            )

        self.speech_feature_paths = [
            os.path.join(self.eeg_path.replace("eeg", "stimuli"), t)
            for t in self.speech_feature_types
        ]

    def _get_file_disk_usage(self):
        total_bytes_used = 0
        for file_name in self.files:
            for p, t in zip(self.speech_feature_paths, self.speech_feature_types):
                speech_file_path = os.path.join(p, f"{file_name}_{t}.npy")
                total_bytes_used += os.path.getsize(speech_file_path)
        return total_bytes_used + super()._get_file_disk_usage()

    def load_data(self, idx: int) -> Mapping[  # type: ignore
        Literal["meta", "eeg", "env", "mel", "wav"],
        np.ndarray | np.memmap | RegressionMetadataElement,
    ]:
        """
        加载样本数据，并返回元数据、EEG段、语音特征段和标签。
        """
        if (
            self._save_on_memory
            and idx in self.memory
            and {"meta", "eeg"}
            .union(set(self.speech_feature_types))
            .issubset(set(self.memory[idx]))
        ):
            data = self.memory[idx]
        else:
            item = super(EegRegressionBaseDataset, self).load_data(idx)
            meta: RegressionMetadataElement = item["meta"]  # type: ignore
            eeg: np.ndarray | np.memmap = item["eeg"]  # type: ignore

            assert isinstance(
                meta, RegressionMetadataElement
            ), f"EEG_REGRESSION_BASE_DATASET:load_data:ASSERTION:TYPE_ERROR: meta must be a RegressionMetadataElement, got {type(meta)}"

            entry = meta.entry

            data: dict = {}
            data["meta"] = meta
            data["eeg"] = eeg
            meta.__setattr__("speech_feature_types", self.speech_feature_types)
            # 加载语音特征
            for speech_type in self.speech_feature_types:
                speech_feature: np.ndarray = np.load(
                    os.path.join(
                        self.speech_feature_paths[
                            self.speech_feature_types.index(speech_type)
                        ],
                        f"{entry}_{speech_type}.npy",
                    ),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                _, start_idx = self._map_idx_to_file_and_segment(idx)

                if self.transform:
                    speech_feature, meta = self.transform(
                        speech_feature,
                        meta=meta,
                        whom=speech_type,
                        when="before_slicing",
                    )  # type: ignore

                speech_segment: np.ndarray = (
                    speech_feature[
                        start_idx : start_idx + self.window_length * getattr(meta, f"{speech_type}_fs")  # type: ignore
                    ]
                    .copy()
                    .astype(np.float32)
                )

                if self.transform:
                    speech_segment, meta = self.transform(
                        speech_segment,
                        meta=meta,
                        whom=speech_type,
                        when="before_returning",
                    )  # type: ignore

                data[speech_type] = speech_segment

            if self._save_on_memory:
                self.memory[idx] = data

        return data  # type: ignore

    def __getitem__(self, idx) -> Mapping[str, np.ndarray | dict]:  # type: ignore
        item = self.load_data(idx)
        meta: dict = item["meta"].model_dump()  # type: ignore
        if "label" in meta.keys():
            del meta["label"]

        for field in ["env", "mel", "wav"]:
            if field not in self.speech_feature_types:
                if field in meta:
                    del meta[field]
                if f"{field}_fs" in meta:
                    del meta[f"{field}_fs"]

        return {"meta": meta, **{k: v for k, v in item.items() if k != "meta"}}  # type: ignore
