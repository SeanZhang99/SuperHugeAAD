import os
from typing import Sequence

import numpy as np
import tqdm
from torch.utils.data import Dataset

from ...utils.validate import validate_kwargs
from ..metadata_processing.data import MetaData, MetaDataElement, MetaDataField
from ..transforms.composer import TransformComposer


class EegDataset(Dataset):
    metadata_cls = MetaDataElement

    def __init__(self, **kwargs):
        """
        Args:
            eeg_path (str): 数据集文件夹路径。
            files (list): 文件名列表。
            metadata (dict): 包含每个试次的元信息。
            segment_length (int): 截取的信号段长度。
            overlap (int): 重叠比例，决定截取步长。
            transform (Transform | None): 应用在样本上的变换函数。
            metadata_fields (list): 需要记录的元数据字段。
        """
        required_keys = ["eeg_path", "files", "metadata", "metadata_fields"]
        self._validate_kwargs(kwargs.keys(), required_keys)

        self.eeg_path: str = kwargs["eeg_path"]
        self.files: Sequence[str] = kwargs["files"]
        self.metadata: MetaData = kwargs["metadata"]
        self.segment_length: int = kwargs.get("fs", 128) * kwargs.get(
            "window_length", 10
        )  # 默认截取长度为1280
        self.overlap: int = kwargs.get("overlap", 1)  # 默认无重叠
        self.transform: TransformComposer | None = kwargs.get("transform", None)
        self.metadata_fields: list[MetaDataField] = kwargs["metadata_fields"]

        # Ensure files are a subset of metadata's keys
        assert set(self.files).issubset(
            self.metadata.keys()
        ), f"Some files specified in `files` do not have corresponding metadata: {set(self.files) - set(self.metadata.keys())}"

        # 计算总样本数目
        self.count_samples()

        self._copy_before_slicing = self.copy_before_slicing()

    def copy_before_slicing(self):
        if self.transform:
            for transform in self.transform.transforms:
                if transform.when == "before_slicing":
                    return True
        return False

    def count_samples(self):

        self.total_samples = 0
        for file in self.files:
            trial_length = self.metadata[file].signal_length
            assert (
                trial_length
            ), f"EEG_DATASET:COUNT_SAMPLES:TRIAL_LENGTH_ERROR: Trial length is not provided for file {file}."
            stride = self.segment_length // self.overlap
            self.total_samples += max(
                0, (trial_length - self.segment_length) // stride + 1
            )

    def __len__(self):
        return self.total_samples

    @property
    def len(self):
        return len(self)

    def __getitem__(self, idx):
        """
        加载样本数据，并返回元数据、信号段和标签。
        """
        file_idx, segment_idx = self._map_idx_to_file_and_segment(idx)
        file_name = self.files[file_idx]
        file_path = os.path.join(self.eeg_path, file_name + ".npy")

        meta = self.metadata[file_name].model_dump()

        eeg: np.ndarray | np.memmap = np.load(
            file_path, mmap_mode="r", allow_pickle=False
        )

        assert (
            eeg.shape[0] == self.metadata[file_name].signal_length
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The shape of the loaded data {eeg.shape} does not match the expected shape {self.metadata[file_name].signal_length}. Problem given with metadata {self.metadata[file_name].model_dump()}"
        assert (
            eeg.ndim == 2
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The loaded data is not 2D, but {eeg.ndim}D. Problem given with metadata {self.metadata[file_name].model_dump()}"
        assert (
            eeg.shape[1] == self.metadata[file_name].channel_infos.__len__()
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The number of channels in the loaded data {eeg.shape[1]} does not match the expected number {self.metadata[file_name].channel_infos.__len__()}. Problem given with metadata {self.metadata[file_name].model_dump()}"

        if self.transform:
            eeg = self.transform(eeg, meta, when="before_slicing", whom="eeg")

        # 加载信号和标签

        # 根据 segment_length 和 overlap 截取信号段
        stride = self.segment_length // self.overlap
        start_idx = segment_idx * stride
        eeg_seg = eeg[start_idx : start_idx + self.segment_length]

        # 应用变换
        if self.transform:
            eeg_seg = self.transform(eeg_seg, meta, when="before_returning", whom="eeg")

        # 获取元数据

        return {"meta": meta, "eeg": eeg_seg.astype(np.float32)}

    def _map_idx_to_file_and_segment(self, idx: int):
        """
        根据全局索引映射到具体文件和信号段索引。

        Args:
            idx (int): 全局索引。

        Returns:
            tuple: 文件索引和信号段索引。
        """

        cumulative: int = 0
        for file_idx, file in enumerate(self.files):
            file_meta = self.metadata[file]
            trial_length: int = file_meta.signal_length
            assert trial_length
            stride: int = self.segment_length // self.overlap
            num_segments: int = max(
                0, (trial_length - self.segment_length) // stride + 1
            )
            if cumulative + num_segments > idx:
                segment_idx = idx - cumulative
                return file_idx, segment_idx
            cumulative += num_segments
        raise IndexError(
            "EEG_DATASET:MAP_IDX_TO_FILE_AND_SEGMENT:INDEX_ERROR: After consuming all files, a valid file_idx and segment_idx pair was not found."
        )

    def _validate_kwargs(self, kwargs: set[str], required_keys: list[str]):
        validate_kwargs(kwargs, required_keys)


if __name__ == "__main__":
    root_path = r"E:/derivatives/SuperHuge"
    datasets = EegDataset().create_datasets(root_path=root_path)
    (x for x in tqdm.tqdm(datasets))
    pass
