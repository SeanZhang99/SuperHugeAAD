import inspect
import os
import pickle
import gc
from collections import namedtuple
from collections.abc import Callable, Iterable
from importlib import import_module
from typing import Any, Sequence

import numpy as np
import torch
import tqdm
from pydantic import BaseModel
from torch.utils.data import Dataset

torch.serialization.add_safe_globals([np.ndarray])

from ...utils.transforms.abc import Transform
from ..metadata_processing.data import (
    ClassifyMetaDataElement,
    DatasetSubjectTrialEntry,
    GroupingFunction,
    MetaData,
    MetaDataElement,
    MetaDataField,
    RegressionMetaDataElement,
)
from ..metadata_processing.filters.composer import MetaDataFilterComposer
from ..metadata_processing.group import loto


class CreateDatasetsInputConfig(BaseModel):
    meta_path: str
    exg_path: str
    meta_filter_func: (
        Callable[[ClassifyMetaDataElement], ClassifyMetaDataElement | None]
        | Callable[[RegressionMetaDataElement], RegressionMetaDataElement | None]
        | None
    )
    meta_group_func: GroupingFunction
    test_fold_idx: int
    val_fold_idx: int
    n_folds: int
    window_length: int
    fs: int
    overlap: int
    transform: Sequence[Transform] | None = None
    metadata_fields: list[MetaDataField]


class EegDataset(Dataset):
    metadata_cls = MetaDataElement

    @classmethod
    def create_datasets(
        cls,
        /,
        root_path: str,
        meta_filter_func: MetaDataFilterComposer | None = None,
        meta_filter_func_args: list = [],
        meta_group_func: GroupingFunction | None = None,
        test_fold_idx: int = 0,
        val_fold_idx: int = 1,
        n_folds: int = 5,
        window_length: int = 10,
        fs: int = 128,
        overlap: int = 1,
        metadata_fields: list[MetaDataField] = [
            "dataset_id",
            "subject_id",
            "trial_id",
            "signal_length",
            "num_channel",
            "fs",
        ],
        transform: Sequence[dict[str, str | dict[str, float]]] | None = None,
        preproc_stage: str | None = None,
        **kwargs,
    ):
        """
        创建 train/val/test 数据集实例。

        Args:
            meta_path (str): 元信息文件路径。
            exg_path (str): 数据集文件夹路径。
            group_func (Callable): 用于分组的键。
            fold_idx (int): 当前 fold 索引。
            n_folds (int): 总 fold 数。
            random_seed (int): 随机种子。
            window_length (int): 截取的信号段长度（秒）。
            fs (int): 采样频率。
            overlap (int): 重叠比例。
            transform (Sequence[Transform] | dict[str, Any] | None): 数据变换。
            metadata_fields (list): 需要记录的元数据字段。

        Returns:
            tuple: 包含 train, val, test 数据集的元组。
        """
        if isinstance(meta_filter_func, dict):
            module_name, func_name = meta_filter_func["class_path"].rsplit(".", 1)
            meta_filter_class = getattr(import_module(module_name), func_name)
            filters = []
            for filter_args in meta_filter_func["init_args"]:
                filter = getattr(
                    import_module(name=filter_args["class_path"].rsplit(".", 1)[0]),
                    filter_args["class_path"].rsplit(".", 1)[1],
                )
                if inspect.isclass(filter):
                    filters.append(filter(**filter_args["init_args"]))
                else:
                    filters.append(filter)
            meta_filter_func = meta_filter_class(*filters)

        if isinstance(meta_group_func, str):
            module_name, func_name = meta_group_func.rsplit(".", 1)
            meta_group_func = getattr(import_module(module_name), func_name)

        if transform:
            assert isinstance(
                transform, list
            ), f"transform must be a list, but got {type(transform)}"
            transforms: list[Transform] | None = [getattr(import_module(kv["class_path"].rsplit(".", 1)[0]), kv["class_path"].rsplit(".", 1)[1])(**kv["init_args"]) for kv in transform]  # type: ignore
        else:
            transforms = None

        meta_group_func = loto if meta_group_func is None else meta_group_func
        metadata_fields = list(
            set(metadata_fields) | set(cls.metadata_cls.model_fields.keys())
        )

        assert (
            0 <= int(test_fold_idx) < int(n_folds)
        ), f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: fold_idx must be in the range [0, {n_folds}), but got {test_fold_idx}"

        assert (
            0 <= int(val_fold_idx) < int(n_folds)
        ), f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: fold_idx must be in the range [0, {n_folds}), but got {val_fold_idx}"

        assert (
            val_fold_idx != test_fold_idx
        ), f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: val_fold_idx and test_fold_idx must be different, but got {val_fold_idx} and {test_fold_idx}"

        if preproc_stage is None:
            preproc_stage = "preprocessed"

        config = CreateDatasetsInputConfig(
            meta_path=os.path.join(root_path, preproc_stage, "meta", "metadata.pkl"),
            exg_path=os.path.join(root_path, preproc_stage, "exg"),
            meta_filter_func=cls.meta_filter_func_parser(
                meta_filter_func, *meta_filter_func_args
            ),
            meta_group_func=meta_group_func,
            test_fold_idx=test_fold_idx,
            val_fold_idx=val_fold_idx,
            n_folds=n_folds,
            window_length=window_length,
            fs=fs,
            overlap=overlap,
            transform=transforms,
            metadata_fields=metadata_fields,
            **kwargs,
        )

        metadata = cls.load_metadata(
            metafile_path=config.meta_path,
            metadata_fields=config.metadata_fields,
        )

        metadata = cls.filt_metadata(metadata, config.meta_filter_func)

        splits = config.meta_group_func(
            metadata=metadata,
            val_fold_idx=config.val_fold_idx,
            test_fold_idx=config.test_fold_idx,
            n_folds=config.n_folds,
        )

        dataset_modes = ["train", "val", "test"]

        datasets = [
            cls(
                exg_path=config.exg_path,
                files=splits[mode],
                metadata=metadata,
                fs=config.fs,
                window_length=config.window_length,
                # overlap=config.overlap if mode == "train" else 1,
                overlap=config.overlap,
                transform=config.transform if mode == "train" else None,
                metadata_fields=config.metadata_fields,
                **kwargs,
            )
            for mode in dataset_modes
        ]

        p = namedtuple("datasets", ["train", "val", "test"])

        return p(*datasets)

    @classmethod
    def filt_metadata(cls, metadata: MetaData, meta_filter_func: Callable | None):
        if meta_filter_func:
            filtered_metadata: MetaData = {}
            for dataset_entry, metadata_element in metadata.items():
                metadata_element = meta_filter_func(metadata_element)
                if metadata_element is not None:
                    filtered_metadata[dataset_entry] = metadata_element
            return filtered_metadata
        else:
            return metadata

    @classmethod
    def meta_filter_func_parser(
        cls, meta_filter_func: Callable | None, *args, **kwargs
    ):
        """
        meta_filter_func_parser

        Args:
            meta_filter_func (Callable | None): 元数据过滤函数。
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            Callable | None: 元数据过滤函数。

        Raises:
            TypeError: meta_filter_func 必须是可调用对象或 None。

        Running logics:
            - 如果 meta_filter_func 是 None，则返回 None。
            - 如果 meta_filter_func 是可调用对象:
                - 获取 meta_filter_func 的签名。
                - 如果 meta_filter_func 的返回注释是 ClassifyMetaDataElement 或 RegressionMetaDataElement 或 None，则返回 meta_filter_func。在这种情况下, meta_filter_func 是一个有效的元数据过滤函数。
                - 如果 meta_filter_func 的返回注释是可调用对象，则返回 meta_filter_func 的调用结果。在这种情况下, meta_filter_func 是一个元数据过滤函数的工厂函数。
            - 否则，抛出 TypeError。
        """
        if meta_filter_func is None:
            return None
        elif isinstance(meta_filter_func, Callable):
            return_args = inspect.signature(meta_filter_func).return_annotation
            if isinstance(return_args, Callable):
                return meta_filter_func(*args, **kwargs)
            else:
                for return_arg in return_args.__args__:
                    if not (issubclass(return_arg, (MetaDataElement, type(None)))):
                        raise TypeError(
                            f"EEG_DATASET:META_FILTER_FUNC_PARSER:TYPE_ERROR: meta_filter_func must: \n1. return a MetaDataElement or None, or \n2. be a factory function returning a callable of (1.), \nbut got {return_args}"
                        )
                return meta_filter_func
        else:
            raise TypeError(
                f"EEG_DATASET:META_FILTER_FUNC_PARSER:TYPE_ERROR: meta_filter_func must be a callable or None, but got {type(meta_filter_func)}"
            )

    @classmethod
    def load_metadata(cls, metafile_path, metadata_fields) -> MetaData:
        """
        从 meta.mat 文件中加载元信息。

        Args:
            metafile_path (str): 元信息文件路径。

        Returns:
            dict[entry->metadata_cls]: 转换后的元信息字典。
        """

        with open(metafile_path, "rb") as f:
            metadata: dict[DatasetSubjectTrialEntry, dict[str, Any]] = pickle.load(f)

        # convert to MetaData object.
        meta_dict = {}
        for entry, data in metadata.items():
            if set(metadata_fields).issubset(set(data.keys())):
                meta_dict[entry] = cls.metadata_cls(**data)
        return meta_dict

    def __init__(self, **kwargs):
        """
        Args:
            exg_path (str): 数据集文件夹路径。
            files (list): 文件名列表。
            metadata (dict): 包含每个试次的元信息。
            segment_length (int): 截取的信号段长度。
            overlap (int): 重叠比例，决定截取步长。
            transform (Transform | None): 应用在样本上的变换函数。
            metadata_fields (list): 需要记录的元数据字段。
        """
        required_keys = ["exg_path", "files", "metadata", "metadata_fields"]
        self._validate_kwargs(kwargs.keys(), required_keys)

        self.exg_path: str = kwargs["exg_path"]
        self.files: Sequence[str] = kwargs["files"]
        self.metadata: MetaData = kwargs["metadata"]
        self.segment_length: int = kwargs.get("fs", 128) * kwargs.get(
            "window_length", 10
        )  # 默认截取长度为1280
        self.overlap: int = kwargs.get("overlap", 1)  # 默认无重叠
        self.transform: Sequence[Transform] | None = kwargs.get("transform", None)
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
            for transform in self.transform:
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
        file_path = os.path.join(self.exg_path, file_name + ".npy")

        exg: np.ndarray | np.memmap = np.load(
            file_path, mmap_mode="r", allow_pickle=False
        )

        assert (
            exg.shape[0] == self.metadata[file_name].signal_length
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The shape of the loaded data {exg.shape} does not match the expected shape {self.metadata[file_name].signal_length}. Problem given with metadata {self.metadata[file_name].model_dump()}"
        assert (
            exg.ndim == 2
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The loaded data is not 2D, but {exg.ndim}D. Problem given with metadata {self.metadata[file_name].model_dump()}"
        assert (
            exg.shape[1] == self.metadata[file_name].channel_infos.__len__()
        ), f"EEG_DATASET:GETITEM:SHAPE_ERROR: The number of channels in the loaded data {exg.shape[1]} does not match the expected number {self.metadata[file_name].channel_infos.__len__()}. Problem given with metadata {self.metadata[file_name].model_dump()}"

        if self.transform:
            for transform in self.transform:
                if transform.when == "before_slicing" and (
                    "eeg" in transform.whom or "all" in transform.whom
                ):
                    exg = transform(exg)

        # 加载信号和标签

        # 根据 segment_length 和 overlap 截取信号段
        stride = self.segment_length // self.overlap
        start_idx = segment_idx * stride
        exg_seg = exg[start_idx : start_idx + self.segment_length]

        # 应用变换
        if self.transform:
            for transform in self.transform:
                if transform.when == "before_returning" and (
                    "eeg" in transform.whom or "all" in transform.whom
                ):
                    exg_seg = transform(exg_seg)

        # 获取元数据
        meta = self.metadata[file_name].model_dump()

        return {"meta": meta, "exg": exg_seg.astype(np.float32)}

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

    def _validate_kwargs(self, kwargs: Iterable, required_keys: Iterable):
        """
        验证 kwargs 是否包含所有必需的键。
        """
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    f"EEG_DATASET:VALIDATE_KWARGS:KEY_ERROR: Missing required key '{key}' in kwargs {kwargs}. You should go back to the caller function and check the kwargs to be validated"
                )


if __name__ == "__main__":
    root_path = r"E:/derivatives/SuperHuge"
    datasets = EegDataset().create_datasets(root_path=root_path)
    (x for x in tqdm.tqdm(datasets))
    pass
