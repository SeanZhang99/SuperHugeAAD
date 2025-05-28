# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 Yuanming Zhang
# This package is adopted based on Pytorch Lightning Template project.

import os
import pickle
from collections.abc import Callable, Sequence
from typing import Any
import warnings

import lightning as pl2
from pydantic import BaseModel, model_validator
from rich.console import Console
from rich.table import Table
import torch
from torch.utils.data import DataLoader

from ..transforms.resample import Resample

from ..datasets.eeg_dataset import EegDataset
from ..datasets.eeg_regression_base_dataset import EegRegressionBaseDataset
from ..datasets.eeg_classify_base_dataset import EegClassifyBaseDataset
from ..datasets.collect_multidataset import collect_multidataset
from ..filters.abc import (
    MetadataFilter,
    ClassifyMetadataFilter,
    RegressionMetadataFilter,
)
from ..filters.classify_filter import get_classify_filter
from ..filters.regress_filter import get_regression_filter
from ..filters.composer import MetaDataFilterComposer
from ..metadata_processing.data import (
    ClassifyMetaDataElement,
    MetaData,
    MetaDataElement,
    MetaDataField,
    GroupingFunction,
    DatasetSubjectTrialEntry,
)
from ..metadata_processing.group import (
    leave_one_out_input_decorator,
    loto,
    test_kul_loto,
)
from ..transforms.composer import TransformComposer
from ..transforms.abc import Transform
from ..transforms.scale import Scale
from ..transforms.filter import Filter


class CreateDatasetsInputConfig(BaseModel):
    dataset_class: type[EegDataset]
    meta_path: str
    eeg_path: str
    meta_filter_func: MetaDataFilterComposer | None = None
    meta_group_func: GroupingFunction
    test_fold_idx: int
    val_fold_idx: int
    n_folds: int
    window_length: int | float
    fs: int
    overlap: int
    transform: TransformComposer | None = None
    metadata_fields: list[MetaDataField]

    @model_validator(mode="after")
    def check_fold_indices(self) -> "CreateDatasetsInputConfig":
        if not (0 <= self.test_fold_idx < self.n_folds):
            raise ValueError(
                f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: fold_idx must be in the range [0, {self.n_folds}), "
                f"but got {self.test_fold_idx}"
            )
        if not (0 <= self.val_fold_idx < self.n_folds):
            raise ValueError(
                f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: fold_idx must be in the range [0, {self.n_folds}), "
                f"but got {self.val_fold_idx}"
            )
        if self.test_fold_idx == self.val_fold_idx:
            raise ValueError(
                f"EEG_DATASET:CREATE_DATASETS:FOLD_IDX_ERROR: val_fold_idx and test_fold_idx must be different, "
                f"but got {self.val_fold_idx} and {self.test_fold_idx}"
            )
        return self


def create_matlab_dinterface():
    return DInterface(
        dataset_class=EegClassifyBaseDataset,
        dataloader_args={
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
        root_path=r"E:\derivatives\SuperHuge",
        window_length=10.0,
        fs=128,
        meta_filter_func=None,
        meta_filter_func_args=["binary_leftright"],
        meta_group_func=test_kul_loto,
        transform=[
            Filter([1.0, 32.0], fs=128, btype="bandpass", order=5),
            Scale(root_path=r"E:\derivatives\SuperHuge", preproc_stage="raw"),
        ],
        preproc_stage="raw",
        metadata_fields=["label"],
        overlap=2,
    )


class DInterface(pl2.LightningDataModule):

    def __init__(
        self,
        /,
        dataset_class: type[EegDataset],
        dataloader_args: dict,
        root_path: str,
        window_length: int | float,
        fs: int,
        meta_filter_func: MetadataFilter | Sequence[MetadataFilter] | None = None,
        meta_filter_func_args: list | None = None,
        meta_group_func: Callable | None = None,
        test_fold_idx: int = 0,
        val_fold_idx: int = 1,
        n_folds: int = 5,
        overlap: int = 1,
        metadata_fields: list[MetaDataField] | None = None,
        transform: Transform | Sequence[Transform] | None = None,
        preproc_stage: str | None = None,
        **kwargs,
    ):
        super().__init__()

        if not dataloader_args.get("num_workers", None):
            if "prefetch_factor" in dataloader_args:
                del dataloader_args["prefetch_factor"]
            if "persistent_workers" in dataloader_args:
                del dataloader_args["persistent_workers"]

        self.dataloader_args = dataloader_args

        preproc_stage = preproc_stage or "preprocessed"
        meta_group_func = meta_group_func or loto

        if metadata_fields is None:
            metadata_fields = []

        metadata_fields.extend(
            [
                "dataset_id",
                "subject_id",
                "trial_id",
                "fs",
                "num_channel",
                "signal_length",
                "channel_infos",
            ]
        )

        self.dataset_cfg = CreateDatasetsInputConfig(
            dataset_class=dataset_class,
            meta_path=os.path.join(root_path, preproc_stage, "meta", "metadata.pkl"),
            eeg_path=os.path.join(root_path, preproc_stage, "eeg"),
            meta_filter_func=self.meta_filter_func_parser(
                dataset_class,
                meta_filter_func,
                *meta_filter_func_args if meta_filter_func_args else [],
            ),
            meta_group_func=leave_one_out_input_decorator(meta_group_func),
            test_fold_idx=test_fold_idx,
            val_fold_idx=val_fold_idx,
            n_folds=n_folds,
            window_length=window_length,
            fs=fs,
            overlap=overlap,
            transform=(
                TransformComposer(
                    *(transform if isinstance(transform, Sequence) else [transform])
                )
                if transform
                else None
            ),
            metadata_fields=list(set(metadata_fields)),
        )

        self.kwargs = kwargs

        self.create_datasets()
        self.print_summary()

    def meta_filter_func_parser(
        self,
        /,
        dataset_class: type[EegDataset],
        meta_filter_func: MetadataFilter | Sequence[MetaDataField] | None,
        *args,
        **kwargs,
    ) -> MetaDataFilterComposer | None:
        """
        meta_filter_func_parser

        Args:
            meta_filter_func (MetaDataFilterComposer | None): Metadata filter composer instance.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            MetaDataFilterComposer | None: Metadata filter composer instance.

        Raises:
            TypeError: If meta_filter_func is not an instance of MetaDataFilterComposer or None.

        Running logics:
            - If meta_filter_func is None, return None.
            - If meta_filter_func is not an instance of MetaDataFilterComposer, raise TypeError.
            - If self.dataset_cfg.dataset_class is a type of EegClassifyBaseDataset and no ClassifyMetadataFilter exists in the composer, add a filter using the classify_filter factory function.
            - If self.dataset_cfg.dataset_class is a type of EegRegressionBaseDataset and no RegressionMetadataFilter exists in the composer, add a filter using the regression_filter factory function.
        """
        if meta_filter_func is None:
            meta_filter_func = MetaDataFilterComposer()
        else:
            meta_filter_func = MetaDataFilterComposer(*meta_filter_func)

        assert isinstance(meta_filter_func, MetaDataFilterComposer)

        # Check for classify dataset and add classify filter if missing
        if issubclass(dataset_class, EegClassifyBaseDataset):
            has_classify_filter = any(
                isinstance(f, ClassifyMetadataFilter) for f in meta_filter_func.filters
            )
            if not has_classify_filter:
                classify_filter = get_classify_filter(*args, **kwargs)
                meta_filter_func.add_filters(classify_filter)

        # Check for regression dataset and add regression filter if missing
        elif issubclass(dataset_class, EegRegressionBaseDataset):
            has_regression_filter = any(
                isinstance(f, RegressionMetadataFilter)
                for f in meta_filter_func.filters
            )
            if not has_regression_filter:
                regression_filter = get_regression_filter(*args, **kwargs)
                meta_filter_func.add_filters(regression_filter)

        return meta_filter_func

    def load_metadata(
        self,
        dataset_class: type[EegDataset],
        metafile_path: str,
        metadata_fields: Sequence[str],
    ) -> MetaData:
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
                meta_dict[entry] = dataset_class.metadata_cls(**data)
        return meta_dict

    def filt_metadata(
        self,
        metadata: MetaData,
        meta_filter_func: Callable[[MetaDataElement], MetaDataElement | None] | None,
    ):
        if meta_filter_func:
            filtered_metadata: MetaData = {}
            for dataset_entry, metadata_element in metadata.items():
                metadata_element = meta_filter_func(metadata_element)
                if metadata_element is not None:
                    filtered_metadata[dataset_entry] = metadata_element
            return filtered_metadata
        else:
            return metadata

    def create_datasets(self):
        metadata = self.load_metadata(
            dataset_class=self.dataset_cfg.dataset_class,
            metafile_path=self.dataset_cfg.meta_path,
            metadata_fields=self.dataset_cfg.metadata_fields,
        )

        metadata = self.filt_metadata(metadata, self.dataset_cfg.meta_filter_func)

        splits = self.dataset_cfg.meta_group_func(
            metadata=metadata,
            val_fold_idx=self.dataset_cfg.val_fold_idx,
            test_fold_idx=self.dataset_cfg.test_fold_idx,
            n_folds=self.dataset_cfg.n_folds,
        )

        dataset_modes = ["train", "val", "test"]

        self.trainset, self.valset, self.testset = (
            self.dataset_cfg.dataset_class(
                eeg_path=self.dataset_cfg.eeg_path,
                files=splits[mode],
                metadata=metadata,
                fs=self.dataset_cfg.fs,
                window_length=self.dataset_cfg.window_length,
                overlap=self.dataset_cfg.overlap,
                transform=self.dataset_cfg.transform,
                metadata_fields=self.dataset_cfg.metadata_fields,
                accept_range=splits.get(f"{mode}_accept_range", None),
                reject_range=splits.get(f"{mode}_reject_range", None),
                **self.kwargs,
            )
            for mode in dataset_modes
        )

    def create_dataloader(self, dataset, *args, **kwargs):
        return DataLoader(
            dataset,
            *args,
            **kwargs,
            **self.dataloader_args,
            collate_fn=collect_multidataset,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.trainset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.valset)

    def test_dataloader(self):
        return self.create_dataloader(self.testset)

    def print_summary(self):
        console = Console()

        # Table for dataset statistics
        stats_table = Table(title="Dataset Summary")
        stats_table.add_column("Set", justify="center", style="cyan", no_wrap=True)
        stats_table.add_column("# Subjects", justify="center", style="magenta")
        stats_table.add_column("# Trials", justify="center", style="green")
        stats_table.add_column("# Samples", justify="center", style="yellow")

        datasets = {
            "Train": self.trainset,
            "Validation": self.valset,
            "Test": self.testset,
        }

        unique_datasets = set()

        for name, dataset in datasets.items():
            num_subjects = len(
                set(
                    f"{meta.dataset_id}-{meta.subject_id}"
                    for meta in dataset.metadata.values()
                )
            )
            num_trials = len(dataset.files)
            num_samples = len(dataset)
            stats_table.add_row(
                name, str(num_subjects), str(num_trials), str(num_samples)
            )

            # Collect unique dataset names
            unique_datasets.update(
                entry.dataset_name for entry in dataset.metadata.values()
            )

        # Table for unique dataset names and IDs
        unique_table = Table(title="Unique Dataset Names")
        unique_table.add_column("Dataset Name", justify="center", style="cyan")

        for entry in sorted(unique_datasets):
            unique_table.add_row(entry)

        # class-wise sample count for classification dataset
        if issubclass(self.dataset_cfg.dataset_class, EegClassifyBaseDataset):
            class_count_table = Table(title="Class-wise Sample Count")
            class_count_table.add_column("Class", justify="center", style="cyan")
            class_count_table.add_column("Samples", justify="center", style="magenta")
            class_count_table.add_column("Percentage", justify="center", style="green")

            for name, dataset in datasets.items():
                class_counts = {}
                for file_idx, sample_counts in enumerate(
                    dataset._per_file_sample_count
                ):
                    metadata_element: ClassifyMetaDataElement = dataset.metadata[
                        dataset.files[file_idx]
                    ]
                    label = metadata_element.label
                    if label not in class_counts:
                        class_counts[label] = 0
                    class_counts[label] += sample_counts

                total_samples = sum(class_counts.values())
                for label, count in class_counts.items():
                    percentage = (count / total_samples) * 100
                    class_count_table.add_row(
                        str(f"{name}-{label}"), str(count), f"{percentage:.2f}%"
                    )

        console.print(stats_table, unique_table, class_count_table)

    @property
    def batch_size(self):
        return self.dataloader_args["batch_size"]

    @batch_size.setter
    def batch_size(self, value):
        assert isinstance(value, int), "Batch size must be an integer."
        self.dataloader_args["batch_size"] = value

    @property
    def fs(self):
        return self.dataset_cfg.fs

    @property
    def window_length(self):
        return self.dataset_cfg.window_length

    @property
    def datasets(self):
        return self.trainset, self.valset, self.testset


if __name__ == "__main__":
    # Test the DInterface class
    dinterface = create_matlab_dinterface()
    print(dinterface.dataset_cfg)
