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

from email.mime import audio
from logging import log
import logging
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from typing import Any
import warnings

import lightning as pl2
import numpy as np
from pydantic import BaseModel, model_validator
from rich.console import Console
from rich.table import Table
from sklearn.random_projection import sample_without_replacement
from torch.utils.data import DataLoader

from ..transforms.resample import Resample

from ..datasets.eeg_dataset import EegDataset
from ..datasets.eeg_regression_base_dataset import EegRegressionBaseDataset
from ..datasets.eeg_classify_base_dataset import EegClassifyBaseDataset
from ..datasets.collect_multidataset import collect_multidataset
from ..metadata_filters.abc import (
    MetadataFilter,
)
from ..metadata_filters.classify_filter import ClassifyMetadataFilter
from ..metadata_filters.regress_filter import RegressionMetadataFilter
from ..metadata_filters.classify_filter import get_classify_filter
from ..metadata_filters.regress_filter import get_regression_filter
from ..metadata_filters.composer import MetadataFilterComposer
from ..metadata_processing.data import (
    ClassifyMetadataElement,
    Metadata,
    MetadataElement,
    MetadataField,
    DatasetSubjectTrialEntry,
)
from ..metadata_processing.group import (
    loto,
)
from ..transforms.composer import TransformComposer
from ..transforms.abc import Transform


class CreateDatasetsInputConfig(BaseModel):
    dataset_class: type[EegDataset]
    meta_path: str
    eeg_path: str
    meta_filter_func: MetadataFilterComposer | None = None
    meta_group_func: Callable
    test_fold_idx: int
    val_fold_idx: int
    n_folds: int
    window_length: int | float
    fs: int
    overlap: int
    transform: TransformComposer | None = None
    metadata_fields: list[MetadataField]

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


class DInterface(pl2.LightningDataModule):

    def __init__(
        self,
        /,
        dataset_class: type[EegDataset],
        dataloader_args: dict[str, Any],
        root_path: str,
        window_length: int | float,
        fs: int,
        meta_filter_func: MetadataFilter | Sequence[MetadataFilter] | None = None,
        add_meta_filter_func: MetadataFilter | Sequence[MetadataFilter] | None = None,
        meta_filter_func_args: Sequence | None = None,
        meta_group_func: Callable | None = None,
        test_fold_idx: int = 0,
        val_fold_idx: int = 1,
        n_folds: int = 5,
        overlap: int = 1,
        metadata_fields: list[MetadataField] | None = None,
        transform: Transform | Sequence[Transform] | None = None,
        preproc_stage: str | None = None,
        summary_verbose: bool = False,
        dataset_args: dict[str, Any] | None = None,
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
                # "channel_infos",
            ]
        )

        self.dataset_cfg = CreateDatasetsInputConfig(
            dataset_class=dataset_class,
            meta_path=os.path.join(root_path, preproc_stage, "meta", "metadata.pkl"),
            eeg_path=os.path.join(root_path, preproc_stage, "eeg"),
            meta_filter_func=self.meta_filter_func_parser(
                dataset_class,
                meta_filter_func,
                add_meta_filter_func,
                *meta_filter_func_args if meta_filter_func_args else [],
            ),
            meta_group_func=meta_group_func,
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

        self.dataset_args = dataset_args

        self.create_datasets()
        self.summary(verbose=summary_verbose)
        self.summary_verbose = summary_verbose

    def meta_filter_func_parser(
        self,
        /,
        dataset_class: type[EegDataset],
        meta_filter_func: (
            MetadataFilter | Sequence[MetadataFilter] | MetadataFilterComposer | None
        ),
        add_meta_filter_func: (
            MetadataFilter | Sequence[MetadataFilter] | MetadataFilterComposer | None
        ),
        *args,
    ) -> MetadataFilterComposer | None:
        """
        meta_filter_func_parser

        Args:
            meta_filter_func (MetadataFilterComposer | None): Metadata filter composer instance.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            MetadataFilterComposer | None: Metadata filter composer instance.

        Raises:
            TypeError: If meta_filter_func is not an instance of MetadataFilterComposer or None.

        Running logics:
            - If meta_filter_func is None, return None.
            - If meta_filter_func is not an instance of MetadataFilterComposer, raise TypeError.
            - If self.dataset_cfg.dataset_class is a type of EegClassifyBaseDataset and no ClassifyMetadataFilter exists in the composer, add a filter using the classify_filter factory function.
            - If self.dataset_cfg.dataset_class is a type of EegRegressionBaseDataset and no RegressionMetadataFilter exists in the composer, add a filter using the regression_filter factory function.
        """
        if meta_filter_func is None:
            meta_filter_func = MetadataFilterComposer()
        elif isinstance(meta_filter_func, Sequence):
            if all(isinstance(f, MetadataFilter) for f in meta_filter_func):
                meta_filter_func = MetadataFilterComposer(*meta_filter_func)
            else:
                raise TypeError(
                    "meta_filter_func must be a sequence of MetadataFilter instances."
                )
        elif isinstance(meta_filter_func, MetadataFilter):
            meta_filter_func = MetadataFilterComposer(meta_filter_func)

        assert isinstance(meta_filter_func, MetadataFilterComposer)

        if add_meta_filter_func is not None:
            if isinstance(add_meta_filter_func, Sequence):
                meta_filter_func.add_filters(*add_meta_filter_func)
            elif isinstance(add_meta_filter_func, MetadataFilter):
                meta_filter_func.add_filters(add_meta_filter_func)

        # Check for classify dataset and add classify filter if missing
        if issubclass(dataset_class, EegClassifyBaseDataset):
            has_classify_filter = any(
                isinstance(f, ClassifyMetadataFilter) for f in meta_filter_func.filters
            )
            if not has_classify_filter:
                for arg in args:
                    classify_filter = get_classify_filter(num_class=arg)
                    meta_filter_func.add_filters(classify_filter)

        # Check for regression dataset and add regression filter if missing
        elif issubclass(dataset_class, EegRegressionBaseDataset):
            has_regression_filter = any(
                isinstance(f, RegressionMetadataFilter)
                for f in meta_filter_func.filters
            )
            if not has_regression_filter:
                for arg in args:
                    regression_filter = get_regression_filter(speech_feature=arg)
                    meta_filter_func.add_filters(regression_filter)

        return meta_filter_func

    def load_metadata(
        self,
        dataset_class: type[EegDataset],
        metafile_path: str,
        metadata_fields: Sequence[str],
    ) -> Metadata:
        """
        从 meta.mat 文件中加载元信息。

        Args:
            metafile_path (str): 元信息文件路径。

        Returns:
            dict[entry->metadata_cls]: 转换后的元信息字典。
        """

        with open(metafile_path, "rb") as f:
            metadata: dict[DatasetSubjectTrialEntry, dict[str, Any]] = pickle.load(f)

        # convert to Metadata object.
        meta_dict = {}
        for entry, data in metadata.items():
            if set(metadata_fields).issubset(set(data.keys())):
                meta_dict[entry] = MetadataElement(**data)
        return meta_dict

    def filt_metadata(
        self,
        metadata: Metadata,
        meta_filter_func: (
            Callable[
                [
                    Metadata,
                ],
                Metadata,
            ]
            | None
        ),
    ):
        # if meta_filter_func:
        #     filtered_metadata: Metadata = {}
        #     for dataset_entry, metadata_element in metadata.items():
        #         metadata_element = meta_filter_func(metadata_element)
        #         if metadata_element is not None:
        #             filtered_metadata[dataset_entry] = metadata_element
        #     return filtered_metadata
        # else:
        #     return metadata
        # update in 2026/06/03: support batch filtering. also change the default behaviour of Composer
        if meta_filter_func is not None:
            return meta_filter_func(metadata)
        else:
            return metadata

    def create_datasets(self):
        metadata = self.load_metadata(
            dataset_class=self.dataset_cfg.dataset_class,
            metafile_path=self.dataset_cfg.meta_path,
            metadata_fields=self.dataset_cfg.metadata_fields,
        )

        metadata = self.filt_metadata(metadata, self.dataset_cfg.meta_filter_func)

        splits: Mapping[str, Sequence] = self.dataset_cfg.meta_group_func(
            metadata=metadata,
            val_fold_idx=self.dataset_cfg.val_fold_idx,
            test_fold_idx=self.dataset_cfg.test_fold_idx,
            n_folds=self.dataset_cfg.n_folds,
        )

        dataset_modes = ["train", "val", "test"]

        self.trainset, self.valset, self.testset = (
            self.dataset_cfg.dataset_class(
                eeg_path=self.dataset_cfg.eeg_path,
                files=sorted(list(splits[mode])),
                metadata=metadata,
                fs=self.fs,
                window_length=self.dataset_cfg.window_length,
                overlap=self.dataset_cfg.overlap,
                transform=self.dataset_cfg.transform,
                metadata_fields=self.dataset_cfg.metadata_fields,
                accept_range=splits.get(f"{mode}_accept_range", None),
                reject_range=splits.get(f"{mode}_reject_range", None),
                stage=mode,
                **self.dataset_args if self.dataset_args else {},
            )
            for mode in dataset_modes
        )

        if self.trainset.transform is not None:
            self.valset.sync_transform_stats(self.trainset.transform)
            self.testset.sync_transform_stats(self.trainset.transform)

    def create_dataloader(self, dataset, *args, **kwargs):
        new_kwargs = {}
        for k, v in kwargs.items():
            if k not in self.dataloader_args:
                new_kwargs[k] = v
        return DataLoader(
            dataset,
            *args,
            **new_kwargs,
            **self.dataloader_args,
            collate_fn=collect_multidataset,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.trainset, shuffle=True)

    def val_dataloader(self):
        return self.create_dataloader(self.valset)

    def test_dataloader(self):
        return self.create_dataloader(self.testset)

    def summary(self, verbose: bool = False):
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
                    if meta.entry in dataset.files
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
        if verbose:
            console.print(stats_table, unique_table)

        # class-wise sample count for classification dataset
        if issubclass(self.dataset_cfg.dataset_class, EegClassifyBaseDataset):
            class_count_table = Table(title="Class-wise Sample Count")
            class_count_table.add_column("Class", justify="center", style="cyan")
            class_count_table.add_column("Samples", justify="center", style="magenta")
            class_count_table.add_column("Percentage", justify="center", style="green")

            global_class_counts = {}

            for name, dataset in datasets.items():
                class_counts = {}
                for file_idx, sample_counts in enumerate(
                    dataset._per_file_sample_count
                ):
                    metadata_element: ClassifyMetadataElement = dataset.metadata[
                        dataset.files[file_idx]
                    ]
                    label = metadata_element.label
                    if label not in class_counts:
                        class_counts[label] = 0
                    class_counts[label] += sample_counts
                    if label not in global_class_counts:
                        global_class_counts[label] = 0
                    global_class_counts[label] += sample_counts

                total_samples = sum(class_counts.values())
                for label, count in class_counts.items():
                    percentage = (count / total_samples) * 100
                    class_count_table.add_row(
                        str(f"{name}-{label}"), str(count), f"{percentage:.2f}%"
                    )

            if verbose:
                console.print(class_count_table)
            self._global_class_counts = global_class_counts

    @property
    def batch_size(self):
        return self.dataloader_args["batch_size"]

    @batch_size.setter
    def batch_size(self, value):
        assert isinstance(value, int), "Batch size must be an integer."
        self.dataloader_args["batch_size"] = value

    @property
    def fs(self):
        if self.dataset_cfg.transform is not None:
            composer = self.dataset_cfg.transform
            assert isinstance(
                composer, TransformComposer
            ), "Transform must be a TransformComposer instance."
            for transform in composer.transforms:
                if isinstance(transform, Resample):
                    return transform.new_fs
        return self.dataset_cfg.fs

    @property
    def window_length(self):
        return self.dataset_cfg.window_length

    @property
    def datasets(self):
        return self.trainset, self.valset, self.testset

    @property
    def sample_weights(self):
        if issubclass(self.dataset_cfg.dataset_class, EegClassifyBaseDataset):
            samples_per_class = self._global_class_counts.values()
            weights: np.ndarray = np.array(
                [1 / count for count in samples_per_class], dtype=np.float32
            )
            weights *= len(samples_per_class) / sum(
                weights
            )  # 再次归一化，使权重平均值 = 1
            weights = weights.tolist()
            return weights
        else:
            log(
                logging.WARNING if self.summary_verbose else logging.DEBUG,
                "Sample weights are only available for classification datasets. Skip operation and return None. This is basically because someone want to use sample weights for regression datasets, which is not a common practice. Skip this message if you know what you are doing.",
            )

    @property
    def num_channels(self):
        if self.dataset_cfg.transform is not None:
            for transform in self.dataset_cfg.transform._transforms:
                if hasattr(transform, "num_channels"):
                    return getattr(transform, "num_channels")
        raise AttributeError(
            "Transform does not have attribute 'num_channels'. Please ensure you have a ChannelSelection transform in the pipeline."
        )
