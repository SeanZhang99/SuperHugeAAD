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

import inspect
from typing import Any

import lightning as pl2
from pydantic import BaseModel
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table

from superhuge.datasets.commons import collate_fn

from .eeg_dataset import EegDataset


class DInterfaceConfig(BaseModel):
    dataset_args: dict[str, Any]
    dataloader_args: dict[str, Any]
    dataset_class: type[EegDataset]


class DInterface(pl2.LightningDataModule):

    def __init__(
        self,
        /,
        dataset_class: type[EegDataset],
        dataset_args: dict,
        dataloader_args: dict,
        summary: bool = True,
    ):
        super().__init__()

        if not dataloader_args.get("num_workers", None):
            del dataloader_args["prefetch_factor"]
            del dataloader_args["persistent_workers"]

        config = DInterfaceConfig(
            dataset_args=dataset_args,
            dataloader_args=dataloader_args,
            dataset_class=dataset_class,
        )
        self.config = config
        self.create_datasets()

        if summary:
            self.print_summary()

    def create_datasets(self):
        required_args = [
            p.name
            for p in inspect.signature(
                self.config.dataset_class.create_datasets
            ).parameters.values()
            if p.default == inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

        for arg in required_args:
            assert (
                arg in required_args
            ), f"DATA_INTERFACE:D_INTERFACE:CREATE_DATASETS:MISSING_REQUIRED_ARG:VALUE_ERROR Missing required argument {arg} for {self.config.dataset_class.create_datasets}. Required arguments are {required_args}"

        self.trainset, self.valset, self.testset = (
            self.config.dataset_class.create_datasets(**self.config.dataset_args)
        )

    def create_dataloader(self, dataset, *args, **kwargs):
        return DataLoader(
            dataset,
            *args,
            **kwargs,
            **self.config.dataloader_args,
            collate_fn=collate_fn.custom_collate_fn,
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
        stats_table.add_column(
            "# Dataset", justify="center", style="cyan", no_wrap=True
        )
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
                    f"{entry.dataset_id}-{entry.subject_id}"
                    for entry in dataset.metadata.values()
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

        console.print(stats_table)

        # Table for unique dataset names and IDs
        unique_table = Table(title="Unique Dataset Names")
        unique_table.add_column("Dataset Name", justify="center", style="cyan")

        for entry in sorted(unique_datasets):
            unique_table.add_row(entry)

        console.print(unique_table)
