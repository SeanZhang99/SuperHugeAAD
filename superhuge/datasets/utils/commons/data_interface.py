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
    ):
        super().__init__()
        config = DInterfaceConfig(
            dataset_args=dataset_args,
            dataloader_args=dataloader_args,
            dataset_class=dataset_class,
        )
        self.config = config
        self.create_datasets()

    def create_datasets(self):
        class_args = list(
            inspect.signature(
                self.config.dataset_class.create_datasets
            ).parameters.keys()
        )

        for key in self.config.dataset_args.keys():
            if key not in class_args:
                raise ValueError(
                    f"DATA_INTERFACE:D_INTERFACE:CREATE_DATASETS:INVALID_DATASET_ARGS:VALUE_ERROR Argument {key} is not a valid argument for {self.config.dataset_class.create_datasets}. Expected arguments are {class_args}"
                )

        self.trainset, self.valset, self.testset = (
            self.config.dataset_class.create_datasets(**self.config.dataset_args)
        )

    def create_dataloader(self, dataset):
        return DataLoader(dataset, **self.config.dataloader_args)

    def train_dataloader(self):
        return self.create_dataloader(self.trainset)

    def val_dataloader(self):
        return self.create_dataloader(self.valset)

    def test_dataloader(self):
        return self.create_dataloader(self.testset)
