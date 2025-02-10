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

from collections.abc import Sequence
from typing import Any
import torch

import lightning as pl2

from .model_template import ModelTemplate


class MInterface(pl2.LightningModule):
    def __init__(
        self,
        /,
        model_class: type[ModelTemplate],
        model_args: dict[str, Any],
        lr: float,
        loss: torch.nn.modules.loss._Loss | Sequence[torch.nn.modules.loss._Loss],
        loss_hparams: Sequence[float] | None = None,
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.precision = precision
        if isinstance(loss, Sequence):
            assert isinstance(loss_hparams, Sequence) and len(loss) == len(
                loss_hparams
            ), f"When specifying multiple losses, you must also specify the corresponding loss weights, but got {loss} and {loss_hparams}"
        elif isinstance(loss, torch.nn.modules.loss._Loss):
            assert (
                loss_hparams is None
            ), f"When specifying a single loss, you should not specify the loss weights, but got {loss} and {loss_hparams}"
        self.model = model_class.create_models(**model_args).to(self.precision)
        self.loss = loss
        self.loss_hparams = loss_hparams
        self.lr = lr
        self.configure_loss()

    def forward(self, input) -> torch.Tensor:
        input = input.to(self.precision)
        return self.model.forward(input)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        target = torch.randn_like(output)
        return self.loss_fn(output, target)

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def configure_loss(self):
        if isinstance(self.loss, Sequence):

            def loss_fn(*args: Sequence[torch.Tensor | int | str]):  # type: ignore
                loss = 0
                for i, loss_fn in enumerate(self.loss):
                    loss += loss_fn(*args[i])
                return (
                    torch.Tensor(loss) if not isinstance(loss, torch.Tensor) else loss
                )

        else:

            def loss_fn(*args: torch.Tensor | int | str):
                loss = self.loss(*args)
                # type: ignore
                return (
                    torch.Tensor(loss) if not isinstance(loss, torch.Tensor) else loss
                )

        self.loss_fn = loss_fn


class ClassifierInterface(MInterface):
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # How pytorch collect_fn handle nested dict:
        # batch["meta"]["dataset_id"][sample_idx] -> torch.Tensor
        meta: dict[str, Any] = batch["meta"]
        exg: torch.Tensor = batch["exg"]
        label: str = batch["label"]
        outputs = self.forward(exg)
        loss = self.loss_fn(outputs, label)  # type: ignore
        if self.training:
            self.log("train_loss", loss)  # Log as 'train_loss' during training
        else:
            self.log("val_loss", loss)
        # for sample_idx in range(len(batch)):
        #     self.log_dict({f"dataset-{batch["meta"][sample_idx]["dataset_id"]:03d}-subject-{batch["meta"][sample_idx]["subject_id"]:03d}-trial-{}": outputs[sample_idx]})
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        loss = self.training_step(batch, batch_idx)
        # do additional things here
        return loss

    def test_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)


class RegressionInterface(MInterface):
    def training_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        inputs = batch["exg"]
        targets = batch["audio"]
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, targets)  # type: ignore
        if self.training:
            self.log("train_loss", loss)  # Log as 'train_loss' during training
        else:
            self.log("val_loss", loss)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        loss = self.training_step(batch, batch_idx)
        # do additional things here
        return loss

    def test_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.validation_step(batch, batch_idx)
