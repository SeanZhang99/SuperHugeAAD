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

from collections.abc import Callable, Sequence
import inspect
from typing import Any, final
import einops
import keras
import torch
from abc import ABC, abstractmethod

import lightning as pl2
from torchmetrics.functional import pearson_corrcoef
from torchmetrics.classification import ConfusionMatrix

from .model_template import ModelTemplate


class MInterface(pl2.LightningModule, ABC):
    def __init__(
        self,
        /,
        model_class: type[ModelTemplate] | Callable[..., ModelTemplate],
        model_args: dict[str, Any],
        lr: float,
        loss: torch.nn.modules.loss._Loss | Sequence[torch.nn.modules.loss._Loss],
        loss_hparams: Sequence[float] | None = None,
        precision: torch.dtype = torch.float32,
        ckpt_path: str | None = None,
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
        if isinstance(model_class, type):
            self.model = model_class.create_models(**model_args).to(self.precision)
        elif isinstance(model_class, Callable):
            self.model = model_class(**model_args).to(self.precision)
        else:
            raise TypeError(
                f"SUPERHUGE:MODELS:MODEL_INTERFACE:__INIT__:{model_class} shoule be a 'torch.nn.Module' with method 'create_models' or a callable returning a model, but got {type(model_class)}"
            )
        if ckpt_path is not None:
            if isinstance(self.model, keras.Model):
                self.model.load_weights(ckpt_path, skip_mismatch=True, by_name=True)
            else:
                self.model.load_state_dict(torch.load(ckpt_path), strict=False)
        self.loss = loss
        self.loss_hparams = loss_hparams
        self.lr = lr
        self.configure_loss()
        self.stage = "train"

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return self.model(*inputs)

    # define training_step, validation_step, test_step in your own subclass
    @abstractmethod
    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def test_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        pass

    @final
    def on_train_epoch_start(self) -> None:
        self.stage = "train"
        return super().on_train_epoch_start()

    @final
    def on_validation_epoch_start(self) -> None:
        self.stage = "val"
        return super().on_validation_epoch_start()

    @final
    def on_test_epoch_start(self) -> None:
        self.stage = "test"
        return super().on_test_epoch_start()

    @final
    def configure_loss(self):
        if isinstance(self.loss, Sequence):
            # add a closure variable to let static type checker know the type of loss_fn
            loss_sequence: Sequence = self.loss

            def loss_fn(*args: Sequence[torch.Tensor | int | str | Sequence[torch.Tensor | int | str]]):  # type: ignore
                loss = 0
                for i, loss_fn in enumerate(loss_sequence):
                    loss += loss_fn(*args[i])
                return (
                    torch.Tensor(loss) if not isinstance(loss, torch.Tensor) else loss
                )

        else:
            single_loss: torch.nn.modules.loss._Loss = self.loss

            def loss_fn(*args: torch.Tensor | int | str):
                loss = single_loss(*args)
                # type: ignore
                return (
                    torch.Tensor(loss) if not isinstance(loss, torch.Tensor) else loss
                )

        self.loss_fn = loss_fn


class ClassifierInterface(MInterface):
    def __init__(self, *args, **kwargs):
        # Get the signature of the parent __init__ method
        parent_signature = inspect.signature(super().__init__)

        # Validate the arguments against the parent's signature
        bound_arguments = parent_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()  # Ensure default values are included

        # Forward the validated arguments to the parent
        super().__init__(*bound_arguments.args, **bound_arguments.kwargs)

        self.confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=bound_arguments.kwargs["model_args"]["num_classes"],
        )

    __init__.__signature__ = inspect.signature(MInterface.__init__)  # type: ignore

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        # How pytorch collect_fn handle nested dict:
        # batch["meta"]["dataset_id"][sample_idx] -> torch.Tensor
        meta: dict[str, Any] = batch["meta"]
        exg: torch.Tensor = batch["exg"]
        label: str = batch["label"]
        outputs = self.forward(exg)
        pred = outputs.argmax(dim=1)
        loss = self.loss_fn(outputs, label)  # type: ignore
        stage = self.stage
        self.log(f"{stage}_loss", loss, prog_bar=True)
        # Log as 'train_loss' during training , 'val_loss' during validation/testing
        for sample_idx in range(len(batch)):
            self.log_dict(
                {
                    # accuracy accumulated and reduced on each trial
                    f"{meta["entry"][sample_idx]}_{stage}_acc": (
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    # accuracy accumulated and reduced on each subject
                    f"dataset-{meta["dataset"]:03d}-subject-{meta["subject"][sample_idx]:03d}_{stage}_acc": (
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    # accuracy accumulated and reduced on each dataset
                    f"dataset-{meta["dataset"]:03d}_{stage}_acc": (
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    # accuracy accumulated and reduced on each class
                    f"{str(label[sample_idx])}_{stage}_acc": (
                        pred[sample_idx] == label[sample_idx]
                    ).float(),
                    # confusion matrix
                }
            )
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

    @abstractmethod
    def get_stats(
        self, x_pred: torch.Tensor, y_pred: torch.Tensor, batch_size: int | None = None
    ):
        pass

    def training_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        # extract input and target, call forward, and calculate loss
        targets: torch.Tensor = batch["audio"]  # type: ignore
        predictions = self.forward(batch["exg"])
        loss = self.loss_fn(targets, predictions).mean()  # type: ignore

        self.get_stats(predictions, targets, batch_size=targets.shape[0])
        self.log(f"{self.stage}/loss", loss, batch_size=targets.shape[0], prog_bar=True)

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


class cEEGridRegressionInterface(RegressionInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_stats(
        self, x_pred: torch.Tensor, y_pred: torch.Tensor, batch_size: int | None = None
    ) -> tuple:
        stats: dict[str, torch.Tensor] = {}

        y_pred_labels = ["a", *[f"u{index}" for index in range(1, y_pred.shape[-1])]]

        x_pred = einops.rearrange(x_pred, "batch time feature -> time batch feature")
        y_pred = einops.rearrange(y_pred, "batch time feature -> time batch feature")

        # label meaning: 'a': attended, 'u+digit': unattended
        for j, label in enumerate(y_pred_labels):
            stats[f"{self.stage}/{label}_pcc"] = pearson_corrcoef(
                x_pred[:, :, 0], y_pred[:, :, j]
            )

        stats[f"{self.stage}/acc"] = (
            torch.argmax(
                torch.stack(
                    [stats[f"{self.stage}/{label}_pcc"] for label in y_pred_labels],
                    dim=1,
                ),
                dim=1,
            )
            == 0
        ).type_as(x_pred)
        self.log_dict(
            {k: v.mean() for k, v in stats.items()},
            batch_size=batch_size,
            prog_bar=True,
        )

        return stats
