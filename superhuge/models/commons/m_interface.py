from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import final
from warnings import warn

import keras
import lightning as pl2
import torch

from superhuge.models.commons import post_model, pre_model


from .model_template import ModelTemplate
from .lambda_layer import LambdaLayer
from superhuge.models.commons import lambda_layer


class MInterface(pl2.LightningModule, ABC):

    def __init__(
        self,
        /,
        *,
        model_class: Callable[..., keras.Model | torch.nn.Module],
        model_args: dict,
        loss: torch.nn.modules.loss._Loss | Sequence[torch.nn.modules.loss._Loss],
        loss_hparams: Sequence[float] | None = None,
        ckpt_path: str | None = None,
        summary: bool = True,
    ):
        super().__init__()
        if isinstance(loss, Sequence):
            assert loss_hparams is None or (
                isinstance(loss_hparams, Sequence) and len(loss) == len(loss_hparams)
            ), f"When specifying multiple losses, you must also specify the corresponding loss weights to be a sequence or None, but got {loss} and {loss_hparams}"
        elif isinstance(loss, torch.nn.modules.loss._Loss):
            assert (
                loss_hparams is None
            ), f"When specifying a single loss, you should not specify the loss weights, but got {loss} and {loss_hparams}"
        assert callable(
            model_class
        ), f"model_class should be a callable, but got {model_class}"
        self.model = model_class(**model_args)

        if ckpt_path is not None:
            if isinstance(self.model, keras.Model):
                self.model.load_weights(ckpt_path, skip_mismatch=True, by_name=True)
            elif isinstance(self.model, torch.nn.Module):
                self.model.load_state_dict(torch.load(ckpt_path), strict=False)
        self.loss = loss
        self.loss_hparams = loss_hparams
        self.configure_loss()
        self.stage = "train"

        self.get_input_size(**model_args)

        if summary:
            if hasattr(self.model, "summary"):
                self.model.summary()
            # else:
            #     torchinfo.summary(self.model, input_size=self.input_size)

        self.pre_model: torch.nn.Module = lambda_layer.LambdaLayer(lambda x: x["exg"])
        self.post_model: torch.nn.Module = torch.nn.Identity()

    @final
    def get_input_size(self, /, **kwargs) -> tuple[int | None, int]:
        if "input_length" in kwargs:
            input_length = kwargs["input_length"]
        elif "window_length" in kwargs and "fs" in kwargs:
            input_length = kwargs["window_length"] * kwargs["fs"]
        else:
            warn(
                f"SUPERHUGE:MODELS:MODEL_INTERFACE:__INIT__: Cannot interfere the input length from {kwargs}, Using None as input_length"
            )
            input_length = None
        if "num_channel" in kwargs:
            num_channel = kwargs["num_channel"]
        elif "num_electrodes" in kwargs:
            num_channel = kwargs["num_electrodes"]
        elif "input_channels" in kwargs:
            num_channel = kwargs["input_channels"]
        else:
            raise ValueError(
                f"SUPERHUGE:MODELS:MODEL_INTERFACE:__INIT__: Cannot interfere the number of channels from {kwargs}"
            )
        self.input_size = (input_length, num_channel)

        return self.input_size

    def forward(self, data) -> torch.Tensor:
        pre_inputs = self.pre_model(data)
        outputs = self.model(pre_inputs)
        post_outputs = self.post_model(outputs)
        return post_outputs

    @abstractmethod
    def training_closure(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """training_closure. This method will be called during `training_step`. It should return the prediction (the output of the model) and the target (`label` in classification or `target` in regression).

        Args:
            data (dict[str, torch.Tensor]): input data dict. should be a dict with key `exg`, `meta` and `label` or `target`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: [output, target/label]

        Example:
            ```python
            def training_closure(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            # Regression task
                outputs = self.forward(data)
                target = data['audio']
                return outputs, target

            def training_closure(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            # Classification task
                outputs = self.forward(data).argmax[1]
                target = data['label']
                return outputs, target
        """
        pass

    @abstractmethod
    def get_stats(
        self, outputs: torch.Tensor, targets: torch.Tensor, meta: dict
    ) -> None:
        """get_stats. This method will be called during `training_step`, `validation_step` and `test_step`. It should calculate the statistics of the model's output and the target.

        Args:
            output (torch.Tensor): the output of the model
            target (torch.Tensor): the target/label

        Returns:
            None

        Example:
            ```python
            def get_stats(self, output: torch.Tensor, target: torch.Tensor) -> None:
                self.log('accuracy', (output == target).float().mean())
            ```
        """
        pass

    @final
    def training_step(self, batch: dict[str], batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = torch.zeros(1, device=self.device)
        batch_size = 0
        for data in batch.values():
            outputs, targets = self.training_closure(data)
            loss += self.loss_fn(outputs, targets).sum()
            batch_size += outputs.shape[0]
            self.get_stats(
                outputs,
                targets,
                data["meta"],
            )

        loss /= batch_size
        self.log(
            f"{self.stage}/loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    @final
    def validation_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

    @final
    def test_step(
        self, batch: dict[str, torch.Tensor | dict], batch_idx: int
    ) -> torch.Tensor:
        return self.training_step(batch, batch_idx)

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
        """Configure the loss function. If multiple losses are provided, the loss function will be a sum of all the losses. If a single loss is provided, the loss function will be the loss itself. The loss function will be stored in `self.loss_fn`. When calling `self.loss_fn`, always put the output of the model first and the target/label as the second.

        Returns:
            None, the configured loss will be accessible through `self.loss_fn`
        """
        if isinstance(self.loss, Sequence):
            # add a closure variable to let static type checker know the type of loss_fn
            if self.loss_hparams is None:
                self.loss_hparams = [1] * len(self.loss)

            self.loss_hparams = torch.tensor(self.loss_hparams, device=self.device)

            def loss_fn(
                *args: torch.Tensor | int | str | Sequence[torch.Tensor | int | str],
            ):
                loss = torch.zeros(1, device=self.device)
                for loss_fn, weight in zip(self.loss, self.loss_hparams):
                    loss += loss_fn(*args) * weight
                return loss

        else:

            def loss_fn(*args: torch.Tensor | int | str):
                return self.loss(*args)

            loss_fn.__repr__ = f"{self.loss.__repr__().split('(')[0]}"

        self.loss_fn = loss_fn
