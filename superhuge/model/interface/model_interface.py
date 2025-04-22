import inspect
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any, final

import keras
import lightning as pl2
import torch
import torchinfo

from ..module.lambda_layer import LambdaLayer
from ..module.model_template import ModelInputArgs


class MInterface(pl2.LightningModule, ABC):
    required_output_keys: list[str]

    def __init__(
        self,
        /,
        *,
        # Must declare using ModelTemplate to get the linked arguments. e.g., fs and window_length in our case. Otherwise, jsonargparse will ignore this argument. If you want other arguments to be linked, please declare them in the ModelTemplate class.
        # module: ModelTemplate,
        model_class: type[torch.nn.Module] | Callable[..., torch.nn.Module],
        model_args: dict[str, Any],
        model_common_args: ModelInputArgs,
        loss: torch.nn.modules.loss._Loss | Sequence[torch.nn.modules.loss._Loss],
        loss_hparams: Sequence[float] | None = None,
        ckpt_path: str | None = None,
    ):
        super().__init__()

        # Check and configure loss
        if isinstance(loss, Sequence):
            assert loss_hparams is None or (
                isinstance(loss_hparams, Sequence) and len(loss) == len(loss_hparams)
            ), f"When specifying multiple losses, you must also specify the corresponding loss weights to be a sequence or None, but got {loss} and {loss_hparams}"
        elif isinstance(loss, torch.nn.modules.loss._Loss):
            assert (
                loss_hparams is None
            ), f"When specifying a single loss, you should not specify the loss weights, but got {loss} and {loss_hparams}"
        self.loss = loss
        self.loss_hparams = loss_hparams
        self.configure_loss()

        # Instantiate main model and possibly load checkpoint
        if model_common_args.num_channels is None:
            from ...utils.channel_enum import NUM_ELECTRODES as num_channels

            model_common_args.num_channels = num_channels
        self.model = model_class(**model_args, **model_common_args.model_dump())

        if ckpt_path is not None:
            if isinstance(self.model, keras.Model):
                self.model.load_weights(ckpt_path, skip_mismatch=True, by_name=True)
            elif isinstance(self.model, torch.nn.Module):
                self.model.load_state_dict(torch.load(ckpt_path), strict=False)

        # Instantiate pre_model and post_model
        self.stage = "train"
        self.pre_model: torch.nn.Module = LambdaLayer(lambda x: x["eeg"])
        self.post_model: torch.nn.Module = torch.nn.Identity()

        # Configure the input/output of the main model.
        self._required_inputs = self.configure_input()
        self.get_input_size(**model_common_args.model_dump())
        self._output_keys = self.configure_output()

        summary: torchinfo.ModelStatistics = torchinfo.summary(
            self.model, input_size=list(self.input_size.values())
        )
        self.output_size = summary.summary_list[0].output_size

    @final
    def get_input_size(self, /, **kwargs) -> list[tuple[int | None, ...]]:
        """
        Get the input size for each required input type based on the model's forward method.

        Args:
            kwargs: Additional arguments to determine input dimensions.

        Returns:
            list[tuple[int | None, ...]]: A list of input sizes in the order specified by self._required_inputs.
        """
        # Explicitly ensure dictionary insert order
        input_sizes = OrderedDict()

        # Determine EEG/EXG input size
        input_length = kwargs["fs"] * kwargs["window_length"]

        num_channels = kwargs["num_channels"]

        for required_input in self._required_inputs:
            # Add EEG/EXG input size if required
            if required_input == "eeg":
                input_sizes["eeg"] = (1, input_length, num_channels)

            # Add audio input size if required
            if "audio" == required_input:
                input_sizes["audio"] = (1, input_length, kwargs["num_audio_features"])

            # Add label input size if required
            if "label" == required_input:
                input_sizes["label"] = (1, 1)

        self.input_size = input_sizes

    def forward(self, data) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.

        Args:
            data (dict): A dictionary containing the input data. Keys may include 'eeg', 'audio', and 'label'.

        Returns:
            tuple: The output of the model after processing through pre_model, model, and post_model,
                along with any additional required outputs from data.
        """
        # Prepare inputs based on the required order from configure_input
        inputs = []
        for input_type in self._required_inputs:
            if input_type == "eeg":
                # Process EEG/EXG input through pre_model
                inputs.append(self.pre_model(data))
            elif input_type == "audio":
                # Extract audio input directly from data
                inputs.append(data["audio"])
            elif input_type == "label":
                # Extract label input directly from data
                inputs.append(data["label"])

        # Pass the inputs to the model in the required order
        model_outputs = self.model(*inputs)

        # Ensure model_outputs is a tuple
        if not isinstance(model_outputs, tuple):
            model_outputs = (model_outputs,)

        # Process the first output (eeg_hat) through post_model
        eeg_output = self.post_model(model_outputs[0])

        # Collect all outputs
        outputs = [eeg_output]

        # Add additional outputs from the model if they exist
        if len(self._output_keys) > 1:
            outputs.extend(model_outputs[1:])

        # Add required outputs from data if not already in model outputs
        for key in self.required_output_keys:
            if key not in self._output_keys:
                outputs.append(data[key])

        return tuple(outputs)

    @abstractmethod
    def get_stats(
        self, outputs: torch.Tensor, targets: torch.Tensor, /, *, meta: dict
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

    def training_closure(self, *args):
        """If you want to do any arbitary modification to the data before or after the calling of `self.forward`, override this method. This method will be called during `training_step`, and therefore, also `validation_step` and `test_step` in our logics. The default implementation is to call `self.forward` directly."""

        # Call the forward method of the model with the provided arguments
        return self.forward(*args)

    @final
    def training_step(self, batch: dict[str], batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = torch.zeros(1, device=self.device)
        batch_size = 0
        for data in batch.values():
            outputs = self.training_closure(data)
            loss += self.loss_fn(*outputs).sum()
            batch_size += outputs[0].shape[0]
            self.get_stats(
                *outputs,
                meta=data["meta"],
            )

        loss /= batch_size
        self.log(
            f"{self.stage}/loss",
            loss,
            batch_size=batch_size,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            enable_graph=False,
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

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.detach().data.norm(2).item()
                self.log(
                    f"grad_norm/{name}",
                    grad_norm,
                    on_epoch=False,
                    batch_size=1,
                    enable_graph=False,
                )
        return super().on_after_backward()

    @final
    def configure_input(self) -> list[str]:
        """
        Configure the input requirements for the model based on the forward method's input signature.

        Returns:
            list[str]: A sequence of strings indicating the required inputs in the order
                       specified by the model's forward method signature.
                       Possible values: 'eeg', 'audio', 'label'.
        """

        required_inputs = ["eeg"]

        # Inspect the forward method of the model
        if hasattr(self.model, "forward"):
            forward_signature = inspect.signature(self.model.forward)
            forward_params = forward_signature.parameters

            # Check for required inputs based on parameter names
            for param_name in forward_params:
                if param_name in ["env", "mel", "audio"]:
                    required_inputs.append("audio")
                elif param_name == "label":
                    required_inputs.append("label")

        return required_inputs

    @final
    def configure_output(self) -> list[str]:
        """
        Configure the output requirements for the model based on the forward method's output signature.

        Returns:
            list[str]: A sequence of strings indicating the outputs of the model.
                    Possible values: 'eeg_hat', 'audio_hat', 'label_hat'.
        """

        output_keys = ["eeg"]  # EEG output is always present by default

        # Inspect the forward method of the model
        if hasattr(self.model, "forward"):
            forward_signature = inspect.signature(self.model.forward)
            return_annotation = forward_signature.return_annotation

            # Check if the return annotation is a tuple or a single value
            if hasattr(return_annotation, "__args__") and isinstance(
                return_annotation.__args__, tuple
            ):
                # If the return type is a tuple, inspect its elements
                for output_type in return_annotation.__args__[1:]:
                    if "audio" in str(output_type).lower():
                        output_keys.append("audio")
                    elif "label" in str(output_type).lower():
                        output_keys.append("label")

        return output_keys
