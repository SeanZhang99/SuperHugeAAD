import inspect

import einops
import torch
from torchmetrics.functional import pearson_corrcoef

from .m_interface import MInterface
from .post_model import regression_post_model
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
)


class RegressionInterface(MInterface):
    def __init__(self, **kwargs):
        # Get the signature of the parent __init__ method
        parent_signature = inspect.signature(super().__init__)

        # Validate the arguments against the parent's signature
        bound_arguments = parent_signature.bind(**kwargs)
        bound_arguments.apply_defaults()

        # Forward the validated arguments to the parent
        super().__init__(**bound_arguments.kwargs)

        self.post_model = regression_post_model(self.input_size)

    __init__.__signature__ = inspect.signature(MInterface.__init__)  # type: ignore

    def training_closure(self, data: dict) -> torch.Tensor:
        outputs: torch.Tensor = self.forward(data)
        if outputs.ndim == 2:
            outputs = einops.rearrange(outputs, "batch time -> batch time 1")
        target: torch.Tensor = data["audio"]  # type: ignore

        return outputs, target

    def get_stats(
        self, x_pred: torch.Tensor, y_pred: torch.Tensor, meta: dict
    ) -> dict[str, torch.Tensor]:
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
                    dim=-1,
                ),
                dim=-1,
            )
            == 0
        ).type_as(x_pred)

        self.log_dict(
            {k: v.mean() for k, v in stats.items()},
            prog_bar=True,
            # on_epoch=True,
            # on_step=False,
            batch_size=x_pred.shape[1],
            sync_dist=True,
        )

        return stats


class Channel1DRegressionInterface(RegressionInterface, ChannelMapping1DInterface):
    pass


class Channel2DRegressionInterface(RegressionInterface, ChannelMapping2DInterface):
    pass
