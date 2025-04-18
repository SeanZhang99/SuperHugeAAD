import inspect

import einops
import torch
from torchmetrics.functional import pearson_corrcoef

from .model_interface import MInterface
from ..tools.post_model import regression_post_model
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
)


class RegressionInterface(MInterface):

    def __init__(self, /, *, num_audio_features: int, **kwargs):
        self.required_output_keys = ["eeg", "audio"]
        super().__init__(**kwargs)

        self.post_model = regression_post_model(self.output_size, num_audio_features)

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
