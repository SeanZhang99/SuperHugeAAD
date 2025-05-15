import torch
import torchinfo

from ..module.post_model import regression_post_model
from .model_interface import MInterface
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
)
from ..loss.regression.pearson_loss import pearson_corrcoef
from ..loss.classify.f1_score import binary_f1_score


class RegressionInterface(MInterface):

    def __init__(self, /, *, num_audio_features: int, **kwargs):
        self.required_output_keys = ["eeg", "audio"]
        super().__init__(**kwargs)

        self._num_audio_features = num_audio_features

        self.post_model = regression_post_model(self.output_size, num_audio_features)
        torchinfo.summary(self.post_model, input_size=self.output_size)

    def get_stats(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        meta: dict,
    ) -> dict[str, torch.Tensor]:
        stats: dict[str, torch.Tensor] = {}

        # label meaning: 'a': attended, 'u+digit': unattended
        speaker_labels = ["a", *[f"u{index}" for index in range(1, y_true.shape[-1])]]

        # pcc: (batch, feature, speaker)
        pcc = pearson_corrcoef(y_pred, y_true, dim=1)
        for j, label in enumerate(speaker_labels):
            stats[f"{self.stage}/{label}_pcc"] = pcc[..., j].mean(dim=1)
            # Compute per-band stats only if more than one band is present
            if pcc.shape[1] > 1:
                for f in range(y_true.shape[1]):
                    stats[f"{self.stage}/{label}_pcc_band_{f}"] = pcc[..., f, j]
            if j >= 1:
                stats[f"{self.stage}/{label}_pcc_diff"] = (
                    pcc[..., 0] - pcc[..., j]
                ).mean(dim=1)

        pcc_mean = pcc.mean(dim=1)

        if pcc_mean.shape[-1] > 1:
            stats[f"{self.stage}/acc"] = (torch.argmax(pcc_mean, dim=-1) == 0).type_as(
                y_pred
            )
            stats[f"{self.stage}/f1"] = binary_f1_score(
                torch.argmax(pcc_mean, dim=-1),
                0,
                positive_label=0,
            )

        self.log_dict(
            {k: v.mean() for k, v in stats.items()},
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=y_pred.shape[0],
            sync_dist=True,
        )

        return stats


class ChannelMapping1DRegressionInterface(
    RegressionInterface, ChannelMapping1DInterface
):
    pass


class ChannelMapping2DRegressionInterface(
    RegressionInterface, ChannelMapping2DInterface
):
    pass
