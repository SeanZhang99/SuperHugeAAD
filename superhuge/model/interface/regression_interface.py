from collections.abc import Sequence

import torch
import torchinfo

from ..loss.classify.f1_score import binary_f1_score
from ..loss.regression.pearson_loss import pearson_corrcoef
from ..module.post_model import regression_post_model
from .channel_mapping_interface import (
    ChannelMapping1DInterface,
    ChannelMapping2DInterface,
)
from .linear_interface import LinearInterface
from .model_interface import MInterface


class RegressionInterface(MInterface):

    def __init__(self, /, *, num_audio_features: int, **kwargs):
        self.required_output_keys = ["eeg", "audio"]
        super().__init__(**kwargs)

        self._num_audio_features = num_audio_features

        self.post_model = regression_post_model(self.output_size, num_audio_features)
        torchinfo.summary(
            self.post_model, input_size=self.output_size, verbose=self.summary_verbose
        )

    def log_metric_and_stats(
        self,
        stats: dict,
        metrics: torch.Tensor,
        metrics_name: str,
        speaker_labels: Sequence[str],
        meta: dict,
    ):
        for j, label in enumerate(speaker_labels):
            stats[f"{self.stage}/{label}_{metrics_name}"] = metrics[..., j].mean(dim=1)
            # Compute pcc difference between the first speaker and the rest
            if j >= 1:
                stats[f"{self.stage}/{label}_{metrics_name}_diff"] = (
                    metrics[..., 0] - metrics[..., j]
                ).mean(dim=1)

        metrics_mean = metrics.mean(dim=1)

        if metrics_mean.shape[-1] > 1:
            pred = torch.argmax(metrics_mean, dim=-1)
            stats[f"{self.stage}/acc_by_{metrics_name}"] = (pred == 0).type_as(metrics)
            stats[f"{self.stage}/f1_by_{metrics_name}"] = binary_f1_score(
                pred,
                torch.zeros((1,), device=metrics_mean.device, dtype=torch.long),
                positive_label=0,
            )

        # if self.stage in ["val", "test"] and metrics_mean.shape[-1] > 1:
        #     # log the pcc and acc metric for each dataset
        #     dataset_id = int(meta["dataset_id"][0])
        #     stats[f"{self.stage}/{dataset_id=}_pcc"] = metrics_mean.mean(dim=1)
        #     stats[f"{self.stage}/{dataset_id=}_acc"] = (
        #         torch.argmax(metrics_mean, dim=-1) == 0
        #     ).type_as(metrics)

        return stats

    def get_stats(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        /,
        *,
        meta: dict,
    ) -> dict[str, torch.Tensor]:
        stats: dict[str, torch.Tensor] = {}

        # label meaning: 'a': attended, 'u+digit': unattended
        speaker_labels = ["a", *[f"u{index}" for index in range(1, y_true.shape[-1])]]

        # pcc: (batch, feature, speaker)
        pcc = pearson_corrcoef(y_pred, y_true, dim=1)
        stats = self.log_metric_and_stats(stats, pcc, "pcc", speaker_labels, meta)

        self.log_dict(
            {k: v.mean() for k, v in stats.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
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


class LinearRegressionInterface(LinearInterface, RegressionInterface):  # type: ignore
    pass
