from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pydantic
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
from ..module.model_template import NumAudioFeaturesMixin


class RegressionInterface(MInterface):

    def __init__(
        self,
        /,
        *,
        num_audio_features: NumAudioFeaturesMixin,  # type: ignore
        **kwargs,
    ):
        num_audio_features: dict = num_audio_features.model_dump()
        self.required_output_keys = ["eeg"]
        self.required_output_keys.extend(
            x for x in num_audio_features.keys() if num_audio_features[x] is not None
        )
        super().__init__(**kwargs)

        assert isinstance(num_audio_features, Mapping)

        self._num_audio_features = num_audio_features

        self.post_model = regression_post_model(
            self.output_size,
            sum(x for x in num_audio_features.values() if x is not None),
        )
        torchinfo.summary(
            self.post_model,
            input_size=self.output_size,
            verbose=self.summary_verbose,
            col_names=[
                "input_size",
                "output_size",
                "num_params",
                "mult_adds",
                "trainable",
            ],
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


class LinearChannelMapping1DRegressionInterface(
    LinearInterface, ChannelMapping1DRegressionInterface
):
    pass


class RegressionInterfaceWithEnvDump(RegressionInterface):
    def on_test_start(self) -> None:
        super().on_test_start()
        base_dir = (
            Path(self.logger.log_dir)  # type: ignore
            if getattr(self, "logger", None) and getattr(self.logger, "log_dir", None)
            else Path(".")
        )
        self._test_env_dir = base_dir / "test_env"
        self._test_env_dir.mkdir(parents=True, exist_ok=True)
        self._test_env_batch_idx = 0

    def _save_test_env_batch(
        self, y_pred: torch.Tensor, y_true: torch.Tensor | None, meta: dict
    ) -> None:
        if not hasattr(self, "_test_env_dir"):
            return

        # ---- DDP safety: add rank to filename ----
        rank = (
            getattr(self.trainer, "global_rank", 0) if hasattr(self, "trainer") else 0
        )

        payload = {"y_pred": y_pred.detach().cpu()}
        if y_true is not None:
            payload["y_true"] = y_true.detach().cpu()

        # dump a few meta fields (add more if you have them)
        for key in ["trial_id", "dataset_id", "subject_id"]:
            if key in meta:
                v = meta[key]
                if torch.is_tensor(v):
                    v = v.detach().cpu().view(-1)
                else:
                    v = torch.as_tensor(v).view(-1)
                payload[key] = v

        # optional: record what meta keys exist (helps debugging)
        payload["meta_keys"] = list(meta.keys())

        out_path = (
            self._test_env_dir
            / f"rank{rank:02d}_batch_{self._test_env_batch_idx:06d}.pt"
        )
        torch.save(payload, out_path)
        self._test_env_batch_idx += 1

    def get_stats(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, /, *, meta: dict
    ) -> dict[str, torch.Tensor]:
        if getattr(self, "stage", None) == "test":
            self._save_test_env_batch(y_pred, y_true, meta)
        return super().get_stats(y_pred, y_true, meta=meta)


class LinearRegressionInterfaceWithEnvDump(
    LinearInterface, RegressionInterfaceWithEnvDump
):
    pass
