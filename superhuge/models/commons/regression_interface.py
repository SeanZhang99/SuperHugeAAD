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

        # Batch * 1
        n_speaker = torch.any(y_pred != 0, dim=1).sum(dim=-1)

        y_pred_labels = ["a", *[f"u{index}" for index in range(1, 5)]]

        x_pred = einops.rearrange(x_pred, "batch time feature -> time batch feature")
        y_pred = einops.rearrange(y_pred, "batch time feature -> time batch feature")

        # label meaning: 'a': attended, 'u+digit': unattended
        for j, label in enumerate(y_pred_labels):
            mask = n_speaker > j
            if torch.any(mask):
                stats[f"{self.stage}/{label}_pcc"] = torch.ones(
                    x_pred.shape[1], device=x_pred.device, dtype=x_pred.dtype
                ) * -float("inf")
                stats[f"{self.stage}/{label}_pcc"][mask] = pearson_corrcoef(
                    x_pred[:, mask, 0], y_pred[:, mask, j]
                )
            else:
                break

        stats[f"{self.stage}/acc"] = (
            torch.argmax(
                torch.stack(
                    [stats[f"{self.stage}/{y_pred_labels[jj]}_pcc"] for jj in range(j)],
                    dim=-1,
                ),
                dim=-1,
            )
            == 0
        ).type_as(x_pred)
        stats[f"{self.stage}/acc"] = stats[f"{self.stage}/acc"][n_speaker > 0]

        for k, v in stats.items():
            assert not torch.isnan(v.mean()), f"{k} is nan"
            self.log(
                k,
                v.mean(),
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=v.shape[0],
            )

        return stats
