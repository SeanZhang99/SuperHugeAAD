import einops
from sympy import Ei
import torch
from collections.abc import Sequence
from typing import final
from torchmetrics.functional import pearson_corrcoef
import lightning as pl2

from .abc import lsBaseABCModel


class lsBaseFinalMixin(lsBaseABCModel, pl2.LightningModule):
    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)
        super(lsBaseABCModel).__init__(**kwargs)
        self.fake_param = torch.nn.Parameter(torch.tensor(1.0))
        self.stage = "train"

    @final
    def get_lag_mtx(self, x: torch.Tensor, lag: Sequence[int]):
        """
        Construct a lagged matrix for the input with given lag.

        Parameters:
        x: torch.Tensor, input tensor. Shape: (batch_size, time_steps, num_features). The lag (or advance) is operated along the time dimension, and lagged signals is inseted into the second dimension. Other dimensions remain unchanged.
        lag: Sequence[int], the lag range. The first element specify where the lag begins, the second element specify where the lag ends. If the lag is negative, the input is advanced.

        Example:
        lag = [-10, 10]. return: (batch_size, 20, time_steps, num_features). -10 means the input is advanced by 10 time steps, 10 means the input is lagged by 10 time steps.
        """
        x_lag = []
        for l in range(lag[0], lag[1]):
            if l >= 0:
                x_lag.append(
                    torch.cat(
                        [
                            torch.zeros((x.shape[0], l, x.shape[-1])).type_as(x),
                            x[:, : x.shape[1] - l, :],
                        ],
                        dim=1,
                    )
                )
            elif l < 0:
                x_lag.append(
                    torch.cat(
                        [
                            x[:, -l:, :],
                            torch.zeros(x.shape[0], -l, x.shape[-1]).type_as(x),
                        ],
                        dim=1,
                    )
                )
        x_lag = torch.stack(
            x_lag,
            dim=1,
        )
        return x_lag

    @final
    def get_stats(self, x_pred: torch.Tensor, y_pred: torch.Tensor):
        stats: dict[str, torch.Tensor] = {}

        y_pred_labels = ["a", *[f"u{index}" for index in range(1, y_pred.shape[-1])]]

        x_pred = einops.rearrange(x_pred, "batch time -> time batch")
        y_pred = einops.rearrange(y_pred, "batch time feature -> time batch feature")

        # label meaning: 'a': attended, 'u+digit': unattended
        for j, label in enumerate(y_pred_labels):
            stats[f"{self.stage}/{label}_pcc"] = pearson_corrcoef(
                x_pred, y_pred[:, :, j]
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
        self.log_dict({k: v.mean() for k, v in stats.items()})

        return stats

    @final
    def get_batch_data(self, batch, batch_idx):
        return batch["exg"], batch["audio"][:, :, 0]

    @final
    def forward(self):
        return self.fake_param * 1

    @final
    def on_train_epoch_start(self):
        self.stage = "train"
        super().on_train_epoch_start()

    @final
    def on_train_epoch_end(self):
        self.fit()

    @final
    def on_validation_epoch_start(self):
        self.stage = "val"
        super().on_validation_epoch_start()

    @final
    def validation_step(self, batch, batch_idx):
        x: torch.Tensor = batch["exg"]
        y: torch.Tensor = batch["audio"]
        x_pred, y_pred = self.predict(x, y)
        stats = self.get_stats(x_pred, y_pred)

    @final
    def on_test_epoch_start(self):
        self.stage = "test"
        super().on_test_epoch_start()

    @final
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    @final
    def configure_optimizers(self):
        return torch.optim.sgd.SGD(self.parameters(), lr=0.0)
