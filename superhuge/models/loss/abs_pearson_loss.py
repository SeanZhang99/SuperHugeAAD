from .pearson_loss import PearsonLoss
import torch


class AbsPearsonLoss(PearsonLoss):
    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return -torch.abs(super().forward(y_pred=y_pred, y_true=y_true, **kwargs))


class ContrastiveAbsPearsonLoss(AbsPearsonLoss):
    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        loss = super().forward(
            y_pred=y_pred[:, :, 0][:, :, torch.newaxis],
            y_true=y_true[:, :, 0][:, :, torch.newaxis],
            **kwargs,
        )
        loss += (
            super().forward(
                y_pred=y_pred[:, :, 0][:, :, torch.newaxis],
                y_true=y_true[:, :, 1][:, :, torch.newaxis],
                **kwargs,
            )
            * 0.5
        )
        loss += (
            super().forward(
                y_pred=y_pred[:, :, 0][:, :, torch.newaxis],
                y_true=y_true[:, :, 2][:, :, torch.newaxis],
                **kwargs,
            )
            * 0.5
        )
        return loss
