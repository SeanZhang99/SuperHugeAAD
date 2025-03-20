from torch.nn.modules.loss import _Loss
import torch


class MSELoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            y_pred[:, :, 0], y_true[:, :, 0], reduction="none"
        )


class ContrastiveMSELoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(
            y_pred[:, :, 0], y_true[:, :, 0], reduction="none"
        )
        loss -= (
            torch.nn.functional.mse_loss(
                y_pred[:, :, 0], y_true[:, :, 1], reduction="none"
            )
            * 0.5
        )
        loss -= (
            torch.nn.functional.mse_loss(
                y_pred[:, :, 0], y_true[:, :, 2], reduction="none"
            )
            * 0.5
        )
        return loss
