from torch.nn.modules.loss import _Loss
import torch


class MSELoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(
            y_pred, y_true[..., 0], reduction="none"
        ).mean(dim=tuple(range(1, y_pred.ndim)))


class ContrastiveMSELoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        if y_pred.ndim == 3 and y_true.ndim == 4:
            y_pred = y_pred.unsqueeze(-1)
        elif y_pred.ndim == 4 and y_true.ndim == 3:
            y_true = y_true.unsqueeze(-1)
        y_pred = torch.broadcast_to(y_pred, y_true.shape)
        mse: torch.Tensor = torch.nn.functional.mse_loss(
            y_pred, y_true, reduction="none"
        ).mean(dim=tuple(range(1, 3)))
        # average across time and feature dimensions
        loss = mse[..., 0]
        for j in range(1, mse.shape[-1]):
            loss -= mse[..., j] / (mse.shape[-1] - 1)
        return loss
