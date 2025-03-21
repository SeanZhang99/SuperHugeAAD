import einops
from torchmetrics.functional import pearson_corrcoef
from torch.nn.modules.loss import _Loss
import torch


class PearsonLoss(_Loss):
    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        y_pred = einops.rearrange(y_pred, "b t f-> t b f")
        y_true = einops.rearrange(y_true, "b t f-> t b f")
        return -pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 0])


class ContrastivePearsonLoss(_Loss):
    def forward(
        self, /, *, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        y_pred = einops.rearrange(y_pred, "b t f-> t b f")
        y_true = einops.rearrange(y_true, "b t f-> t b f")
        loss = -pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 0])
        loss += pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 1]) * 0.5
        loss += pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 2]) * 0.5
        return loss
