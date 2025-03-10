from einops import rearrange
from torchmetrics.functional import pearson_corrcoef
from torch.nn.modules.loss import _Loss
from torch import Tensor


class ContrastivePearsonLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        y_pred = rearrange(y_pred, "b t f-> t b f")
        y_true = rearrange(y_true, "b t f-> t b f")
        loss = -pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 0])
        loss += pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 1]) * 0.5
        loss += pearson_corrcoef(y_pred[:, :, 0], y_true[:, :, 1]) * 0.5
        return loss
