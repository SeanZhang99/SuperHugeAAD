from .pearson_loss import PearsonLoss, ContrastivePearsonLoss
from torch.nn.modules.loss import _Loss
import torch


class TwoStagePearsonLoss(_Loss):
    def __init__(self, apply_contrastive_after: int = 10):
        super().__init__()
        self.apply_contrastive_after = apply_contrastive_after
        self.pearson_loss = PearsonLoss()
        self.contrastive_pearson_loss = ContrastivePearsonLoss()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        current_epoch: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if current_epoch < self.apply_contrastive_after:
            return self.pearson_loss(y_pred=y_pred, y_true=y_true)
        else:
            return (
                self.contrastive_pearson_loss(y_pred=y_pred, y_true=y_true) * 0.1
                + self.pearson_loss(y_pred=y_pred, y_true=y_true) * 0.9
            )
