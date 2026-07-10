from torch.nn.modules.loss import _Loss
import torch


def pearson_corrcoef(
    y_pred: torch.Tensor, y_true: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    pearson_corrcoef Compute the pearson correlation coefficient between y_pred and y_true. y_pred and y_true can be >= 3D tensors. The correlation coefficient is computed along the specified dimension. The other dimensions are kept.

    :param y_pred: _description_
    :type y_pred: torch.Tensor
    :param y_true: _description_
    :type y_true: torch.Tensor
    :param dim: which dimension to compute the correlation coefficient over. (usually the time dimension)
    :type dim: int
    :return: _description_
    :rtype: torch.Tensor
    """

    while y_pred.ndim > y_true.ndim:
        y_true = y_true.unsqueeze(dim=y_true.ndim)
    while y_true.ndim > y_pred.ndim:
        y_pred = y_pred.unsqueeze(dim=y_pred.ndim)

    # Compute mean of y_true and y_pred along the specified dimension.
    y_true_mean = torch.mean(y_true, dim=dim, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=dim, keepdim=True)

    # Compute the numerator and denominator of the pearson correlation
    numerator = torch.sum(
        (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=dim, keepdim=True
    )
    var_true = torch.sum((y_true - y_true_mean) ** 2, dim=dim, keepdims=True)  # type: ignore
    var_pred = torch.sum((y_pred - y_pred_mean) ** 2, dim=dim, keepdims=True)  # type: ignore
    denominator = torch.sqrt(var_true * var_pred)

    # Compute the pearson correlation
    return divide_no_nan(numerator, denominator).squeeze(dim=dim)


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    safe_result = torch.zeros_like(a)
    mask = b != 0
    safe_result[mask] = a[mask] / b[mask]
    return safe_result


class PearsonLoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return -pearson_corrcoef(y_pred, y_true[..., 0], dim=1).mean(dim=1)


class ContrastivePearsonLoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        pcc = pearson_corrcoef(y_pred, y_true, dim=1).mean(dim=1)
        # pcc: (batch, speaker). feature dimension is averaded across.
        loss = -pcc[:, 0]
        for j in range(1, y_true.shape[3]):
            loss += pcc[:, j] / (y_true.shape[3] - 1)
        return loss


class SumPearsonLoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        # sum across feature and speaker dimension
        return -pearson_corrcoef(y_pred, y_true, dim=1).mean(dim=(1, 2))


class AbsPearsonLoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return -torch.abs(pearson_corrcoef(y_pred, y_true[..., 0], dim=1).mean(dim=-1))


class ContrastiveAbsPearsonLoss(_Loss):
    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:

        pcc = pearson_corrcoef(y_pred, y_true, dim=1).abs().mean(dim=1)
        # pcc: (batch, speaker). feature dimension is averaged across.
        loss = -pcc[:, 0]
        for j in range(1, y_true.shape[3]):
            loss += pcc[:, j] / (y_true.shape[3] - 1)
        return loss
