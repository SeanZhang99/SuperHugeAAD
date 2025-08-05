import torch


def prepare_targets(
    preds: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures targets has same shape as preds. Handles:
    - scalar targets: broadcast
    - shape (B, 1): squeeze to (B,)
    - mismatched shapes: raise error

    Args:
        preds (torch.Tensor): Predicted labels, shape (B,)
        targets (torch.Tensor): True labels, shape (B,) or scalar

    Returns:
        torch.Tensor: Broadcasted/squeezed targets with shape matching preds
    """
    if targets.ndim == 1 and targets.shape[0] == 1:
        targets = targets.repeat(preds.shape[0])

    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds.squeeze(1)

    if preds.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Shape mismatch: preds {preds.shape}, targets {targets.shape}"
        )

    return preds, targets


def binary_f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    /,
    *,
    positive_label: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the F1-score for binary classification focusing on the positive class.

    Args:
        preds (torch.Tensor): Predicted labels (0 or 1), shape (N,)
        targets (torch.Tensor): True labels (0 or 1), shape (N,)
        positive_label (int): Which label is considered positive (default: 1)
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: F1-score as scalar.
    """
    preds, targets = prepare_targets(preds, targets)

    TP = ((preds == positive_label) & (targets == positive_label)).sum()
    FP = ((preds == positive_label) & (targets != positive_label)).sum()
    FN = ((preds != positive_label) & (targets == positive_label)).sum()

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def micro_f1_score(
    preds: torch.Tensor, targets: torch.Tensor, /, *, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the micro-averaged F1-score across all classes.

    Args:
        preds (torch.Tensor): Predicted labels, shape (N,)
        targets (torch.Tensor): True labels, shape (N,)
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Micro F1-score as scalar.
    """
    preds, targets = prepare_targets(preds, targets)

    TP = (preds == targets).sum()
    total_pred = preds.numel()

    precision = TP / (total_pred + eps)
    recall = TP / (targets.numel() + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1


def macro_f1_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    /,
    *,
    num_classes: int | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the macro-averaged F1-score (equal weight per class).

    Args:
        preds (torch.Tensor): Predicted labels, shape (N,)
        targets (torch.Tensor): True labels, shape (N,)
        num_classes (int, optional): Number of classes; inferred if None.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Macro F1-score as scalar.
    """
    preds, targets = prepare_targets(preds, targets)

    if num_classes is None:
        num_classes = int(torch.max(torch.cat([preds, targets]))) + 1

    f1_list = []

    for cls in range(num_classes):
        pred_pos = preds == cls
        true_pos = targets == cls

        TP = (pred_pos & true_pos).sum()
        FP = (pred_pos & ~true_pos).sum()
        FN = (~pred_pos & true_pos).sum()

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_list.append(f1)

    return torch.stack(f1_list).mean()
