import torch
from .f1_score import prepare_targets


def binary_auc_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    /,
    *,
    positive_label: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes AUC (Area Under ROC Curve) for binary classification.

    Args:
        probs (torch.Tensor): Probabilities for the positive class, shape (N,)
        targets (torch.Tensor): True binary labels (0 or 1), shape (N,)
        positive_label (int): Which class is considered positive (default: 1)
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: AUC score (scalar)
    """
    probs, targets = prepare_targets(probs, targets)

    # Sort by predicted prob
    _, indices = torch.sort(probs, descending=True)
    sorted_targets = targets[indices]

    pos = (sorted_targets == positive_label).float()
    neg = (sorted_targets != positive_label).float()

    tp_cumsum = torch.cumsum(pos, dim=0)
    fp_cumsum = torch.cumsum(neg, dim=0)

    tpr = tp_cumsum / (pos.sum() + eps)
    fpr = fp_cumsum / (neg.sum() + eps)

    auc = torch.trapz(tpr, fpr)
    return auc


def binary_roc_curve(
    probs: torch.Tensor,
    targets: torch.Tensor,
    /,
    *,
    positive_label: int = 0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the ROC curve points for binary classification.

    Args:
        probs (torch.Tensor): Probabilities for the positive class, shape (N,)
        targets (torch.Tensor): True binary labels (0 or 1), shape (N,)
        positive_label (int): Which class is considered positive (default: 1)
        eps (float): Small value to avoid division by zero

    Returns:
        (fpr, tpr): Two tensors of shape (N,) for ROC curve plotting
    """
    probs, targets = prepare_targets(probs, targets)

    _, indices = torch.sort(probs, descending=True)
    sorted_targets = targets[indices]

    pos = (sorted_targets == positive_label).float()
    neg = (sorted_targets != positive_label).float()

    tp_cumsum = torch.cumsum(pos, dim=0)
    fp_cumsum = torch.cumsum(neg, dim=0)

    tpr = tp_cumsum / (pos.sum() + eps)
    fpr = fp_cumsum / (neg.sum() + eps)

    return fpr, tpr


def multiclass_auc_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    /,
    *,
    average: str = "macro",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes AUC for binary or multi-class classification using One-vs-Rest (OvR),
    from raw logits (no softmax required externally).

    Args:
        logits (torch.Tensor): Raw model outputs, shape (N, C)
        targets (torch.Tensor): True class labels, shape (N,)
        average (str): "macro" or "weighted"
        eps (float): Small value to avoid division by zero

    Returns:
        torch.Tensor: Averaged AUC score over classes
    """
    assert average in ["macro", "weighted"], f"Invalid average: {average}"
    assert probs.ndim == 2, "Expected logits shape (N, C)"
    assert targets.ndim == 1, "Expected targets shape (N,)"
    assert probs.shape[0] == targets.shape[0], "Batch size mismatch"

    num_classes = probs.shape[1]
    assert targets.max().item() < num_classes, "Target index out of bounds"

    # Apply softmax to get class probabilities
    probs = torch.softmax(probs, dim=1)

    aucs = []
    weights = []

    for class_index in range(num_classes):
        binary_targets = (targets == class_index).float()
        class_probs = probs[:, class_index]

        # Sort by predicted score
        _, indices = torch.sort(class_probs, descending=True)
        sorted_targets = binary_targets[indices]

        pos = sorted_targets
        neg = 1.0 - pos

        tp_cumsum = torch.cumsum(pos, dim=0)
        fp_cumsum = torch.cumsum(neg, dim=0)

        tpr = tp_cumsum / (pos.sum() + eps)
        fpr = fp_cumsum / (neg.sum() + eps)

        auc = torch.trapz(tpr, fpr)
        aucs.append(auc)
        weights.append((targets == class_index).sum())

    if len(aucs) == 0:
        return torch.tensor(0.0, device=probs.device, dtype=torch.float32)

    aucs_tensor = torch.stack(aucs)
    weights_tensor = torch.stack(weights).float()

    if average == "macro":
        return aucs_tensor.mean()
    elif average == "weighted":
        return (aucs_tensor * weights_tensor / weights_tensor.sum()).sum()
    else:
        return torch.tensor(0.0, device=probs.device, dtype=torch.float32)
