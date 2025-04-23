# grouped_adam.py
import torch
from torch.optim import Optimizer
from collections.abc import Iterator
from torch.nn import Parameter


class AutoDetachBiasDecay(Optimizer):

    def __new__(
        cls,
        /,
        named_params: Iterator[tuple[str, Parameter]],
        optimizer_class: type[Optimizer],
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-2,
        **kwargs,
    ):
        decay, no_decay = [], []
        for name, param in named_params:
            if not param.requires_grad:
                continue
            if "bias" in name or "Norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        grouped_params = [
            {"params": decay, "weight_decay": weight_decay, "lr": lr * 1.7},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr * 0.3},
        ]

        return optimizer_class(grouped_params, **kwargs)
