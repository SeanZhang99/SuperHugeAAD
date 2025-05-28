import torch


class CrossEntropyLoss(torch.nn.modules.loss._Loss):
    def forward(
        self, outputs: torch.Tensor, targets: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(
            outputs, targets, reduction="none", *args, **kwargs
        )
