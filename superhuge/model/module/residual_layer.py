import torch


class ResidualLayer(torch.nn.Module):
    def __init__(self, *args: torch.nn.Module):
        super().__init__()
        self.layers = torch.nn.ModuleList(args)

    def forward(self, x):
        y = torch.zeros_like(x)
        for layer in self.layers:
            y += layer(x)
        return y
