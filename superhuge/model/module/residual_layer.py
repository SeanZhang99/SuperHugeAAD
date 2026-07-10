import torch


class ResidualLayer(torch.nn.Module):
    def __init__(self, *args: torch.nn.Module):
        super().__init__()
        self.layers = torch.nn.ModuleList(args)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                y = layer(x)
            else:
                y += layer(x)  # type: ignore
        return y  # type: ignore
