from torch import nn
from einops.layers.torch import Rearrange, Reduce

from ...utils.validate import validate_kwargs


class SimpleCNN(nn.Module):
    def __init__(
        self,
        /,
        *,
        temporal_kernel_size: int,
        num_kernels: int,
        **kwargs,
    ):
        validate_kwargs(kwargs, ["num_channels"])
        super().__init__()
        self._simple_cnn = nn.Sequential(
            Rearrange("batch time channel -> batch 1 time channel"),
            # Padding order: last dimension, second last, ...
            nn.ZeroPad2d((0, 0, 0, temporal_kernel_size - 1)),
            nn.Conv2d(
                in_channels=1,
                out_channels=num_kernels,
                kernel_size=(temporal_kernel_size, kwargs["num_channels"]),
            ),
            # nn.BatchNorm2d(num_kernels),
            nn.ReLU(),
            Reduce(
                "batch num_kernels time channel -> batch time num_kernels",
                "mean",
                channel=1,
            ),
        )

    def forward(self, x):
        return self._simple_cnn(x)
