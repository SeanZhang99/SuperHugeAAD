from torch import nn
from einops.layers.torch import Rearrange, Reduce


def simple_cnn(
    temporal_kernel_size: int,
    num_kernels: int,
    num_chan: int,
):
    return nn.Sequential(
        Rearrange("batch time channel -> batch 1 time channel"),
        nn.ZeroPad2d((0, temporal_kernel_size - 1, 0, 0)),
        nn.Conv2d(
            in_channels=1,
            out_channels=num_kernels,
            kernel_size=(temporal_kernel_size, num_chan),
        ),
        nn.ReLU(),
        Reduce("batch num_kernels time channel -> batch time num_kernels", "mean"),
    )
