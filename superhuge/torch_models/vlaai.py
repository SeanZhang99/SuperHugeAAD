import gc
import torch
import torch.nn as nn
import einops
from einops.layers.torch import EinMix
from typing import Annotated, Sequence
from pydantic import BaseModel, Field


class ExtractorParams(BaseModel):
    num_kernels: int | Sequence[int]
    kernel_sizes: int | Sequence[int]
    num_layers: Annotated[int | None, Field(gt=0)] = None


class VlaaiParams(BaseModel):
    nb_blocks: Annotated[int, Field(gt=0)]
    use_skip: bool
    input_channels: Annotated[int, Field(gt=0)]
    extractor_args: dict = {}
    output_context_args: dict = {}


def extractor(
    num_kernels: Sequence[int] | int,
    kernel_sizes: Sequence[int] | int,
    input_channels: int,
    num_layers: int | None = None,
):

    params = ExtractorParams(
        num_kernels=num_kernels,
        kernel_sizes=kernel_sizes,
        num_layers=num_layers,
    )

    assert not (
        (isinstance(num_kernels, (int)) or len(num_kernels) == 1)
        or (isinstance(kernel_sizes, (int)) or len(kernel_sizes) == 1)
        or num_layers is None
    ), "Cannot infer the number of layers from given arguments. Expected num_kernels or kernel_sizes be full-length sequence (with length equal to the number of layers), or num_layers not None. If you intend to use a single layer, please set num_layers=1."

    if isinstance(num_kernels, int):
        num_kernels = [num_kernels] * (num_layers if num_layers else len(kernel_sizes))
    elif len(num_kernels) == 1:
        num_kernels = list(num_kernels) * (
            num_layers if num_layers else len(kernel_sizes)
        )

    if isinstance(kernel_sizes, int):
        kernel_sizes = [kernel_sizes] * (num_layers if num_layers else len(num_kernels))
    elif len(kernel_sizes) == 1:
        kernel_sizes = list(kernel_sizes) * (
            num_layers if num_layers else len(num_kernels)
        )

    layers = nn.Sequential()
    for i, (num_kernel, kernel_size) in enumerate(zip(num_kernels, kernel_sizes)):
        layers.append(
            nn.Sequential(
                nn.ZeroPad1d((0, kernel_size - 1)),
                nn.Conv1d(
                    input_channels if i == 0 else num_kernels[i - 1],
                    num_kernel,
                    kernel_size,
                ),
                nn.LayerNorm([num_kernel, 1280]),
                nn.LeakyReLU(),
            )
        )
    return layers


def output_context(
    input_channels: int,
    time_dim: int,
    kernel_size: int,
):
    return nn.Sequential(
        nn.ZeroPad1d((kernel_size - 1, 0)),
        nn.Conv1d(input_channels, input_channels, kernel_size),
        nn.LayerNorm([input_channels, time_dim]),
        nn.LeakyReLU(),
    )


def vlaai_block(
    *,
    extractor_args: dict,
    output_context_kernel_size: int,
    num_channels: int,
    time_dim: int,
):
    return nn.Sequential(
        extractor(**extractor_args),
        EinMix(
            "b k t -> b c t",
            weight_shape="k t",
            bias_shape="c",
            c=num_channels,
            k=extractor_args["num_kernels"][-1],
        ),
        output_context(
            input_channels=num_channels,
            kernel_size=output_context_kernel_size,
            time_dim=time_dim,
        ),
    )


class VLAAI(nn.Module):

    def __init__(
        self,
        /,
        *,
        nb_blocks: int,
        extractor_args: dict,
        output_context_kernel_size: int,
        use_skip: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.use_skip = use_skip
        if isinstance(nb_blocks, str):
            nb_blocks = int(nb_blocks)

        self.vlaai_block = nn.ModuleList(
            vlaai_block(
                extractor_args=ExtractorParams(**extractor_args).model_dump(),
                output_context_args=output_context_kernel_size,
                num_channels=kwargs["num_channels"],
                time_dim=kwargs["fs"] * kwargs["window_length"],
            )
            for _ in range(nb_blocks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward Call forward pass of VLAAI model.

        :param x: input tensor
        :type x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        # 将输入从 (b, t, c) 转换为 (b, c, t)
        x = einops.rearrange(x, "b t c -> b c t")
        x_hat = x
        for i, block in enumerate(self.vlaai_block):
            if self.use_skip and i > 0:
                x_hat = x_hat + x
            x_hat = block(x_hat)

        x_hat = einops.rearrange(x_hat, "b c t -> b t c")

        return x_hat


class VLAAI_ws(VLAAI):
    """
    VLAAI_ws weight sharing version of VLAAI.

    :param VLAAI: VLAAI class
    :type VLAAI: a subclass of torch.nn.Module
    """

    def __init__(self, /, **kwargs):
        """
        __init__ instantiate VLAAI_ws object.
        :param kwargs: kwargs for VLAAI. You may refer to VLAAI class for more details.
        """
        super().__init__(**kwargs)
        block = vlaai_block(
            extractor_args=ExtractorParams(**kwargs["extractor_args"]).model_dump(),
            output_context_args=kwargs["output_context_args"],
            num_channels=kwargs["num_channels"],
            time_dim=kwargs["fs"] * kwargs["window_length"],
        )
        self.vlaai_block = nn.ModuleList(block for _ in range(kwargs["nb_blocks"]))
        gc.collect()
