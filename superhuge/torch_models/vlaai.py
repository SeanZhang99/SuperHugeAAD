import inspect
import torch
import torch.nn as nn
import einops
from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from ..models.commons.lazy_layernorm import LazyLayerNorm
from superhuge.models.commons.residual_layer import ResidualLayer


class ExtractorParams(BaseModel):
    num_kernels: Annotated[Sequence[int], Field(min_length=1)]
    kernel_sizes: Annotated[Sequence[int], Field(min_length=1)]
    num_layers: Annotated[int | None, Field(gt=0)] = None
    name: str = "extractor"


class OutputContextParams(BaseModel):
    input_channels: Annotated[int, Field(gt=0)]
    num_kernel: Annotated[int, Field(gt=0)] = 64
    kernel_size: Annotated[int, Field(gt=0)] = 32
    name: str = "output_context_model"


class VlaaiParams(BaseModel):
    nb_blocks: Annotated[int, Field(gt=0)] = 4
    use_skip: bool = True
    input_channels: Annotated[int, Field(gt=0)] = 64
    extractor_args: dict = {}
    output_context_args: dict = {}
    name: str = "vlaai"


def extractor_layer(
    num_kernels: int,
    kernel_size: int,
    input_channels: int,
):

    return nn.Sequential(
        nn.ZeroPad1d((0, kernel_size - 1)),
        nn.Conv1d(input_channels, num_kernels, kernel_size),
        nn.LayerNorm([num_kernels, 1280]),
        nn.LeakyReLU(),
    )


def extractor(
    num_kernels: Sequence[int] = (256, 256, 256, 128, 128),
    kernel_sizes: Sequence[int] = (8,) * 5,
    num_layers: int | None = None,
    input_channels: int = 68,
):
    params = ExtractorParams(
        num_kernels=num_kernels,
        kernel_sizes=kernel_sizes,
        num_layers=num_layers,
    )
    assert (
        len(num_kernels) == len(kernel_sizes)
        or len(num_kernels) == 1
        or len(kernel_sizes) == 1
    ), f"num_kernels and kernel_sizes must: 1. have the same length, or 2. one of them contain only one element , but got {len(num_kernels)} and {len(kernel_sizes)}"
    num_kernels = list(num_kernels)
    kernel_sizes = list(kernel_sizes)
    if len(num_kernels) == 1:
        num_kernels = num_kernels * (num_layers if num_layers else len(kernel_sizes))
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes * (num_layers if num_layers else len(num_kernels))
    layers = []
    for i, (num_kernel, kernel_size) in enumerate(zip(num_kernels, kernel_sizes)):
        layers.append(
            extractor_layer(
                input_channels=input_channels if i == 0 else num_kernels[i - 1],
                num_kernels=num_kernel,
                kernel_size=kernel_size,
            )
        )
    return nn.Sequential(*layers)


def output_context(
    input_channels: int = 16,
    num_kernel: int = 64,
    kernel_size: int = 32,
):
    return nn.Sequential(
        nn.ZeroPad1d((kernel_size - 1, 0)),
        nn.Conv1d(input_channels, num_kernel, kernel_size),
        nn.LayerNorm([num_kernel, 1280]),
        nn.LeakyReLU(),
    )


class VLAAI(nn.Module):
    @staticmethod
    def create_model(**kwargs):
        return VLAAI(**kwargs)

    def __init__(
        self,
        nb_blocks: int = 4,
        window_length: int = 10,
        fs: int = 128,
        use_skip: bool = True,
        input_channels: int = 64,
        extractor_args: dict = {},
        output_context_args: dict = {},
        name="vlaai",
    ):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.use_skip = use_skip
        self.input_channels = input_channels
        self.window_length = window_length
        self.fs = fs
        if isinstance(nb_blocks, str):
            nb_blocks = int(nb_blocks)

        self.extractor_model = extractor(
            input_channels=input_channels, **extractor_args
        )
        self.output_context_model = output_context(
            input_channels=extractor_args.get(
                "num_kernels",
                [
                    128,
                ],
            )[-1],
            **output_context_args,
        )

        dense_in_features = output_context_args.get("num_kernel")
        self.dense_layers = nn.ModuleList(
            [nn.Linear(dense_in_features, input_channels) for _ in range(nb_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数：
            x: 输入张量，形状为 (batch, window_length * fs, input_channels)
        返回：
            输出张量
        """
        # 将输入从 (b, t, c) 转换为 (b, c, t)
        reshaped_eeg = x.permute(0, 2, 1)
        for i in range(self.nb_blocks):
            if self.use_skip:
                if i == 0:
                    out = self.extractor_model(reshaped_eeg)
                else:

                    out = self.extractor_model(reshaped_eeg + out)
            else:
                out = self.extractor_model(reshaped_eeg)

            out = self.output_context_model(out)
            # 将输出从 (b, c, t) 转换为 (b, t, c) 以便应用全连接层
            out = out.permute(0, 2, 1)
            out = self.dense_layers[i](out)
            # 如果不是最后一个 block，则需要再转换回 (b, c, t) 供下一 block 使用
            if i < self.nb_blocks - 1:
                out = out.permute(0, 2, 1)

        return out


# def __repr__(self) -> str:
# return f"VLAAI(nb_blocks={self.nb_blocks},
# use_skip={self.use_skip},
# input_channels={self.input_channels},
#  window_length={self.window_length}, fs={self.fs}) \n {super().__repr__()}"

# create_model.__signature__ = inspect.signature(VLAAI.__init__)
