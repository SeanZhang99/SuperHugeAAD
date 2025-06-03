import gc
from typing import Annotated, Sequence

import einops
from einops.layers.torch import EinMix
from pydantic import BaseModel, Field, model_validator
from torch import Tensor, nn

from ..module.convnd_with_constraint import convNd_with_constraint


class ExtractorParams(BaseModel):
    num_kernels: int | Sequence[int]
    kernel_sizes: int | Sequence[int]
    num_layers: Annotated[int | None, Field(gt=0)] = None

    @model_validator(mode="before")
    @classmethod
    def validate_and_expand(cls, values: dict):
        nk = values.get("num_kernels")
        ks = values.get("kernel_sizes")
        nl = values.get("num_layers")

        def is_single(x):
            return isinstance(x, int) or (hasattr(x, "__len__") and len(x) == 1)

        # Cannot infer num_layers
        if is_single(nk) and is_single(ks) and nl is None:
            raise ValueError(
                "Cannot infer the number of layers from given arguments. "
                "Expected num_kernels or kernel_sizes to be full-length sequences, "
                "or num_layers not None. If you intend to use a single layer, please set num_layers=1."
            )

        # Infer num_layers
        if nl is None:
            if not is_single(nk):
                nl = len(nk)
            else:
                nl = len(ks) if not is_single(ks) else 1
            values["num_layers"] = nl

        # Expand num_kernels
        if isinstance(nk, int):
            values["num_kernels"] = [nk] * nl
        elif hasattr(nk, "__len__") and len(nk) == 1:
            values["num_kernels"] = list(nk) * nl

        # Expand kernel_sizes
        if isinstance(ks, int):
            values["kernel_sizes"] = [ks] * nl
        elif hasattr(ks, "__len__") and len(ks) == 1:
            values["kernel_sizes"] = list(ks) * nl

        return values


def extractor(
    num_kernels: Sequence[int] | int,
    kernel_sizes: Sequence[int] | int,
    input_channels: int,
    input_time_dim: int,
    num_layers: int,
    drouput: float = 0.0,
):

    layers = nn.Sequential()
    for i, (num_kernel, kernel_size) in enumerate(zip(num_kernels, kernel_sizes)):
        layers.append(
            nn.Sequential(
                nn.ZeroPad1d((0, kernel_size - 1)),
                convNd_with_constraint(
                    nd=1,
                    max_norm=2,
                    in_channels=input_channels if i == 0 else num_kernels[i - 1],
                    out_channels=num_kernel,
                    kernel_size=kernel_size,
                ),
                # nn.LayerNorm([num_kernel, input_time_dim]),
                nn.BatchNorm1d(num_kernel),
                nn.ELU(),
                nn.Dropout(drouput),
            )
        )
    return layers


def output_context(
    input_channels: int,
    time_dim: int,
    kernel_size: int,
    drouput: float = 0.0,
):
    return nn.Sequential(
        nn.ZeroPad1d((kernel_size - 1, 0)),
        convNd_with_constraint(
            nd=1,
            max_norm=2,
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=kernel_size,
        ),
        # nn.LayerNorm([input_channels, time_dim]),
        nn.BatchNorm1d(input_channels),
        nn.ELU(),
        nn.Dropout(drouput),
    )


def vlaai_block(
    *,
    extractor_args: ExtractorParams,
    output_context_kernel_size: int,
    num_channels: int,
    time_dim: int,
    drouput: float = 0.0,
):
    return nn.Sequential(
        extractor(
            **extractor_args.model_dump(),
            input_channels=num_channels,
            drouput=drouput,
            input_time_dim=time_dim,
        ),
        # EinMix(
        #     "b k t -> b c t",
        #     weight_shape="k c",
        #     bias_shape="c",
        #     c=num_channels,
        #     k=extractor_args.num_kernels[-1],
        # ),
        nn.Dropout(drouput),
        output_context(
            input_channels=extractor_args.num_kernels[-1],
            kernel_size=output_context_kernel_size,
            time_dim=time_dim,
            drouput=drouput,
        ),
    )


class VLAAI(nn.Module):

    def __init__(
        self,
        /,
        *,
        nb_blocks: int,
        extractor_args: ExtractorParams,
        output_context_kernel_size: int,
        use_skip: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_skip = use_skip
        if isinstance(nb_blocks, str):
            nb_blocks = int(nb_blocks)

        self._vlaai_blocks = nn.ModuleList(
            vlaai_block(
                extractor_args=extractor_args,
                output_context_kernel_size=output_context_kernel_size,
                num_channels=kwargs["num_channels"],
                time_dim=kwargs["fs"] * kwargs["window_length"],
                drouput=dropout,
            )
            for _ in range(nb_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
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
        for i, block in enumerate(self._vlaai_blocks):
            if self.use_skip and i > 0:
                # Problem:
                # 1st layer: no effect. then x_hat get normalized after x_hat=block(x_hat)
                # 2nd layer: x_hat already normalized, but x not normalized.
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
            extractor_args=kwargs["extractor_args"],
            output_context_kernel_size=kwargs["output_context_kernel_size"],
            num_channels=kwargs["num_channels"],
            time_dim=kwargs["fs"] * kwargs["window_length"],
            drouput=kwargs["dropout"],
        )
        self.vlaai_block = nn.ModuleList(block for _ in range(kwargs["nb_blocks"]))
        gc.collect()
