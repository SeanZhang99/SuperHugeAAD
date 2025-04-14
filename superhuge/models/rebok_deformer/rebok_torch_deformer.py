import torch
import torch.nn as nn
from einops import rearrange
from pydantic import BaseModel, Field
from typing import Callable, Annotated
from einops.layers.torch import Rearrange, EinMix
#from .torch_deformer import *
from ...torch_models.deformer import transformer_encoder_layer,preconv, output_mlp
from ...models.commons.multi_head_attention import MultiHeadAttention
from ...models.commons.residual_layer import ResidualLayer

# ===========================
# Configuration Classes
# ===========================
class RebokTransformerParams(BaseModel):
    depth: Annotated[int, Field(gt=0)]
    num_heads: Annotated[int, Field(gt=0)]
    dim_heads: Annotated[int, Field(gt=0)]
    num_kernels: Annotated[int, Field(gt=0)]
    temporal_kernel_size: Annotated[int, Field(gt=0)]
    ff_hidden_dims: Annotated[int, Field(gt=0)]
    dp_rate: Annotated[float, Field(ge=0, le=1)]


class RebokDeformerParams(BaseModel):
    window_length: Annotated[int, Field(gt=0)]
    fs: Annotated[int, Field(gt=0)]
    num_kernels: Annotated[int, Field(gt=0)]
    temporal_kernel_size: Annotated[int, Field(gt=0)]
    mha_depth: Annotated[int, Field(gt=0)]
    mha_num_heads: Annotated[int, Field(gt=0)]
    mha_dim_heads: Annotated[int, Field(gt=0)]
    ff_hidden_dim: Annotated[int, Field(gt=0)]
    num_electrodes: Annotated[int, Field(gt=0)]
    dp_rate: Annotated[float, Field(ge=0, le=1)]
    preconv_callable: Callable[..., nn.Module] | None = None
    transformer_callable: Callable[..., nn.Module] | None = None


import torch
import torch.nn as nn

class PositionalEmbedding2D(nn.Module):
    """
    动态支持任意长度 [B, C, T] 的位置嵌入
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.pos_embedding = None

    def forward(self, x):
        B, C, T = x.shape

        # 初始化 or 调整位置嵌入长度
        if self.pos_embedding is None or self.pos_embedding.shape[-1] != T:
            self.pos_embedding = nn.Parameter(torch.randn(1, C, T, device=x.device))

        return x + self.pos_embedding



# ===========================
# Transformer Module
# ===========================

# ================
# 主体 Transformer
# ================
def rebok_transformer(
    depth: int,
    num_heads: int,
    dim_heads: int,
    num_kernels: int,
    fg_cnn_temporal_kernel_size: int,
    ff_hidden_dims: int,
    dp_rate: float,
    mha_embed_dim: int,
) -> nn.Module:
    """
    不显式写 forward，直接返回一个 nn.Sequential，
    其中每一层都是 rebok_transformer_layer(...) 生成的子模块。
    
    Args:
        depth (int): Transformer 层数
        num_heads (int): MultiHeadAttention 的 head 数量
        dim_heads (int): 每个 head 的维度
        fg_cnn_temporal_kernel_size (int): fg_cnn 的卷积核大小
        ff_hidden_dims (int): 前馈网络隐藏层大小
        dp_rate (float): Dropout 比例

    Returns:
        nn.Module: 一个顺序容器 (nn.Sequential)，包含若干层 rebok_transformer_layer
    """
    layer = transformer_encoder_layer(
        fg_cnn_num_kernels=num_kernels,
        fg_cnn_kernel_size=fg_cnn_temporal_kernel_size,
        time_dim=mha_embed_dim,
        mha_embed_dim=mha_embed_dim,
        mha_num_heads=num_heads,
        ff_hidden_dim=ff_hidden_dims,
        dropout=dp_rate,
            )
    return nn.Sequential(
    *[layer for _ in range(depth)]
        )

def deformer(
    num_electrodes: int,
    num_kernels: int,
    temporal_kernel_size: int,
    mha_embed_dim: int,
    mha_depth: int,
    mha_num_heads: int,
    mha_dim_heads: int,
    ff_hidden_dim: int,
    dropout: float = 0.0,
    window_length: int = 0,
):
    """
    input_shape: Sequence[int]
    num_kernels: int
    temporal_kernel_size: int
    mha_depth: int
    mha_num_heads: int
    mha_dim_per_head: int
    ff_hidden_dim: int
    dropout: float, Optional, default 0.0
    """
    return nn.Sequential(
        EinMix("b t c -> b 1 k t", weight_shape="c k", c=num_electrodes, k=num_kernels),
        preconv(
            num_kernels,
            num_kernels,
            temporal_kernel_size,
        ),  # (b, num_kernels, 1, num_time)
        Rearrange("b k c t -> b (k c) t"),  # (b, num_kernels, num_time)
        rebok_transformer(
            mha_depth,          # depth
            mha_num_heads,      # num_heads
            mha_dim_heads,      # dim_heads
            num_kernels,        # num_kernels
            temporal_kernel_size,
            ff_hidden_dim,     # ff_hidden_dims
            dropout,            # dp_rate
            mha_embed_dim,    # mha_embed_dim
        ),
        Rearrange("b k t -> b t k"),
        output_mlp(num_kernels, ff_hidden_dim, num_electrodes),
    )

def rebok_deformer(
    window_length: int,
    fs: int,
    num_kernels: int,
    temporal_kernel_size: int,
    mha_depth: int,
    mha_num_heads: int,
    mha_dim_heads: int,
    mha_embed_dim: int,
    ff_hidden_dim: int,
    time_dim: int,
    num_electrodes: int,
    dp_rate: float,
):
    """
    Constructs a deformer model with a transformer architecture. The transformer layer uses a single MHA module multiple times.

    Args:
        window_length (int): Length of the window.
        fs (int): Sampling frequency.
        num_kernels (int): Number of kernels.
        temporal_kernel_size (int): Size of the temporal kernel.
        mha_depth (int): Number of transformer layers.
        mha_num_heads (int): Number of attention heads.
        mha_dim_heads (int): Dimension of each attention head.
        ff_hidden_dim (int): Dimension of the hidden layer in the feedforward network.
        num_electrodes (int): Number of electrodes.
        dp_rate (float): Dropout rate.
        preconv_callable (Optional[Callable]): Callable to create the preconv model.
        transformer_callable (Optional[Callable]): Callable to create the transformer model.
    """
    params = RebokDeformerParams(
        window_length=window_length,
        fs=fs,
        num_kernels=num_kernels,
        temporal_kernel_size=temporal_kernel_size,
        mha_depth=mha_depth,
        mha_num_heads=mha_num_heads,
        mha_dim_heads=mha_dim_heads,
        ff_hidden_dim=ff_hidden_dim,
        num_electrodes=num_electrodes,
        dp_rate=dp_rate,
    )

    model = deformer(
        num_electrodes,
        num_kernels,
        temporal_kernel_size,
        mha_embed_dim,
        mha_depth,
        mha_num_heads,
        mha_dim_heads,
        ff_hidden_dim,
        dropout = dp_rate,
        window_length=window_length,
        )

    return model


# class DeformerWrapper(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.model = rebok_deformer(**kwargs)

#     def forward(self, x):
#         return self.model(x)
