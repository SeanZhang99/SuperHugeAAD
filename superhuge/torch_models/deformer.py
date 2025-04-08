# ===========================
# Imports
# ===========================
import torch
from einops.layers.torch import Rearrange, EinMix
from pydantic import BaseModel
from torch import nn

from ..models.commons.lazy_layernorm import LazyLayerNorm

from ..models.commons.multi_head_attention import MultiHeadAttention

from ..models.commons.residual_layer import ResidualLayer


# ===========================
# Configuration Classes
# ===========================
class createDeformerInputConfig(BaseModel):
    window_length: int
    fs: int
    num_kernels: int
    temporal_kernel_size: int
    mha_depth: int
    mha_num_heads: int
    mha_dim_heads: int
    ff_hidden_dim: int
    num_electrodes: int
    dp_rate: float


# ===========================
# Utility Functions
# ===========================
def feed_forward(in_features, hidden_dim, dropout=0.0):
    model = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, in_features),
        nn.Dropout(dropout),
    )
    return model


def fg_cnn(num_kernels, kernel_size, dropout):
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Conv1d(
            num_kernels,
            num_kernels,
            kernel_size,
            padding="same",
        ),
        nn.BatchNorm1d(num_kernels),
        nn.ELU(),
    )


def transformer_encoder_layer(
    fg_cnn_num_kernels: int,
    fg_cnn_kernel_size: int,
    time_dim: int,
    mha_embed_dim: int,
    mha_num_heads: int,
    ff_hidden_dim: int,
    dropout: float = 0.0,
):
    """
    Args:
        input: (b, k, t)
        fg_cnn_kernel_size: int
        mha_num_heads: int
        ff_hidden_dim: int
    """
    return nn.Sequential(
        ResidualLayer(
            fg_cnn(
                fg_cnn_num_kernels,
                fg_cnn_kernel_size,
                dropout,
            ),
            nn.Sequential(
                ResidualLayer(
                    # 有点反直觉，EEG-Deformer原文就是将一个(1,t)的vector视为一个kernel的embedding，计算在kernel上的attention score
                    MultiHeadAttention(
                        time_dim,
                        mha_num_heads,
                        mha_embed_dim,
                        dropout,
                    ),
                    nn.Identity(),
                ),
                LazyLayerNorm(1),
                feed_forward(
                    time_dim,
                    ff_hidden_dim,
                    dropout,
                ),
            ),
        ),
        LazyLayerNorm(1),
    )


def transformer(
    depth: int,
    fg_cnn_num_kernels: int,
    fg_cnn_kernel_size: int,
    time_dim: int,
    mha_embed_dim: int,
    mha_num_heads: int,
    ff_hidden_dim: int,
    dropout: float = 0.0,
):
    """
    Args:
        input_shape: (b, k, t)
        depth: int
        fg_cnn_args: dict
            num_kernel: int, Optional, infer from input,
            kernel_size: int,
            dp_rate: float, Optional, infer from dropout
        mha_args: dict
            num_heads: int
            embed_dim: int, Optional, infer from input,
            dropout: float, Optional, infer from dropout
        ff_args: dict
            in_features: int, Optional, infer from input,
            hidden_dim: int,
            dropout: float, Optional, infer from dropout
        dropout: float, Optional, default 0.0
    """
    return nn.Sequential(
        *[
            transformer_encoder_layer(
                fg_cnn_num_kernels,
                fg_cnn_kernel_size,
                time_dim,
                mha_embed_dim,
                mha_num_heads,
                ff_hidden_dim,
                dropout,
            )
            for _ in range(depth)
        ]
    )


# ===========================
# Custom Layers
# ===========================
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(input)


# ===========================
# Model Components
# ===========================
def preconv(
    num_electrodes: int,
    out_chan: int,
    kernel_size: int,
):
    """
    input: torch.Tensor
    out_chan: int
    kernel_size: int
    """
    assert isinstance(out_chan, int)
    assert isinstance(kernel_size, int)
    return nn.Sequential(
        Conv2dWithConstraint(
            1,
            out_chan,
            (1, kernel_size),
            max_norm=2,
            padding="same",
        ),
        Conv2dWithConstraint(
            out_chan,
            out_chan,
            (num_electrodes, 1),
            max_norm=2,
            padding="valid",
        ),
        nn.BatchNorm2d(out_chan),
        nn.ELU(),
    )


def output_mlp(in_features, ff_hidden_dim, num_electrodes):
    return nn.Sequential(
        nn.Linear(in_features, ff_hidden_dim),
        nn.ELU(),
        nn.Linear(
            ff_hidden_dim,
            num_electrodes,
        ),
    )


# ===========================
# Main Model Definition
# ===========================
def deformer(
    num_electrodes: int,
    num_kernels: int,
    temporal_kernel_size: int,
    time_dim: int,
    mha_embed_dim: int,
    mha_depth: int,
    mha_num_heads: int,
    ff_hidden_dim: int,
    dropout: float = 0.0,
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
        # Rearrange("b t c -> b 1 c t"),
        EinMix("b t c -> b 1 k t", weight_shape="c k", c=num_electrodes, k=num_kernels),
        preconv(
            num_kernels,
            num_kernels,
            temporal_kernel_size,
        ),  # (b, num_kernels, 1, num_time)
        Rearrange("b k c t -> b k (c t)"),  # (b, num_kernels, num_time)
        transformer(
            mha_depth,
            num_kernels,
            temporal_kernel_size,
            time_dim,
            mha_embed_dim,
            mha_num_heads,
            ff_hidden_dim,
            dropout,
        ),
        Rearrange("b k t -> b t k"),
        output_mlp(num_kernels, ff_hidden_dim, num_electrodes),
    )
