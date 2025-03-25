# This is the script of EEG-Deformer
# This is the network script
from collections.abc import Iterable
from threading import local

import einops
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from pydantic import BaseModel
from torch import nn


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


def cnn_block(
    in_chan,
    out_chan,
    kernel_size,
    num_chan,
):
    return nn.Sequential(
        Conv2dWithConstraint(
            in_chan,
            out_chan,
            kernel_size,
            max_norm=2,
            padding="same",
        ),
        Conv2dWithConstraint(
            out_chan,
            out_chan,
            (num_chan, 1),
            max_norm=2,
            padding="valid",
        ),
        nn.BatchNorm2d(out_chan),
        nn.ELU(),
    )


class Deformer(nn.Module):

    @staticmethod
    def create_models(
        *,
        window_length: int,
        fs: int,
        num_kernels: int,
        temporal_kernel_size: int,
        mha_depth: int,
        mha_num_heads: int,
        mha_dim_heads: int,
        ff_hidden_dim: int,
        num_electrodes: int,
        dp_rate: float,
        **kwargs: dict,
    ):

        model_config = createDeformerInputConfig(
            num_kernels=num_kernels,
            temporal_kernel_size=temporal_kernel_size,
            mha_depth=mha_depth,
            mha_num_heads=mha_num_heads,
            mha_dim_heads=mha_dim_heads,
            fs=fs,
            window_length=window_length,
            ff_hidden_dim=ff_hidden_dim,
            num_electrodes=num_electrodes,
            dp_rate=dp_rate,
        )
        return Deformer(model_config=model_config)

    def __init__(self, *, model_config: createDeformerInputConfig):
        super().__init__()

        num_kernel = model_config.num_kernels
        temporal_kernel = model_config.temporal_kernel_size

        num_time = model_config.fs * model_config.window_length
        num_chan = model_config.num_electrodes

        self.cnn_encoder1 = cnn_block(
            in_chan=1,
            out_chan=num_kernel,
            kernel_size=(1, temporal_kernel),
            num_chan=num_chan,
        )

        self.transformer = Transformer(
            dim=num_time,
            config=model_config,
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(model_config.num_kernels, model_config.ff_hidden_dim),
            nn.ELU(),
            nn.Linear(
                model_config.ff_hidden_dim,
                model_config.num_electrodes,
            ),
        )

    def forward(self, eeg, *args):
        # eeg: (b, time, chan)
        eeg = einops.rearrange(eeg, "b t c -> b 1 c t")
        x = self.cnn_encoder1(eeg)  # (b, num_kernel, 1, num_time)
        x = einops.rearrange(x, "b k c t -> b k (c t)")
        x = self.transformer(x)
        x = einops.rearrange(x, "b k t -> b t k")
        return self.mlp_head(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def feed_forward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        config: createDeformerInputConfig,
    ):
        super().__init__()
        heads = config.mha_num_heads
        dim_head = config.mha_dim_heads
        dropout = config.dp_rate
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = torch.softmax(dots, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
        )

    def __init__(
        self,
        dim,
        config: createDeformerInputConfig,
    ):
        super().__init__()
        depth = config.mha_depth
        self.chan_attn_layers: Iterable = nn.ModuleList([])
        time_dim = dim
        for i in range(depth):
            self.chan_attn_layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=time_dim,
                            config=config,
                        ),
                        feed_forward(
                            dim=time_dim,
                            hidden_dim=config.ff_hidden_dim,
                            dropout=config.dp_rate,
                        ),
                        self.cnn_block(
                            in_chan=config.num_kernels,
                            kernel_size=config.temporal_kernel_size,
                            dp=config.dp_rate,
                        ),
                        nn.LayerNorm(
                            [
                                time_dim,
                            ]
                        ),
                        nn.LayerNorm(
                            [
                                time_dim,
                            ]
                        ),
                    ]
                )
            )

    def forward(self, x):
        for i, (attn, ff, cnn, ln1, ln2) in enumerate(self.chan_attn_layers):
            x_cg = ln1(attn(x) + x)
            x_fg = cnn(x)
            x = ln2(ff(x_cg) + x_fg)
        return x


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
