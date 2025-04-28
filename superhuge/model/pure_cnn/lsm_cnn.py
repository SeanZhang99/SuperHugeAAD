from pydantic import BaseModel, Field
from typing import Annotated, Sequence, Tuple
from torch import nn, Tensor
from einops.layers.torch import EinMix, Rearrange


def learnable_mapping(
    num_chan: int,
    chan_dim: int,
    dropout: float = 0.0,
) -> nn.Module:
    class lsm_config(BaseModel):
        num_chan: Annotated[int, Field(ge=1)]
        chan_dim: Annotated[int, Field(ge=1)]
        dropout: Annotated[float, Field(ge=0.0, le=1.0)]

    lsm_cfg = lsm_config(num_chan=num_chan, chan_dim=chan_dim, dropout=dropout)

    return nn.Sequential(
        EinMix(
            "batch time channel -> batch channels time",
            weight_shape="channel channels",
            channel=lsm_cfg.num_chan,
            channels=lsm_cfg.chan_dim**2,
        ),
        nn.BatchNorm1d(lsm_cfg.chan_dim**2),
        nn.ELU(),
        nn.Dropout(lsm_cfg.dropout),
        Rearrange(
            "batch (c1 c2) time -> batch 1 time c1 c2",
            c1=lsm_cfg.chan_dim,
        ),
    )


def cnn_block(
    in_features: int,
    num_layers: int,
    temporal_kernel_size: int | Sequence[int],
    channel_kernel_size: int | Sequence[int],
    num_kernels: int | Sequence[int],
    dropout: float = 0.0,
) -> Tuple[nn.Module, Tensor]:
    class cnn_config(BaseModel):
        num_layers: Annotated[int, Field(ge=1)]
        temporal_kernel_size: Annotated[Sequence[int], Field(min_length=1)]
        channel_kernel_size: Annotated[Sequence[int], Field(min_length=1)]
        num_kernels: Annotated[Sequence[int], Field(min_length=1)]
        dropout: Annotated[float, Field(ge=0.0, le=1.0)]
        in_features: Annotated[int, Field(ge=1)]

    if isinstance(num_kernels, int):
        num_kernels = [num_kernels] * num_layers
    if isinstance(temporal_kernel_size, int):
        temporal_kernel_size = [temporal_kernel_size] * num_layers
    if isinstance(channel_kernel_size, int):
        channel_kernel_size = [channel_kernel_size] * num_layers

    cfg = cnn_config(
        num_layers=num_layers,
        temporal_kernel_size=temporal_kernel_size,
        channel_kernel_size=channel_kernel_size,
        num_kernels=num_kernels,
        dropout=dropout,
        in_features=in_features,
    )

    assert len(cfg.temporal_kernel_size) == cfg.num_layers
    assert len(cfg.channel_kernel_size) == cfg.num_layers
    assert len(cfg.num_kernels) == cfg.num_layers

    model = nn.Sequential()
    for layer_idx in range(num_layers):
        cnn = nn.Sequential(
            nn.Conv3d(
                in_channels=(
                    cfg.in_features
                    if layer_idx == 0
                    else cfg.num_kernels[layer_idx - 1]
                ),
                out_channels=cfg.num_kernels[layer_idx],
                kernel_size=(
                    cfg.temporal_kernel_size[layer_idx],
                    cfg.channel_kernel_size[layer_idx],
                    cfg.channel_kernel_size[layer_idx],
                ),
                padding="same",
            ),
            nn.ELU(),
            nn.BatchNorm3d(cfg.num_kernels[layer_idx]),
            nn.Dropout(cfg.dropout),
        )
        model.append(cnn)

    return model


class LSM_CNN(nn.Module):
    def __init__(
        self,
        /,
        *,
        lsm_chan_dim: int,
        cnn_num_layers: int,
        cnn_temporal_kernel_size: int | Sequence[int],
        cnn_channel_kernel_size: int | Sequence[int],
        cnn_num_kernels: int | Sequence[int],
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self._lsm_cnn = nn.Sequential(
            learnable_mapping(kwargs["num_channels"], lsm_chan_dim, dropout),
            cnn_block(
                in_features=1,
                num_layers=cnn_num_layers,
                temporal_kernel_size=cnn_temporal_kernel_size,
                channel_kernel_size=cnn_channel_kernel_size,
                num_kernels=cnn_num_kernels,
                dropout=dropout,
            ),
            Rearrange(
                "batch feature time c1 c2 -> batch time (feature c1 c2)",
            ),
        )

    def forward(self, x, *args, **kwargs):
        return self._lsm_cnn(x)
