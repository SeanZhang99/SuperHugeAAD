import torch
import einops

from ...datasets.metadata_processing.data import MetaDataElement


CHANNEL1D_ENUM: dict[str, int] = {
    # "AF3",
    # "AF4",
    # "AF7",
    # "AF8",
    # "AFz",
    # "C1",
    # "C2",
    # "C3",
    # "C4",
    # "C5",
    # "C6",
    # "CP1",
    # "CP2",
    # "CP3",
    # "CP4",
    # "CP5",
    # "CP6",
    # "CPz",
    # "Cz",
    # "EXG3",
    # "EXG4",
    # "EXG5",
    # "EXG6",
    # "F1",
    # "F2",
    # "F3",
    # "F4",
    # "F5",
    # "F6",
    # "F7",
    # "F8",
    # "FC1",
    # "FC2",
    # "FC3",
    # "FC4",
    # "FC5",
    # "FC6",
    # "FCz",
    # "FT10",
    # "FT7",
    # "FT8",
    # "FT9",
    # "Fp1",
    # "Fp2",
    # "Fpz",
    # "Fz",
    # "Iz",
    # "O1",
    # "O2",
    # "Oz",
    # "P1",
    # "P10",
    # "P2",
    # "P3",
    # "P4",
    # "P5",
    # "P6",
    # "P7",
    # "P8",
    # "P9",
    # "PO10",
    # "PO3",
    # "PO4",
    # "PO7",
    # "PO8",
    # "PO9",
    # "POz",
    # "Pz",
    # "T7",
    # "T8",
    # "TP7",
    # "TP8",
}

CHANNEL2D_ENUM: dict[str, tuple[int]] = {}


class Channel1D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, metadata: MetaDataElement) -> torch.Tensor:
        y = torch.zeros(*x.shape[:-1], len(CHANNEL1D_ENUM))
        for b in range(x.shape[0]):
            for c in range(x.shape[-1]):
                chan_name = metadata.channel_infos[c]["name"][b]
                y[b, :, CHANNEL1D_ENUM[chan_name]] = x[b, :, c]
        return y


class Channel2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, metadata: MetaDataElement) -> torch.Tensor:
        y = torch.zeros(*x.shape[:-1], *max(CHANNEL2D_ENUM.values()) + 1)
        for b in range(x.shape[0]):
            for c in range(x.shape[-1]):
                chan_name = metadata.channel_infos[c]["name"][b]
                y[b, :, *CHANNEL2D_ENUM[chan_name]] = x[b, :, c]
        return y
