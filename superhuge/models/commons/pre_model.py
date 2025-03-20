import torch
import einops

from ...datasets.metadata_processing.data import MetaData, MetaDataElement


class Channel1D(torch.nn.Module):
    CHANNEL1D_ENUM = {
        "AF3": 33,
        "AF4": 52,
        "AF7": 32,
        "AF8": 51,
        "AFz": 53,
        "C1": 38,
        "C2": 59,
        "C3": 6,
        "C4": 27,
        "C5": 39,
        "C6": 60,
        "CP1": 11,
        "CP2": 22,
        "CP3": 41,
        "CP4": 62,
        "CP5": 10,
        "CP6": 23,
        "CPz": 49,
        "Cz": 0,
        "F1": 34,
        "F2": 54,
        "F3": 4,
        "F4": 29,
        "F5": 35,
        "F6": 55,
        "F7": 3,
        "F8": 30,
        "FC1": 5,
        "FC2": 28,
        "FC3": 37,
        "FC4": 57,
        "FC5": 7,
        "FC6": 26,
        "FCz": 58,
        "FT10": 25,
        "FT7": 36,
        "FT8": 56,
        "FT9": 8,
        "Fp1": 2,
        "Fp2": 31,
        "Fpz": 50,
        "Fz": 1,
        "Iz": 47,
        "O1": 15,
        "O2": 18,
        "Oz": 17,
        "P1": 42,
        "P10": 65,
        "P2": 63,
        "P3": 12,
        "P4": 21,
        "P5": 43,
        "P6": 64,
        "P7": 13,
        "P8": 20,
        "P9": 44,
        "PO10": 19,
        "PO3": 46,
        "PO4": 67,
        "PO7": 45,
        "PO8": 66,
        "PO9": 14,
        "POz": 48,
        "Pz": 16,
        "T7": 9,
        "T8": 24,
        "TP7": 40,
        "TP8": 61,
    }

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Perform a 1D channel rearrangement of the input tensor x, based on the given metadata.

        Args:
            x (torch.Tensor): The input tensor. expected shape: batch * time * channel
            metadata (dict): The metadata. expected key: channel_infos
                channel_infos should have this structure:
                metadata["channel_infos"] = {1: {"name": [`channel_name_sample1`,`channel_name_sample2`]}}
                In our implementation, the multidataset collect fn `collect_multidataset.collect_multidataset` handles data from different datasets,grouping them into different keys. And the date_interface will handle this grouped data, pass each group (correspond to samples coming from one specific dataset) into the forward path. Therefore, in this object, x is expected to be from the same dataset, thus with the same channel arrangement, making it possible to perform batch-wise channel rearrangement.
        """
        y = torch.zeros(*x.shape[:-1], len(self.CHANNEL1D_ENUM))
        # z-score normalization over batch
        x_mean = x.mean(dim=(1, 2), keepdim=True)
        x_std = x.std(dim=(1, 2), keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)
        for c in metadata["channel_infos"].keys():
            chan_name = metadata["channel_infos"][c]["name"][0]
            if chan_name in self.CHANNEL1D_ENUM:
                y[:, :, self.CHANNEL1D_ENUM[chan_name]] = x[:, :, c - 1]
        return y


class Channel2D(torch.nn.Module):
    CHANNEL2D_ENUM: dict[str, tuple[int]] = {}

    max_row = max([v[0] for v in CHANNEL2D_ENUM.values()])
    max_col = max([v[1] for v in CHANNEL2D_ENUM.values()])

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        y = torch.zeros(*x.shape[:-1], self.max_row, self.max_col)
        # z-score normalization over batch
        x_mean = x.mean(dim=(1, 2), keepdim=True)
        x_std = x.std(dim=(1, 2), keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)
        for c in metadata["channel_infos"].keys():
            chan_name = metadata["channel_infos"][c]["name"][0]
            if chan_name in self.CHANNEL2D_ENUM:
                y[:, :, *self.CHANNEL2D_ENUM[chan_name]] = x[:, :, c - 1]
        return y
