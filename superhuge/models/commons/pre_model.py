import torch
import einops

from ...datasets.metadata_processing.data import MetaData, MetaDataElement


class Channel1D(torch.nn.Module):
    CHANNEL1D_ENUM = {
        "AF3": 0,
        "AF4": 1,
        "AF7": 2,
        "AF8": 3,
        "AFz": 4,
        "C1": 5,
        "C2": 6,
        "C3": 7,
        "C4": 8,
        "C5": 9,
        "C6": 10,
        "CP1": 11,
        "CP2": 12,
        "CP3": 13,
        "CP4": 14,
        "CP5": 15,
        "CP6": 16,
        "CPz": 17,
        "Cz": 18,
        "F1": 19,
        "F2": 20,
        "F3": 21,
        "F4": 22,
        "F5": 23,
        "F6": 24,
        "F7": 25,
        "F8": 26,
        "FC1": 27,
        "FC2": 28,
        "FC3": 29,
        "FC4": 30,
        "FC5": 31,
        "FC6": 32,
        "FCz": 33,
        "FT10": 34,
        "FT7": 35,
        "FT8": 36,
        "FT9": 37,
        "Fp1": 38,
        "Fp2": 39,
        "Fpz": 40,
        "Fz": 41,
        "Iz": 42,
        "O1": 43,
        "O2": 44,
        "Oz": 45,
        "P1": 46,
        "P10": 47,
        "P2": 48,
        "P3": 49,
        "P4": 50,
        "P5": 51,
        "P6": 52,
        "P7": 53,
        "P8": 54,
        "P9": 55,
        "PO10": 56,
        "PO3": 57,
        "PO4": 58,
        "PO7": 59,
        "PO8": 60,
        "PO9": 61,
        "POz": 62,
        "Pz": 63,
        "T7": 64,
        "T8": 65,
        "TP7": 66,
        "TP8": 67,
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
    CHANNEL2D_ENUM: dict[str, tuple[int]] = {
        "AF3": (1, 3),
        "AF4": (1, 7),
        "AF7": (1, 1),
        "AF8": (1, 9),
        "AFz": (1, 5),
        "C1": (4, 4),
        "C2": (4, 6),
        "C3": (4, 3),
        "C4": (4, 7),
        "C5": (4, 2),
        "C6": (4, 8),
        "CP1": (5, 4),
        "CP2": (5, 6),
        "CP3": (5, 3),
        "CP4": (5, 7),
        "CP5": (5, 2),
        "CP6": (5, 8),
        "CPz": (5, 5),
        "Cz": (4, 5),
        "F1": (2, 4),
        "F2": (2, 6),
        "F3": (2, 3),
        "F4": (2, 7),
        "F5": (2, 2),
        "F6": (2, 8),
        "F7": (2, 1),
        "F8": (2, 9),
        "FC1": (3, 4),
        "FC2": (3, 6),
        "FC3": (3, 3),
        "FC4": (3, 7),
        "FC5": (3, 2),
        "FC6": (3, 8),
        "FCz": (3, 5),
        "FT10": (3, 10),
        "FT7": (3, 1),
        "FT8": (3, 9),
        "FT9": (3, 0),
        "Fp1": (0, 4),
        "Fp2": (0, 6),
        "Fpz": (0, 5),
        "Fz": (2, 5),
        "Iz": (9, 5),
        "O1": (8, 4),
        "O2": (8, 6),
        "Oz": (8, 5),
        "P1": (6, 4),
        "P10": (6, 10),
        "P2": (6, 6),
        "P3": (6, 3),
        "P4": (6, 7),
        "P5": (6, 2),
        "P6": (6, 8),
        "P7": (6, 1),
        "P8": (6, 9),
        "P9": (6, 0),
        "PO10": (7, 10),
        "PO3": (7, 3),
        "PO4": (7, 7),
        "PO7": (7, 1),
        "PO8": (7, 9),
        "PO9": (7, 0),
        "POz": (7, 5),
        "Pz": (6, 5),
        "T7": (4, 1),
        "T8": (4, 9),
        "TP7": (5, 1),
        "TP8": (5, 9),
    }

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
