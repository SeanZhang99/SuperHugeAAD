import torch
from einops.layers.torch import EinMix

from ...utils.channel_enum import CHANNEL1D_ENUM, CHANNEL2D_ENUM, NUM_ELECTRODES


class Channel1D(torch.nn.Module):
    _num_electrodes = NUM_ELECTRODES

    def __init__(self, /, **kwargs):
        super().__init__()

    def forward(self, data: dict) -> torch.Tensor:
        """
        Perform a 1D channel rearrangement of the input tensor x, based on the given metadata.

        Args:
            x (torch.Tensor): The input tensor. expected shape: batch * time * channel
            metadata (dict): The metadata. expected key: channel_infos
                channel_infos should have this structure:
                metadata["channel_infos"] = {1: {"name": [`channel_name_sample1`,`channel_name_sample2`]}}
                In our implementation, the multidataset collect fn `collect_multidataset.collect_multidataset` handles data from different datasets,grouping them into different keys. And the date_interface will handle this grouped data, pass each group (correspond to samples coming from one specific dataset) into the forward path. Therefore, in this object, x is expected to be from the same dataset, thus with the same channel arrangement, making it possible to perform batch-wise channel rearrangement.
        """
        x: torch.Tensor = data["eeg"]
        metadata: dict = data["meta"]
        y = torch.zeros(
            *x.shape[:-1], len(CHANNEL1D_ENUM), device=x.device, dtype=x.dtype
        )
        original_ch_idx = []
        target_ch_idx = []

        for c in metadata["channel_infos"].keys():
            chan_name = metadata["channel_infos"][c]["name"]
            if isinstance(chan_name, list):
                chan_name = chan_name[0]
            if chan_name in CHANNEL1D_ENUM.__members__.keys():
                original_ch_idx.append(c - 1)
                target_ch_idx.append(CHANNEL1D_ENUM[chan_name].value)
        y[..., target_ch_idx] = x[..., original_ch_idx]

        return y

    @property
    def num_channels(self):
        return self._num_electrodes

    @property
    def num_electrodes(self):
        return self._num_electrodes


class Channel1DMixer(Channel1D):
    def __init__(self, /, num_mix_out_channels, **kwargs):
        super().__init__(**kwargs)
        self._channel_mixer = torch.nn.Sequential(
            EinMix(
                "b t c -> b t m",
                m=num_mix_out_channels,
                c=self.num_electrodes,
                weight_shape="c m",
                bias_shape="m",
            ),
        )
        self._num_mix_out_channels = num_mix_out_channels

    def forward(self, data: dict) -> torch.Tensor:
        """
        Perform a 1D channel rearrangement of the input tensor x, based on the given metadata.

        Args:
            x (torch.Tensor): The input tensor. expected shape: batch * time * channel
            metadata (dict): The metadata. expected key: channel_infos
                channel_infos should have this structure:
                metadata["channel_infos"] = {1: {"name": [`channel_name_sample1`,`channel_name_sample2`]}}
                In our implementation, the multidataset collect fn `collect_multidataset.collect_multidataset` handles data from different datasets,grouping them into different keys. And the date_interface will handle this grouped data, pass each group (correspond to samples coming from one specific dataset) into the forward path. Therefore, in this object, x is expected to be from the same dataset, thus with the same channel arrangement, making it possible to perform batch-wise channel rearrangement.
        """
        out: torch.Tensor = super().forward(data)
        return self._channel_mixer(out)

    @property
    def num_channels(self):
        return self._num_mix_out_channels


class Channel2D(torch.nn.Module):

    max_row = max([v.value for v in CHANNEL2D_ENUM])
    max_col = max([v.value for v in CHANNEL2D_ENUM])

    def __init__(self):
        super().__init__()

    def forward(self, data: dict) -> torch.Tensor:
        x: torch.Tensor = data["eeg"]
        metadata = data["meta"]
        y = torch.zeros(*x.shape[:-1], self.max_row, self.max_col)  # type: ignore
        # z-score normalization over batch
        x_mean = x.mean(dim=(1, 2), keepdim=True)
        x_std = x.std(dim=(1, 2), keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)
        rearrange_idx = []
        original_idx = []
        for c in metadata["channel_infos"].keys():
            chan_name = metadata["channel_infos"][c]["name"][0]
            if chan_name in CHANNEL2D_ENUM:
                rearrange_idx.append(CHANNEL2D_ENUM[chan_name].value)
                original_idx.append(c - 1)
        y[..., rearrange_idx] = x[..., original_idx]
        return y
