from .m_interface import MInterface
from .pre_model import Channel1D, Channel2D

import lightning as pl2


class ChannelMapping1DInterface(MInterface):
    def __init__(self, /, *, num_electrodes: int, num_mix_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.pre_model = Channel1D(num_electrodes, num_mix_channels)


class ChannelMapping2DInterface:
    def __init__(self, *args, **kwargs):
        self.pre_model = Channel2D()
        # super().__init__(*args, **kwargs)
