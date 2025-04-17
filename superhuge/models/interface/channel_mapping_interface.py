from .m_interface import MInterface
from ..tools.pre_model import Channel1D, Channel1DMixer, Channel2D

import lightning as pl2


class ChannelMapping1DInterface(MInterface):
    def __init__(self, /, **kwargs):
        super().__init__(**kwargs)
        self.pre_model = Channel1D()


class ChannelMixing1DInterface(MInterface):
    def __init__(self, /, *, num_mix_out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.pre_model = Channel1DMixer(num_mix_out_channels)


class ChannelMapping2DInterface:
    def __init__(self, *args, **kwargs):
        self.pre_model = Channel2D()
        # super().__init__(*args, **kwargs)
