from .m_interface import MInterface
from .pre_model import Channel1D, Channel2D


class ChannelMapping1DInterface(MInterface):
    pre_model = Channel1D()


class ChannelMapping2DInterface(MInterface):
    pre_model = Channel2D()
