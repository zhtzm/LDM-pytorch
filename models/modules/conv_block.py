from torch import nn

from models.base.base_conv import NormActConv, ConvNormAct


class GNSiLUConv2d(NormActConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 num_groups=32
                 ):
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        act = nn.SiLU()

        super(GNSiLUConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)


class Conv2dGNSiLU(ConvNormAct):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 num_groups=32
                 ):
        norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        act = nn.SiLU()

        super(Conv2dGNSiLU, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)


class BNReLUConv2d(NormActConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0
                 ):
        norm = nn.BatchNorm2d(in_channels)
        act = nn.ReLU6()

        super(BNReLUConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)


class Conv2dBNReLU(ConvNormAct):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0
                 ):
        norm = nn.BatchNorm2d(out_channels)
        act = nn.ReLU6()

        super(Conv2dBNReLU, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)


class INReLUConv2d(NormActConv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        norm = nn.InstanceNorm2d(in_channels)
        act = nn.ReLU6()

        super(INReLUConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)


class Conv2dINReLU(ConvNormAct):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        norm = nn.InstanceNorm2d(out_channels)
        act = nn.ReLU6()

        super(Conv2dINReLU, self).__init__(in_channels, out_channels, kernel_size, stride, padding, norm, act)
