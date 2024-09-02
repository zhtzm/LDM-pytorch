import torch
from torch import nn


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 norm,
                 act):
        super(ConvNormAct, self).__init__()
        self.norm_layer = norm
        self.act_layer = act
        self.conv_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.act_layer(self.norm_layer(x))
        return x


class NormActConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 norm: nn.Module,
                 act: nn.Module,):
        super(NormActConv, self).__init__()
        self.norm_layer = norm
        self.act_layer = act
        self.conv_layer = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_layer(self.norm_layer(x))
        x = self.conv_layer(x)
        return x
