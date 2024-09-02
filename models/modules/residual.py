from typing import Union, Type

from torch import nn

from models.base.base_conv import NormActConv, ConvNormAct


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_type: Union[Type[NormActConv], Type[ConvNormAct]] = None,
            time_emb_dim: int = None,
            **kwargs
    ):
        super(ResidualBlock, self).__init__()

        self.block1 = block_type(in_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs)
        self.block2 = block_type(out_channels, out_channels, kernel_size=3, stride=1, padding=1, **kwargs)

        self.time_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        out = self.block1(x)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(time_emb)[:, :, None, None]

        out = self.block2(out) + self.residual_connection(x)

        return out
