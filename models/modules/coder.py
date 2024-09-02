from typing import Type, Union

import torch
from torch import nn

from models.base.base_conv import ConvNormAct, NormActConv
from models.modules.attn import ScaledDotAttention
from models.modules.residual import ResidualBlock
from models.modules.sampler import DownSample, UpSample
from utils.initialization_utils import import_class_from_string


class Encoder(nn.Module):
    def __init__(self,
                 init_channels: int,
                 base_channels: int,
                 final_channels: int,
                 channel_mults: list,
                 block_type: Union[Type[NormActConv], Type[ConvNormAct], str],
                 dropout_rate: float,
                 num_res_blocks: int = 2,
                 **kwargs
                 ):
        super(Encoder, self).__init__()
        if isinstance(block_type, str):
            block_type, _ = import_class_from_string(block_type)

        self.head = nn.Conv2d(init_channels, base_channels, kernel_size=3, stride=1, padding=1)

        layers = []
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = mult * base_channels

            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(
                    in_channels=now_channels,
                    out_channels=out_channels,
                    block_type=block_type,
                    time_emb_dim=None,
                    **kwargs
                ))
                now_channels = out_channels

            if i != len(channel_mults) - 1:
                layers.append(DownSample(now_channels))

        self.downs = nn.Sequential(*layers)

        self.mid = nn.Sequential(
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **kwargs
            ),
            ScaledDotAttention(
                channels=now_channels,
                dropout=dropout_rate,
            ),
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **kwargs
            )
        )

        self.tail = block_type(now_channels, 2 * final_channels, kernel_size=3, stride=1, padding=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.downs(x)
        x = self.mid(x)
        x = self.tail(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 init_channels: int,
                 base_channels: int,
                 final_channels: int,
                 channel_mults: list,
                 block_type: Union[Type[NormActConv], Type[ConvNormAct], str],
                 dropout_rate: float,
                 num_res_blocks: int = 2,
                 **kwargs
                 ):
        super(Decoder, self).__init__()
        if isinstance(block_type, str):
            block_type, _ = import_class_from_string(block_type)

        now_channels = base_channels * channel_mults[-1]

        self.head = nn.Conv2d(final_channels, now_channels, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **kwargs
            ),
            ScaledDotAttention(
                channels=now_channels,
                dropout=dropout_rate,
            ),
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **kwargs
            )
        )

        layers = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                layers.append(ResidualBlock(
                    in_channels=now_channels,
                    out_channels=out_channels,
                    block_type=block_type,
                    time_emb_dim=None,
                    **kwargs
                ))
                now_channels = out_channels

            if i != 0:
                layers.append(UpSample(now_channels))

        self.ups = nn.Sequential(*layers)

        self.tail = block_type(now_channels, init_channels, kernel_size=3, stride=1, padding=1, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.mid(x)
        x = self.ups(x)
        x = self.tail(x)
        return x
