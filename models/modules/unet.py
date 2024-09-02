from typing import Union, Type

import torch
from torch import nn

from models.base.base_conv import NormActConv, ConvNormAct
from models.modules.attn import MultiHeadAttention, ScaledDotAttention
from models.modules.embbeder import TimeEmbedding
from models.modules.residual import ResidualBlock
from models.modules.sampler import DownSample, UpSample
from utils.initialization_utils import import_class_from_string


class UNet(nn.Module):
    def __init__(self,
                 t_max: int,
                 init_channels: int,
                 base_channels: int,
                 channel_mults: list,
                 block_type: Union[Type[NormActConv], Type[ConvNormAct], str],
                 dropout_rate: float = 0.0,
                 num_res_blocks=2,
                 time_emb_scale: float = 1.0,
                 n_heads: int = 8,
                 use_t_emb: bool = True,
                 attn_level: list = None,
                 conv_params: dict = None,
                 ):
        super(UNet, self).__init__()
        if attn_level is None:
            attn_level = []
        if conv_params is None:
            conv_params = {}
        if isinstance(block_type, str):
            block_type, _ = import_class_from_string(block_type)

        self.time_embedding, t_out_dim = self.init_time_embedding(use_t_emb, base_channels, t_max, time_emb_scale)

        self.head = nn.Conv2d(init_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    block_type=block_type,
                    time_emb_dim=t_out_dim,
                    **conv_params
                ))
                if i in attn_level:
                    self.downs.append(MultiHeadAttention(
                        channels=out_channels,
                        num_heads=n_heads,
                        dropout=dropout_rate
                    ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(now_channels))
                channels.append(now_channels)

        self.mid = nn.Sequential(
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **conv_params
            ),
            MultiHeadAttention(
                channels=now_channels,
                num_heads=n_heads,
                dropout=dropout_rate,
            ),
            ResidualBlock(
                in_channels=now_channels,
                out_channels=now_channels,
                block_type=block_type,
                time_emb_dim=None,
                **conv_params
            )
        )

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    block_type=block_type,
                    time_emb_dim=t_out_dim,
                    **conv_params
                ))
                if i in attn_level:
                    self.ups.append(MultiHeadAttention(
                        channels=out_channels,
                        num_heads=n_heads,
                        dropout=dropout_rate
                    ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(UpSample(now_channels))

        assert len(channels) == 0

        self.tail = block_type(now_channels, init_channels, kernel_size=3, stride=1, padding=1, **conv_params)

    def forward(self, x: torch.Tensor, time: int = None) -> torch.Tensor:
        if self.time_embedding is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            time_emb = self.time_embedding(time)
        else:
            time_emb = None

        x = self.head(x)

        skips = [x]
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
            if not isinstance(layer, ScaledDotAttention):
                skips.append(x)

        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, time_emb)
            else:
                x = layer(x)

        x = self.tail(x)
        return x

    @staticmethod
    def init_time_embedding(use_t_emb, base_channels, t_max, time_emb_scale):
        if use_t_emb:
            t_out_dim = 4 * base_channels
            time_embedding = TimeEmbedding(
                t_max=t_max,
                emb_dim=base_channels,
                out_dim=t_out_dim,
                scale=time_emb_scale
            )
        else:
            return None, None

        return time_embedding, t_out_dim
