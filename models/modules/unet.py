import torch
from torch import nn

from models.modules.Conv import GNSiLUConv2D
from models.modules.residual import ResidualBlock
from models.modules.sampler import DownSample, UpSample
from models.modules.embbeder import TimeEmbedding


class UNet(nn.Module):
    def __init__(self,
                 t_max: int,
                 init_channels: int,
                 base_channels: int,
                 channel_mults: list,
                 use_attention: list = None,
                 num_res_blocks=2,
                 num_groups=32,
                 dropout: int = 0.1,
                 time_emb_scale: float = 1.0,
                 use_t_emb: bool = True
                 ):
        super(UNet, self).__init__()
        use_attention = () if use_attention is None else tuple(use_attention)
        for a in use_attention:
            assert -1 < a < len(channel_mults)

        t_out_dim = None
        if use_t_emb:
            t_out_dim = 4 * base_channels
            self.time_embedding = TimeEmbedding(
                t_max=t_max,
                emb_dim=base_channels,
                out_dim=t_out_dim,
                scale=time_emb_scale
            )
        else:
            self.time_embedding = None

        self.head = nn.Conv2d(init_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=t_out_dim,
                    use_attention=i in use_attention
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(now_channels))
                channels.append(now_channels)

        self.mid = nn.ModuleList([
            ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=t_out_dim,
                    use_attention=True
            ),
            ResidualBlock(
                    now_channels,
                    now_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=t_out_dim,
                    use_attention=False
            )
        ])

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=t_out_dim,
                    use_attention=(len(channel_mults) - i - 1) in use_attention
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(UpSample(now_channels))

        assert len(channels) == 0

        self.tail = GNSiLUConv2D(now_channels, init_channels, kernel_size=3, padding=1, num_groups=num_groups)

    def forward(self, x, time=None):
        if self.time_embedding is not None:
            if time is None:
                raise ValueError("time conditioning was specified but tim is not passed")
            time_emb = self.time_embedding(time)
        else:
            time_emb = None

        x = self.head(x)

        skips = [x]
        for layer in self.downs:
            x = layer(x, time_emb) if isinstance(layer, ResidualBlock) else layer(x)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
                x = layer(x, time_emb)
            else:
                x = layer(x)

        x = self.tail(x)
        return x
