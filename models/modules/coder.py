from torch import nn

from models.modules.residual import ResidualBlock
from models.modules.sampler import DownSample, UpSample
from models.modules.swish import Swish


class Encoder(nn.Module):
    def __init__(self,
                 init_channels: int,
                 base_channels: int,
                 final_channels: int,
                 channel_mults: list,
                 use_attention: list = None,
                 num_res_blocks=2,
                 num_groups=32,
                 dropout: int = 0.1
                 ):
        super(Encoder, self).__init__()
        use_attention = () if use_attention is None else tuple(use_attention)
        for a in use_attention:
            assert -1 < a < len(channel_mults)

        self.head = nn.Conv2d(init_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        now_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = mult * base_channels

            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=None,
                    use_attention=i in use_attention
                ))
                now_channels = out_channels

            if i != len(channel_mults) - 1:
                self.downs.append(DownSample(now_channels))

        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                num_groups=num_groups,
                time_emb_dim=None,
                use_attention=True
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                num_groups=num_groups,
                time_emb_dim=None,
                use_attention=False
            )
        ])

        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups, now_channels),
            Swish(),
            nn.Conv2d(now_channels, 2 * final_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        for layer in self.downs:
            x = layer(x)
        for layer in self.mid:
            x = layer(x)
        x = self.tail(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 init_channels: int,
                 base_channels: int,
                 final_channels: int,
                 channel_mults: list,
                 use_attention: list = None,
                 num_res_blocks=2,
                 num_groups=32,
                 dropout: int = 0.1
                 ):
        super(Decoder, self).__init__()
        use_attention = () if use_attention is None else tuple(use_attention)
        for a in use_attention:
            assert -1 < a < len(channel_mults)

        now_channels = base_channels * channel_mults[-1]

        self.head = nn.Conv2d(final_channels, now_channels, kernel_size=3, padding=1)

        self.mid = nn.ModuleList([
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                num_groups=num_groups,
                time_emb_dim=None,
                use_attention=True
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                dropout,
                num_groups=num_groups,
                time_emb_dim=None,
                use_attention=False
            )
        ])

        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.ups.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    dropout,
                    num_groups=num_groups,
                    time_emb_dim=None,
                    use_attention=(len(channel_mults) - i - 1) in use_attention
                ))
                now_channels = out_channels

            if i != 0:
                self.ups.append(UpSample(now_channels))

        self.tail = nn.Sequential(
            nn.GroupNorm(num_groups, now_channels),
            Swish(),
            nn.Conv2d(now_channels, init_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        for layer in self.mid:
            x = layer(x)
        for layer in self.ups:
            x = layer(x)
        x = self.tail(x)
        return x
