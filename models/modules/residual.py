from torch import nn

from models.modules.Conv import GNSiLUConv2D
from models.modules.attn import AttentionBlock


class ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            time_emb_dim: int = None,
            num_groups: int = 32,
            use_attention: bool = False,
    ):
        super(ResidualBlock, self).__init__()

        self.block1 = GNSiLUConv2D(in_channels, out_channels, kernel_size=3, padding=1, num_groups=num_groups)
        self.block2 = GNSiLUConv2D(out_channels, out_channels, kernel_size=3, padding=1, num_groups=num_groups)

        self.time_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, dropout, num_groups)

    def forward(self, x, time_emb=None):
        out = self.block1(x)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(time_emb)[:, :, None, None]

        out = self.block2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out
