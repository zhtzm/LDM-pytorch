from torch import nn
from torch.nn import init

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

        self.norm_1 = nn.GroupNorm(num_groups, in_channels)
        self.act_1 = nn.SiLU()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm_2 = nn.GroupNorm(num_groups, out_channels)
        self.act_2 = nn.SiLU()
        self.conv_2 = nn.Sequential(
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_bias = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.residual_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()
        self.attention = nn.Identity() if not use_attention else AttentionBlock(out_channels, dropout, num_groups)

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x, time_emb=None):
        out = self.act_1(self.norm_1(x))
        out = self.conv_1(out)

        if self.time_bias is not None:
            if time_emb is None:
                raise ValueError("time conditioning was specified but time_emb is not passed")
            out += self.time_bias(time_emb)[:, :, None, None]

        out = self.act_2(self.norm_2(out))
        out = self.conv_2(out) + self.residual_connection(x)
        out = self.attention(out)

        return out
