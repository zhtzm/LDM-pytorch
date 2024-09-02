import torch
from torch import nn


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super(DownSample, self).__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super(UpSample, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_transpose(x)
