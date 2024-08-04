import torch
from torch import nn
from torch.nn import functional as F


class DownSample(nn.Module):
    def __init__(self, in_channels: int):
        super(DownSample, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    @staticmethod
    def pool2d(x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'max':
            return F.max_pool2d(x, stride=2, kernel_size=2)
        elif mode == 'mean':
            return F.avg_pool2d(x, stride=2, kernel_size=2)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, mode: str = 'max') -> torch.Tensor:
        return self.conv(self.pool2d(x, mode))


class UpSample(nn.Module):
    def __init__(self, in_channels: int):
        super(UpSample, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode=mode))
