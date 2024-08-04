from torch import nn


class GNSiLUConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 num_groups=32,
                 inplace=False
                 ):
        super(GNSiLUConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups
        self.inplace = inplace

        self.module = nn.Sequential(
            nn.GroupNorm(num_groups=self.num_groups, num_channels=self.in_channels),
            nn.SiLU(inplace=self.inplace),
            nn.Conv2d(self.in_channels,
                      self.out_channels,
                      self.kernel_size,
                      self.stride,
                      self.padding,
                      bias=False)
        )

    def forward(self, x):
        return self.module(x)
