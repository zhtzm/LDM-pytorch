import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dropout: float = 0.0,
                 num_groups: int = 32):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.dropout = dropout
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.proj_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)

        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        attn = torch.bmm(q, k) * (C ** -0.5)  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)  # 归一化

        attn = self.dropout_layer(attn)

        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h = torch.bmm(attn, v)  # (B, HW, C)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        h = self.proj(h)

        return x + h
