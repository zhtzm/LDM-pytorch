import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dropout: float,
                 num_groups: int = 32):
        super(AttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.dropout = dropout
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.proj_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)
        w = F.dropout(w, p=self.dropout, training=self.training)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 dropout: float,
                 n_heads: int,
                 num_groups: int = 32):
        super(MultiHeadAttentionBlock, self).__init__()

        self.in_channels = in_channels
        self.dropout = dropout
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.proj_q = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_k = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj_v = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)

        assert C % self.n_heads == 0, "channel dimension must be divisible by n_heads"
        cdh = int(C / self.n_heads)
        scale = 1 / torch.sqrt(cdh)

        h = self.norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.view(B, self.n_heads, -1, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.n_heads, -1, H * W)
        w = torch.matmul(q, k) * scale
        assert list(w.shape) == [B, self.n_heads, H * W, H * W]
        w = F.softmax(w, dim=-1)
        w = F.dropout(w, p=self.dropout, training=self.training)

        v = v.view(B, self.n_heads, -1, H * W).permute(0, 1, 3, 2)
        h = torch.matmul(w, v)
        assert list(h.shape) == [B, self.n_heads, H * W, cdh]
        h = h.permute(0, 1, 3, 2).view(B, C, H * W)
        h = self.proj(h)

        return (x + h).view(B, C, H, W)
