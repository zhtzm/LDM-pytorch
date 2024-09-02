import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 dropout: float = 0.0,
                 scale: float = None):
        super(ScaledDotAttention, self).__init__()

        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.proj_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout_layer = nn.Dropout(dropout)

        self.scale = scale if scale is not None else (channels ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Normalization and Q, K, V projections in one step
        h = self.norm(x)
        qkv = self.proj_qkv(h)
        q, k, v = qkv.chunk(3, dim=1)  # Split the projections

        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = k.reshape(B, C, H * W)  # (B, C, HW)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)

        # Attention mechanism using the scale variable
        attn = torch.bmm(q, k) * self.scale  # (B, HW, HW)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        h = torch.bmm(attn, v)  # (B, HW, C)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)

        # Final projection
        h = self.proj(h)

        return x + h


class MultiHeadAttention(ScaledDotAttention):
    def __init__(self,
                 channels: int,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 scale: float = None):
        super(MultiHeadAttention, self).__init__(channels, dropout, scale)
        assert channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Normalization and Q, K, V projections
        h = self.norm(x)
        qkv = self.proj_qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, num_heads, HW, head_dim)

        # Scaled dot-product attention for each head
        attn = torch.einsum('bnqd,bnkd->bnqk', q, k) * self.scale  # (B, num_heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout_layer(attn)

        h = torch.einsum('bnqk,bnvd->bnqd', attn, v)  # (B, num_heads, HW, head_dim)
        h = h.permute(0, 2, 3, 1).reshape(B, C, H, W)  # Concatenate heads

        # Final projection
        h = self.proj(h)

        return x + h