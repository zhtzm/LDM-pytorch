import math

import torch
from torch import nn
from torch.nn import init


class TimeEmbedding(nn.Module):
    def __init__(self,
                 t_max: int,
                 emb_dim: int,
                 out_dim: int,
                 scale: float):
        super(TimeEmbedding, self).__init__()
        pre = self.init_t_emb(t_max, emb_dim, scale)

        self.module_list = nn.Sequential(
            nn.Embedding.from_pretrained(pre),
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

        self.initialize()

    @staticmethod
    def init_t_emb(t_max: int, emb_dim: int, scale: float) -> torch.Tensor:
        """
        Sine and cosine trigonometric position encoding
        :param scale: Scale of the position encoding
        :param t_max: Specify the maximum length
        :param emb_dim: Specify the embedding dimension
        :return: position embedding
        """
        assert emb_dim % 2 == 0, "emb_dim must be an even number"
        emb = torch.arange(0, emb_dim, step=2) / emb_dim * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(t_max).float() * scale
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [t_max, emb_dim // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [t_max, emb_dim // 2, 2]
        emb = emb.view(t_max, emb_dim)

        return emb

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t: int) -> torch.Tensor:
        return self.module_list(t)
