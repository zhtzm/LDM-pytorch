import numpy as np
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self,
                 t_max: int,
                 emb_dim: int,
                 out_dim: int,
                 scale: float):
        super(TimeEmbedding, self).__init__()
        self.register_buffer('t_emb', self.init_t_emb(t_max, emb_dim, scale))

        self.module_list = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    @staticmethod
    def init_t_emb(t_max: int, emb_dim: int, scale: float) -> torch.Tensor:
        """
        Sine and cosine trigonometric position encoding using NumPy for efficiency.
        :param scale: Scale of the position encoding
        :param t_max: Specify the maximum length
        :param emb_dim: Specify the embedding dimension
        :return: position embedding as a torch.Tensor
        """
        assert emb_dim % 2 == 0, "emb_dim must be an even number"

        emb = np.arange(0, emb_dim, step=2) / emb_dim * np.log(10000)
        emb = np.exp(-emb)
        pos = np.arange(t_max).astype(np.float32) * scale
        emb = pos[:, None] * emb[None, :]
        assert emb.shape == (t_max, emb_dim // 2)

        sin_emb = np.sin(emb)
        cos_emb = np.cos(emb)
        emb = np.stack([sin_emb, cos_emb], axis=-1)
        assert emb.shape == (t_max, emb_dim // 2, 2)
        emb = emb.reshape(t_max, emb_dim).astype(np.float32)

        return torch.from_numpy(emb)

    def forward(self, t: int) -> torch.Tensor:
        return self.module_list(self.t_emb[t])
