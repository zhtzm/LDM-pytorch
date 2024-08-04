import torch
from torch import nn

from models.modules.coder import Encoder, Decoder


class VAE(nn.Module):
    def __init__(self,
                 coder_config: dict,
                 embed_dim: int
                 ):
        super(VAE, self).__init__()
        self.encoder = Encoder(**coder_config)
        self.decoder = Decoder(**coder_config)

        self.quant_conv_mean = nn.Conv2d(2 * coder_config["final_channels"], embed_dim, kernel_size=1)
        self.quant_conv_logvar = nn.Conv2d(2 * coder_config["final_channels"], embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, coder_config["final_channels"], kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        mean = self.quant_conv_mean(h)
        log_var = self.quant_conv_logvar(h)
        return mean, log_var

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    @staticmethod
    def reparameterization(mean, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(self, x: torch.Tensor):
        mean, log_var = self.encode(x)

        z = self.reparameterization(mean, log_var)

        recon = self.decode(z)
        return recon, mean, log_var

    @staticmethod
    def loss_function(x: torch.Tensor,
                      recon: torch.Tensor,
                      mean: torch.Tensor,
                      log_var: torch.Tensor,
                      kl_weight: float = 1.0
                      ) -> torch.Tensor:
        recon_loss = torch.nn.functional.l1_loss(recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp())
        loss = recon_loss + kl_loss * kl_weight
        return loss

    def loss_forward(self, x: torch.Tensor):
        recon, mean, log_var = self.forward(x)
        loss = self.loss_function(x, recon, mean, log_var)[0]
        return loss
