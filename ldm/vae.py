import torch
from torch import nn

from ldm.modules.coder import Encoder, Decoder
from ldm.modules.distribution import DiagonalGaussianDistribution


class VAE(nn.Module):
    def __init__(self,
                 coder_config: dict,
                 embed_dim: int
                 ):
        super(VAE, self).__init__()
        self.encoder = Encoder(**coder_config)
        self.decoder = Decoder(**coder_config)

        self.quant_conv = nn.Conv2d(2 * coder_config["final_channels"], 2 * embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(embed_dim, coder_config["final_channels"], kernel_size=1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor, sample_posterior=True):
        posterior = self.encode(x)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)
        return dec, posterior
