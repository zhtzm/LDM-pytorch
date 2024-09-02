from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from models.base.base_model import BaseModel
from models.modules.unet import UNet


class DDPM(BaseModel):
    def __init__(self,
                 beta: list,
                 t_max: int,
                 channel: int,
                 size: int | list,
                 clip_denoised: bool,
                 sample_img_every_t: int,
                 loss_type: str,
                 unet_config: dict):
        assert len(beta) == 2

        super(DDPM, self).__init__()

        self.t_max = t_max
        self.register_schedule(beta)
        self.channel = channel
        self.loss_type = loss_type
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.clip_denoised = clip_denoised
        self.sample_img_every_t = sample_img_every_t
        self.wrapper = DiffusionWrapper(unet_config, None)

    def register_schedule(self, beta: list):
        start, end = beta if beta[0] < beta[1] else (beta[1], beta[0])

        betas = np.linspace(start, end, num=self.t_max)
        alphas = 1. - betas
        cumprod_alphas = np.cumprod(alphas)
        # cumprod_alphas_prev is used to be cumprod_alpha_t_minus_one given t
        cumprod_alphas_prev = np.append(1., cumprod_alphas[:-1])
        # ensure the size of cumprod_alphas is equal to cumprod_alphas_prev
        assert len(cumprod_alphas) == len(cumprod_alphas_prev)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("cumprod_alphas", to_torch(cumprod_alphas))
        self.register_buffer('cumprod_alphas_prev', to_torch(cumprod_alphas_prev))

        # the m1 in the symbol below represents the subtraction of one

        # sqrt_cumprod_alphas and sqrt_one_minus_cumprod_alphas are used to compute x_t
        # x_t = sqrt_cumprod_alphas[t] * x_0 + sqrt_one_minus_cumprod_alphas[t] * epsilon(a sample of N(0, I))
        self.register_buffer("sqrt_cumprod_alphas", to_torch(np.sqrt(cumprod_alphas)))
        self.register_buffer("sqrt_one_minus_cumprod_alphas", to_torch(np.sqrt(1. - cumprod_alphas)))
        # above is diffusion process

        # below is used for denoising process
        # sqrt_recip_cumprod_alphas and sqrt_recip_cumprod_alphas_m1 are used to compute x_0 given x_t
        # x_0 = sqrt_recip_cumprod_alphas[t] * x_t + sqrt_recip_cumprod_alphas_m1[t] * epsilon(training by unet)
        self.register_buffer('sqrt_recip_cumprod_alphas', to_torch(np.sqrt(1. / cumprod_alphas)))
        self.register_buffer('sqrt_recip_cumprod_alphas_m1', to_torch(np.sqrt(1. / cumprod_alphas - 1)))
        # posterior_variance in calculating the square of the variance
        posterior_variance = betas * (1. - cumprod_alphas_prev) / (1. - cumprod_alphas)
        self.register_buffer("posterior_variance", to_torch(
            posterior_variance
        ))
        # convert to log format for easy calculation and avoid being all positive, here I have set up the Nether
        self.register_buffer("posterior_log_variance_clipped", to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))
        ))
        # mean = posterior_mean_n0[t] * x_0 + posterior_mean_nt[t] * x_t
        # the parameter used to calculate the mean
        self.register_buffer('posterior_mean_n0', to_torch(
            betas * np.sqrt(cumprod_alphas_prev) / (1. - cumprod_alphas)
        ))
        # the parameter used to calculate the mean
        self.register_buffer('posterior_mean_nt', to_torch(
            (1. - cumprod_alphas_prev) * np.sqrt(alphas) / (1. - cumprod_alphas)
        ))

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
        assert t.dim() == 1 and t.shape[0] == x_shape[0]
        return a.gather(-1, t).reshape(x_shape[0], *((1,) * (len(x_shape) - 1)))

    def diffusion(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (self._extract(self.sqrt_cumprod_alphas, t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_cumprod_alphas, t, x_start.shape) * noise)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (self._extract(self.sqrt_recip_cumprod_alphas, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recip_cumprod_alphas_m1, t, x_t.shape) * noise)

    def loss_function(self, noise: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        if self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise, reduction='sum')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise, reduction='sum')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.shape[-3:] == (self.channel, *self.size)
        device = self.betas.device
        x = x.to(device) if not x.device == device else x
        t = torch.randint(self.t_max, size=(x.shape[0],), device=x.device)

        noise = torch.randn_like(x)
        x_t = self.diffusion(x, t, noise)
        predicted_noise = self.wrapper(x_t, t)

        return noise, predicted_noise

    def loss_forward(self, x: torch.Tensor) -> torch.Tensor:
        noise, predicted_noise = self.forward(x)
        loss = self.loss_function(noise, predicted_noise)
        return {
            'loss': loss
        }

    @torch.no_grad()
    def denoising(self, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        b, *_, device = *x_t.shape, x_t.device

        x_0 = self.predict_x0(x_t, t, self.wrapper(x_t, t))
        if clip_denoised:
            x_0.clamp_(-1., 1.)

        posterior_mean = (
                self._extract(self.posterior_mean_n0, t, x_t.shape) * x_0 +
                self._extract(self.posterior_mean_nt, t, x_t.shape) * x_t
        )
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        posterior_standard_deviation = (0.5 * posterior_log_variance).exp()

        noise = torch.randn_like(x_t, device=device)
        nonzero_mask = torch.Tensor(t != 0).reshape(b, 1, 1, 1)
        return posterior_mean + nonzero_mask * posterior_standard_deviation * noise

    @torch.no_grad()
    def denoising_loop(self, shape: tuple, return_intermediates: bool):
        device = self.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device)

        intermediates = {self.t_max: img} if return_intermediates else None
        for i in tqdm(reversed(range(0, self.t_max)), desc='Sampling t', total=self.t_max):
            t_batch = torch.tensor(i, dtype=int, device=device).repeat(batch_size)
            img = self.denoising(img, t_batch, clip_denoised=self.clip_denoised)
            if i % self.sample_img_every_t == 0 and return_intermediates:
                intermediates[i] = img

        return img, intermediates

    @torch.no_grad()
    def sample(self, batch_size: int, return_intermediates: bool = False):
        shape = (batch_size, self.channel, *self.size)
        return self.denoising_loop(shape, return_intermediates)


class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config, conditioning_key):
        super(DiffusionWrapper, self).__init__()
        self.unet = UNet(**unet_config)
        if conditioning_key == 'None':
            self.conditioning_key = None
        else:
            self.conditioning_key = conditioning_key
        self.assert_conditioning_key()

    def assert_conditioning_key(self):
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid'], \
            "Conditioning key must be either 'concat', 'crossattn' or 'hybrid'"

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.unet(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.unet(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.unet(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.unet(xc, t, context=cc)
        else:
            raise NotImplementedError()

        return out
