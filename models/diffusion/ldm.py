import torch
from tqdm import tqdm

from models.diffusion.ddpm import DDPM
from utils.initialization_utils import import_class_from_string
from utils.model_data_utils import disable_train_mode
from utils.train_test_util import load_checkpoint


class LDM(DDPM):
    def __init__(self,
                 ae_model: dict,
                 condition_model: dict,
                 ddpm_config: dict,
                 conditioning_key: str
                 ):
        super(LDM, self).__init__(**ddpm_config)
        assert ae_model is not None
        self.ae_model = self.initialize_ae_model(ae_model)
        self.condition_model = self.initialize_condition_model(condition_model)

        self.wrapper.conditioning_key = conditioning_key
        self.wrapper.assert_conditioning_key()

    @staticmethod
    def initialize_condition_model(model_cfg):
        if model_cfg['target']:
            condition_model_class, _ = import_class_from_string(model_cfg['target'])
            condition_model = condition_model_class(**model_cfg['params'])
            if model_cfg['ckpt_file']:
                condition_model, _ = load_checkpoint(model_cfg['ckpt_file'], condition_model)
                return disable_train_mode(condition_model)
            return condition_model
        return None

    @staticmethod
    def initialize_ae_model(model_cfg):
        if model_cfg['target']:
            ae_model_class, _ = import_class_from_string(model_cfg['target'])
            ae_model = ae_model_class(**model_cfg['params'])
            if model_cfg['ckpt_file']:
                ae_model, _ = load_checkpoint(model_cfg['ckpt_file'], ae_model)
                return disable_train_mode(ae_model)
            return ae_model
        return None

    def forward(self, x: torch.Tensor, condition: list = None):
        assert x.dim() == 4
        mean, log_var = self.ae_model.encode(x)

        x_enc = self.ae_model.reparameterize(mean, log_var)
        assert x_enc.shape[-3:] == (self.channel, *self.size)
        if self.wrapper.conditioning_key is not None:
            assert condition is not None
            assert self.condition_model is not None
            condition = self.condition_model(condition)
        t = torch.randint(self.t_max, size=(x_enc.shape[0],), device=x_enc.device)

        noise = torch.randn_like(x_enc)
        x_t = self.diffusion(x_enc, t, noise)
        predicted_noise = self.wrapper(x_t, t, condition)

        return noise, predicted_noise

    def loss_forward(self, x: torch.Tensor, condition: list = None) -> torch.Tensor:
        noise, predicted_noise = self.forward(x, condition)
        loss = self.loss_function(noise, predicted_noise)
        return {
            'loss': loss
        }

    def denoising(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, clip_denoised=True):
        b, *_, device = *x_t.shape, x_t.device

        x_0 = self.predict_x0(x_t, t, self.wrapper(x_t, t, condition))
        if clip_denoised:
            x_0.clamp_(-1., 1.)

        posterior_mean = (
                self._extract(self.posterior_mean_n0, t, x_t.shape) * x_0 +
                self._extract(self.posterior_mean_nt, t, x_t.shape) * x_t
        )
        posterior_log_variance_v2 = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        posterior_variance = (0.5 * posterior_log_variance_v2).exp()

        noise = torch.randn_like(x_t, device=device)
        nonzero_mask = torch.Tensor(t != 0).reshape(b, 1, 1, 1)
        return posterior_mean + nonzero_mask * posterior_variance * noise

    @torch.no_grad()
    def denoising_loop(self, shape: tuple, condition: torch.Tensor, return_intermediates: bool = False):
        device = self.betas.device
        batch_size = shape[0]
        img = torch.randn(shape, device=device)

        intermediates = {self.t_max: img} if return_intermediates else None
        for i in tqdm(reversed(range(0, self.t_max)), desc='Sampling t', total=self.t_max):
            t_batch = torch.tensor(i, device=device).repeat(batch_size)
            img = self.denoising(img, t_batch, condition, clip_denoised=self.clip_denoised)
            if i % self.sample_img_every_t == 0 and return_intermediates:
                intermediates[i] = img

        return img, intermediates

    @torch.no_grad()
    def sample(self, batch_size: int, return_intermediates: bool, condition: list = None) -> torch.Tensor:
        shape = (batch_size, self.channel, *self.size)
        if self.wrapper.conditioning_key is not None:
            assert condition is not None
            assert self.condition_model is not None
            condition = self.condition_model(condition)
        latent_img, latent_intermediates = self.denoising_loop(shape, condition, return_intermediates)
        img = self.ae_model.decode(latent_img)
        intermediates = {t: self.ae_model.decode(img) for t, img in latent_intermediates.items()} \
            if return_intermediates else None
        return img, intermediates
