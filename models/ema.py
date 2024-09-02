from copy import deepcopy

import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, model, decay=0.999, bias_correction=False, num_updates=0, device=None, use_fp16=False):
        super(EMA, self).__init__()
        self.module = deepcopy(model).eval()
        self.decay = decay
        self.bias_correction = bias_correction
        self.num_updates = num_updates
        self.device = device
        self.use_fp16 = use_fp16

        if self.device is not None:
            self.module.to(self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(),
                                      model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(self.device)
                if self.use_fp16:
                    # choose whether to use mixed precision (FP16) to reduce memory usage
                    ema_v = ema_v.half()
                    model_v = model_v.half()

                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        if self.bias_correction:
            debias_term = 1 - self.decay ** (self.num_updates + 1)
            self._update(model, update_fn=lambda e, m: (self.decay * e + (1 - self.decay) * m) / debias_term)
            self.num_updates += 1
        else:
            self._update(model, update_fn=lambda e, m: self.decay * e + (1 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def save(self, path):
        torch.save(self.module.state_dict(), path)

    def load(self, path):
        self.module.load_state_dict(torch.load(path, map_location=self.device))

    def to(self, device):
        self.device = device
        self.module.to(device)
