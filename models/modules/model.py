import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, class_name:str = None):
        super(Model, self).__init__()
        self.class_name = class_name

    def forward(self, x: torch.Tensor):
        pass

    def loss_function(self, **kwargs):
        pass

    def loss_forward(self, x: torch.Tensor):
        pass
