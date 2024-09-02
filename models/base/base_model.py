from abc import ABC, abstractmethod

from torch import nn


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss_function(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def loss_forward(self, x) -> dict:
        raise NotImplementedError
