from abc import ABC, abstractmethod

from models.base.base_model import BaseModel


class BaseVAE(BaseModel, ABC):
    def __init__(self):
        super(BaseVAE, self).__init__()

    @abstractmethod
    def encode(self, x):
        raise NotImplementedError

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError

    @abstractmethod
    def reparameterize(self, mean, log_var):
        raise NotImplementedError
