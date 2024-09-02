from abc import ABC, abstractmethod

from models.base.base_model import BaseModel


class BaseDDPM(BaseModel, ABC):
    def __init__(self):
        super(BaseDDPM, self).__init__()

