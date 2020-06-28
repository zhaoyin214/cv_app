from abc import abstractmethod
from functools import partial
import torch.nn as nn


class ModelInterface(nn.Module):
    """Abstract class for models"""

    @abstractmethod
    def set_dropout_ratio(self, ratio):
        """Sets dropout ratio of the model"""

    @abstractmethod
    def get_input_res(self):
        """Returns input resolution"""
