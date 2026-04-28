from abc import ABC, abstractmethod
from torch import nn

class BaseBlock(nn.Module, ABC):
    """Abstract parent class for all reusable TorchBCI blocks."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass