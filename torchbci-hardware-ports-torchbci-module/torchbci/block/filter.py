import torch
from .block import Block
import torch.nn.functional as F

class Filter(Block):
    """Base class for all filter blocks.

    Args:
        Block (torch.nn.Module): Inherits from the Block class.
    """    
    def __init__(self):
        super().__init__()



class JimsFilter(Filter):
    """A simple smoothing (moving average) filter for preprocessing input signals.

    Args:
        window_size (int, optional): Size of the averaging window used for filtering.
            Defaults to 21.
    """

    def __init__(self, window_size: int = 21):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a 1D average pooling filter to smooth the input signal.

        Args:
            x (torch.Tensor): Input tensor of shape ``(channels, time)``.

        Returns:
            torch.Tensor: Filtered tensor of the same shape as input.
        """
        x = F.avg_pool1d(
            x.unsqueeze(0),
            kernel_size=self.window_size,
            stride=1,
            padding=self.window_size // 2,
            count_include_pad=False
        ).squeeze(0)
        return x
