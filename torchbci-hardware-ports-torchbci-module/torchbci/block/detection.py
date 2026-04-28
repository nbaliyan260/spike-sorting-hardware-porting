import torch
from .block import Block
from typing import List, Tuple
import torch.nn.functional as F



class Detection(Block):
    """Base class for all detection blocks.

    Args:
        Block (torch.nn.Module): Inherits from the Block class.
    """    
    def __init__(self):
        super().__init__()



class JimsDetection(Detection):
    """Peak detection module for identifying significant activations
    in preprocessed signals.

    Args:
        threshold (int, optional): Amplitude threshold used to identify peaks.
            Defaults to 50.
    """

    def __init__(self, threshold: int = 50):
        super().__init__()
        self.threshold = threshold

    def thresholding(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out values below the detection threshold.

        Args:
            x (torch.Tensor): Input tensor of shape ``(channels, time)``.

        Returns:
            torch.Tensor: Tensor where all values below the threshold are set to zero.
        """
        x = x - self.threshold
        x[x < 0] = 0
        return x
  
    def find_peaks(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find row-wise peaks in a 2D tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing a tensor of peak coordinates and a tensor of peak values.
        """
        active = (x != 0)
        peaks = []
        peak_vals = []
        for chn in range(active.shape[0]):
            active_chn = active[chn]
            diff = active_chn[1:].int() - active_chn[:-1].int() 
            starts = torch.nonzero(diff == 1, as_tuple=True)[0] + 1
            ends = torch.nonzero(diff == -1, as_tuple=True)[0] + 1
            if active_chn[0]:
                starts = torch.cat((torch.tensor([0]), starts))
            if active_chn[-1]:
                ends = torch.cat((ends, torch.tensor([active_chn.shape[0]])))
            for start, end in zip(starts, ends):
                if (end - start) < 5:
                    continue # skip very short peaks, likely noise
                region = x[chn, start:end]
                max_val = region.max().item()
                max_ind = region.argmax() + start
                peaks.append([chn, max_ind.item()])
                peak_vals.append(max_val)
        peaks = torch.tensor(peaks)
        peak_vals = torch.tensor(peak_vals)
        return peaks, peak_vals
    
    # TODO: Confirm this replicates the original forward function
    def forward(self, x):
        detect = self.thresholding(x)
        return self.find_peaks(detect)