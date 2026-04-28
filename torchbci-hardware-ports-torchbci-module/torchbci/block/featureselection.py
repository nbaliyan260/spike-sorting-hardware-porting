import torch
from .block import Block
from typing import List, Tuple


class FeatureSelection(Block):
    def __init__(self):
        super().__init__()



# TODO: peak-to-trough
# TODO: width
# TODO: etc.

class JimsFeatureSelection(Block):
    def __init__(self,
                    frame_size: int,
                    normalize: str,):
            super().__init__()
            self.frame_size = frame_size
            self.normalize = normalize

    def forward(self, x: torch.Tensor, peaks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract frames around detected peaks and normalize them.

        Args:
            x (torch.Tensor): Input tensor of shape ``(channels, time)``.
            peaks (torch.Tensor): Tensor of peak coordinates of shape ``(num_peaks, 2)``.

        Raises:
            ValueError: If the frame size is not odd.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the extracted frames and their metadata.
        """        
        if self.frame_size % 2 == 0:
            raise ValueError("Frame size must be odd.")
        half_size = self.frame_size // 2
        frame_starts = peaks[:, 1] - half_size # ()
        frame_ends = peaks[:, 1] + half_size + 1
        frame_chns = peaks[:, 0]

        frames = []
        frames_meta = []

        for chn, start, end in zip(frame_chns, frame_starts, frame_ends):
            if start < 0 or end > x.shape[1]:
                continue
            frame = x[chn, start:end]
            frames.append(frame)
            frames_meta.append([chn, start + half_size])
        frames = torch.stack(frames)
        frames_meta = torch.tensor(frames_meta)
        if self.normalize == "minmax":
            frames = (frames - frames.min(dim=1, keepdim=True)[0]) / (frames.max(dim=1, keepdim=True)[0] - frames.min(dim=1, keepdim=True)[0])
        elif self.normalize == "zscore":
            frames = (frames - frames.mean(dim=1, keepdim=True)) / frames.std(dim=1, keepdim=True)
        else:
            pass # no normalization
        return frames, frames_meta