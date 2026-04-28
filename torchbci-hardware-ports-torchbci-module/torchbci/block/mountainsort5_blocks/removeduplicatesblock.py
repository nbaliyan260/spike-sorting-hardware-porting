import torch

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class RemoveDuplicateTimesBlock(MountainSort5Block):
    """
    Remove duplicate spike times, keeping the first occurrence.
    """

    def __init__(self):
        super().__init__()

    def run_block(self, batch):
        times = batch.times
        channel_indices = batch.channel_indices
        assert times is not None and channel_indices is not None

        if len(times) == 0:
            return batch

        keep = torch.where(torch.diff(times) > 0)[0]
        keep = torch.cat([torch.tensor([0], device=times.device), keep + 1])

        batch.times = times[keep]
        batch.channel_indices = channel_indices[keep]
        return batch