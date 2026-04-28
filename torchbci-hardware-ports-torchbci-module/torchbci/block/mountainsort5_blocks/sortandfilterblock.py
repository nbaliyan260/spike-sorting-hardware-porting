import torch

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class SortTimesBlock(MountainSort5Block):
    def __init__(self):
        super().__init__()

    def run_block(self, batch):
        times = batch.times
        labels = batch.labels
        assert times is not None and labels is not None

        sort_inds = torch.argsort(times, stable=True)
        batch.times = times[sort_inds]
        batch.labels = labels[sort_inds]
        return batch


class RemoveOutOfBoundsBlock(MountainSort5Block):
    def __init__(self, params):
        super().__init__()
        self.T1 = params.snippet_T1
        self.T2 = params.snippet_T2

    def run_block(self, batch):
        times = batch.times
        labels = batch.labels
        assert times is not None and labels is not None and batch.traces is not None

        N = batch.num_timepoints
        valid = (times >= self.T1) & (times < N - self.T2)
        batch.times = times[valid]
        batch.labels = labels[valid]
        return batch


class ReorderUnitsBlock(MountainSort5Block):
    def __init__(self):
        super().__init__()

    def run_block(self, batch):
        labels = batch.labels
        peak_channels = batch.peak_channel_indices
        assert labels is not None and peak_channels is not None

        K = int(labels.max().item()) if len(labels) > 0 else 0
        if K == 0:
            return batch

        aa = peak_channels.to(torch.float32).clone()
        for k in range(1, K + 1):
            if (labels == k).sum() == 0:
                aa[k - 1] = float("inf")

        new_labels_mapping = torch.argsort(torch.argsort(aa, stable=True), stable=True) + 1
        batch.labels = new_labels_mapping[labels - 1]
        return batch