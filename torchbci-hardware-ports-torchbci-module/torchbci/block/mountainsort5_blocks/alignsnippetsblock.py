import torch

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class AlignSnippetsBlock(MountainSort5Block):
    """
    Roll snippets by cluster alignment offset and update times.
    """

    def __init__(self):
        super().__init__()

    def run_block(self, batch):
        snippets = batch.snippets
        times = batch.times
        labels = batch.labels
        offsets = batch.alignment_offsets
        assert snippets is not None and times is not None
        assert labels is not None and offsets is not None

        batch.snippets = _align_snippets(snippets, offsets, labels)
        batch.times = _offset_times(times, -offsets, labels)
        return batch


def _align_snippets(
    snippets: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if len(labels) == 0:
        return snippets

    aligned = torch.zeros_like(snippets)
    K = int(labels.max().item())

    for k in range(1, K + 1):
        inds = torch.where(labels == k)[0]
        if len(inds) > 0:
            aligned[inds] = torch.roll(snippets[inds], shifts=int(offsets[k - 1].item()), dims=1)

    return aligned


def _offset_times(
    times: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if len(labels) == 0:
        return times

    times2 = torch.zeros_like(times)
    K = int(labels.max().item())

    for k in range(1, K + 1):
        inds = torch.where(labels == k)[0]
        times2[inds] = times[inds] + offsets[k - 1]

    return times2