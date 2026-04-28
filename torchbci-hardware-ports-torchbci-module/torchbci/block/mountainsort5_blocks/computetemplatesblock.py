import torch

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class ComputeTemplatesBlock(MountainSort5Block):
    """
    Compute median templates per cluster.
    """

    def __init__(self):
        super().__init__()

    def run_block(self, batch):
        snippets = batch.snippets
        labels = batch.labels
        assert snippets is not None and labels is not None

        templates = _compute_templates(snippets, labels)
        batch.templates = templates

        K = templates.shape[0]
        if K > 0:
            min_per_channel = templates.min(dim=1).values
            batch.peak_channel_indices = torch.argmin(min_per_channel, dim=1)
        else:
            batch.peak_channel_indices = torch.zeros(0, dtype=torch.int64, device=snippets.device)

        return batch


def _compute_templates(
    snippets: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    L, T, M = snippets.shape

    if L == 0:
        return torch.zeros((0, T, M), dtype=torch.float32, device=snippets.device)

    K = int(labels.max().item())
    templates = torch.zeros((K, T, M), dtype=torch.float32, device=snippets.device)

    for k in range(1, K + 1):
        cluster = snippets[labels == k]
        if cluster.shape[0] == 0:
            templates[k - 1] = float("nan")
        else:
            templates[k - 1] = torch.quantile(cluster, 0.5, dim=0)

    return templates