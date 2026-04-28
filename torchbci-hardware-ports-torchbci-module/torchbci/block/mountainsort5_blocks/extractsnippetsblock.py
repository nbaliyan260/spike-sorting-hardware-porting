from typing import Optional

import torch

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class ExtractSnippetsBlock(MountainSort5Block):
    """
    Extract spike snippets from traces.
    """

    def __init__(self, params):
        super().__init__()
        self.T1 = params.snippet_T1
        self.T2 = params.snippet_T2
        self.mask_radius = params.snippet_mask_radius

    def run_block(self, batch):
        traces = batch.traces
        times = batch.times
        channel_indices = batch.channel_indices
        channel_locations = batch.channel_locations
        assert traces is not None and times is not None

        batch.snippets = self._extract(traces, times, channel_indices, channel_locations)
        return batch

    def _extract(
        self,
        traces: torch.Tensor,
        times: torch.Tensor,
        channel_indices: Optional[torch.Tensor],
        channel_locations: Optional[torch.Tensor],
    ) -> torch.Tensor:
        M = traces.shape[1]
        L = times.shape[0]
        device = traces.device

        if L == 0:
            return torch.zeros((0, self.T1 + self.T2, M), dtype=traces.dtype, device=device)

        window = torch.arange(-self.T1, self.T2, device=device)
        extract_indices = times.unsqueeze(1) + window.unsqueeze(0)
        snippets = traces[extract_indices]

        if self.mask_radius is not None and channel_indices is not None and channel_locations is not None:
            dists = torch.cdist(channel_locations, channel_locations)
            adj_matrix = dists <= self.mask_radius
            valid_channels = adj_matrix[channel_indices]
            valid_mask = valid_channels.unsqueeze(1).to(snippets.dtype)
            snippets *= valid_mask

        return snippets