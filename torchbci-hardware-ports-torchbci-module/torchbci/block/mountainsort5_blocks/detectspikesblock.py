import math
from typing import Tuple

import torch
import torch.nn.functional as F

from torchbci.block.mountainsort5_blocks.mountainsort5block import MountainSort5Block


class DetectSpikesBlock(MountainSort5Block):
    """
    Detect spikes via temporal max-pooling and spatial peak finding.

    Reads:
      batch.traces
      batch.channel_locations

    Writes:
      batch.times
      batch.channel_indices
    """

    def __init__(self, params, sampling_frequency: float):
        super().__init__()
        self.detect_threshold = params.detect_threshold
        self.detect_sign = params.detect_sign
        self.margin_left = params.snippet_T1
        self.margin_right = params.snippet_T2
        self.channel_radius = params.detect_channel_radius
        self.time_radius = int(
            math.ceil(params.detect_time_radius_msec / 1000 * sampling_frequency)
        )

    def run_block(self, batch):
        traces = batch.traces
        channel_locations = batch.channel_locations
        assert traces is not None and channel_locations is not None

        times, channel_indices = self._detect(traces, channel_locations)
        batch.times = times
        batch.channel_indices = channel_indices
        return batch

    def _detect(
        self,
        traces: torch.Tensor,
        channel_locations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, M = traces.shape
        device = traces.device

        if self.detect_sign > 0:
            oriented = -traces
        elif self.detect_sign == 0:
            oriented = -torch.abs(traces)
        else:
            oriented = traces.clone()
        traces_inv = -oriented

        pool_kernel = 2 * self.time_radius + 1
        traces_inv_t = traces_inv.t().unsqueeze(0)
        temporal_max = F.max_pool1d(
            traces_inv_t,
            kernel_size=pool_kernel,
            stride=1,
            padding=self.time_radius,
        ).squeeze(0).t()

        is_temporal_peak = traces_inv == temporal_max
        is_suprathreshold = traces_inv >= self.detect_threshold

        valid_mask = is_temporal_peak & is_suprathreshold
        if self.margin_left > 0:
            valid_mask[:self.margin_left, :] = False
        if self.margin_right > 0:
            valid_mask[N - self.margin_right:, :] = False

        if self.channel_radius is not None:
            dists = torch.cdist(channel_locations, channel_locations)
            adj_matrix = dists <= self.channel_radius
        else:
            adj_matrix = torch.ones((M, M), dtype=torch.bool, device=device)

        spatial_max = torch.empty_like(temporal_max)
        for m in range(M):
            neighbors = adj_matrix[m]
            spatial_max[:, m] = temporal_max[:, neighbors].max(dim=1).values

        valid_mask = valid_mask & (traces_inv >= spatial_max)

        times_pt, channels_pt = torch.nonzero(valid_mask, as_tuple=True)
        times = times_pt.to(torch.int32)
        channel_indices = channels_pt.to(torch.int32)

        if len(times) > 0:
            inds = torch.argsort(times, stable=True)
            times = times[inds]
            channel_indices = channel_indices[inds]

        return times.to(torch.int64), channel_indices