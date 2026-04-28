from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import json

import torch
import torch.nn as nn

from torchbci.block.mountainsort5_blocks import (
    AlignSnippetsBlock,
    AlignTemplatesBlock,
    ComputePCABlock,
    ComputeTemplatesBlock,
    DetectSpikesBlock,
    ExtractSnippetsBlock,
    Isosplit6ClusteringBlock,
    OffsetTimesToPeakBlock,
    RemoveDuplicateTimesBlock,
    RemoveOutOfBoundsBlock,
    ReorderUnitsBlock,
    SortTimesBlock,
)


@dataclass
class SortingParameters:
    detect_threshold: float = 5.5
    detect_channel_radius: Optional[float] = None
    detect_time_radius_msec: float = 0.5
    detect_sign: int = -1
    snippet_T1: int = 20
    snippet_T2: int = 20
    snippet_mask_radius: Optional[float] = None
    npca_per_channel: int = 3
    npca_per_subdivision: int = 10
    skip_alignment: bool = False


@dataclass
class SortingBatch:
    traces: Optional[torch.Tensor] = None
    channel_locations: Optional[torch.Tensor] = None
    sampling_frequency: Optional[float] = None

    times: Optional[torch.Tensor] = None
    channel_indices: Optional[torch.Tensor] = None

    snippets: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None

    labels: Optional[torch.Tensor] = None

    templates: Optional[torch.Tensor] = None
    peak_channel_indices: Optional[torch.Tensor] = None

    alignment_offsets: Optional[torch.Tensor] = None
    offsets_to_peak: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        for v in self.__dict__.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    @property
    def num_channels(self) -> int:
        assert self.traces is not None
        return self.traces.shape[1]

    @property
    def num_timepoints(self) -> int:
        assert self.traces is not None
        return self.traces.shape[0]

    @property
    def num_spikes(self) -> int:
        assert self.times is not None
        return self.times.shape[0]


class MountainSort5Pipeline(nn.Module):
    """
    TorchBCI port of the MountainSort5 top-level pipeline.
    """

    def __init__(self, params: SortingParameters, sampling_frequency: float):
        super().__init__()
        self.params = params
        self.sampling_frequency = sampling_frequency

        self.detect_spikes = DetectSpikesBlock(params, sampling_frequency)
        self.remove_duplicates = RemoveDuplicateTimesBlock()
        self.extract_snippets = ExtractSnippetsBlock(params)

        self.compute_pca = ComputePCABlock(params)
        self.clustering = Isosplit6ClusteringBlock(params)
        self.compute_templates = ComputeTemplatesBlock()

        self.align_templates = AlignTemplatesBlock()
        self.align_snippets = AlignSnippetsBlock()

        self.offset_times_to_peak = OffsetTimesToPeakBlock(params.detect_sign, params.snippet_T1)

        self.sort_times = SortTimesBlock()
        self.remove_out_of_bounds = RemoveOutOfBoundsBlock(params)
        self.reorder_units = ReorderUnitsBlock()

    @torch.inference_mode()
    def forward(self, batch: SortingBatch) -> SortingBatch:
        batch = self.detect_spikes(batch)
        batch = self.remove_duplicates(batch)
        batch = self.extract_snippets(batch)

        batch = self.compute_pca(batch)
        batch = self.clustering(batch)
        batch = self.compute_templates(batch)

        if not self.params.skip_alignment:
            batch = self.align_templates(batch)
            batch = self.align_snippets(batch)

            batch.features = None
            batch = self.compute_pca(batch)
            batch = self.clustering(batch)
            batch = self.compute_templates(batch)

            batch = self.offset_times_to_peak(batch)

        batch = self.sort_times(batch)
        batch = self.remove_out_of_bounds(batch)
        batch = self.reorder_units(batch)

        return batch

    def config_json(self) -> str:
        return json.dumps(asdict(self.params), indent=2)