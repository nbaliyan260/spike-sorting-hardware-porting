from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Union
import json

import numpy as np
import torch
from torch import nn

from torchbci.block.combinato_blocks import (
    Preprocessor,
    ThresholdDetector,
    WaveformExtractor,
    CubicUpsampler,
    PeakAligner,
    WaveletFeatureExtractor,
    FeatureSelector,
    SPCClusterer,
    ClusterDefiner,
    TemplateMatcher,
    ArtifactDetector,
)


@dataclass
class CombinatoConfig:
    sample_rate: int = 30000
    spc_path: Union[str, Path] = '~/spc/cluster_linux64.exe'
    device: str = 'cpu'
    output_dir: Union[str, Path] = '/tmp/combinato_ported'

    def to_json(self) -> str:
        d = asdict(self)
        d['spc_path'] = str(self.spc_path)
        d['output_dir'] = str(self.output_dir)
        return json.dumps(d, indent=2)


class CombinatoPipeline(nn.Module):
    """First TorchBCI port of Combinato, preserving the naive per-channel sequential flow."""

    def __init__(self, cfg: CombinatoConfig):
        super().__init__()
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.device = cfg.device

        self.pre = Preprocessor(sample_rate=cfg.sample_rate)
        self.detector = ThresholdDetector(sample_rate=cfg.sample_rate)
        self.extractor = WaveformExtractor(self.pre)
        self.upsampler = CubicUpsampler()
        self.aligner = PeakAligner()
        self.c1 = WaveletFeatureExtractor()
        self.c2 = FeatureSelector()
        self.c3 = SPCClusterer(cluster_path=os.path.expanduser(str(cfg.spc_path)))
        self.c4 = ClusterDefiner()
        self.c5 = TemplateMatcher()
        self.c6 = ArtifactDetector()

        if cfg.device == 'cuda':
            self.pre = self.pre.cuda()
            self.detector = self.detector.cuda()
            self.extractor = self.extractor.cuda()
            self.upsampler = self.upsampler.cuda()
            self.aligner = self.aligner.cuda()
            self.c1 = self.c1.cuda()
            self.c2 = self.c2.cuda()
            self.c6 = self.c6.cuda()

    @torch.inference_mode()
    def forward(self, raw_data: np.ndarray) -> Dict[int, Dict[str, Any]]:
        n_samples, n_channels = raw_data.shape
        output_dir = Path(self.cfg.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[int, Dict[str, Any]] = {}

        for ch in range(n_channels):
            signal = torch.tensor(
                raw_data[:, ch].astype(np.float64),
                dtype=torch.float64,
                device=self.device,
            )

            folder = output_dir / f'ch{ch}'
            folder.mkdir(parents=True, exist_ok=True)

            denoised, detected = self.pre(signal)
            _, neg_idx, _ = self.detector(detected)

            if len(neg_idx) < 15:
                continue

            spikes_raw, _, _ = self.extractor(denoised, neg_idx, 'neg')
            spikes_up = self.upsampler(spikes_raw)
            spikes_aligned, kept_mask = self.aligner(spikes_up)

            spk = -spikes_aligned.detach().cpu().numpy()
            spk_t = torch.tensor(spk, dtype=torch.float64, device=self.device)

            features = self.c1(spk_t)
            features_sel, selected_features = self.c2(features)

            clu, tree = self.c3(features_sel.cpu(), str(folder), 'combinato', 12345.0)
            sort_idx, groups, temperature = self.c4(clu, tree)

            sort_idx_np = sort_idx.detach().cpu().numpy().astype(np.uint16).copy()
            match_idx = np.zeros(len(sort_idx_np), dtype=np.int8)
            self.c5(spk, sort_idx_np, match_idx)
            _, artifact_ids = self.c6(spk_t, sort_idx, sign='neg')

            results[ch] = {
                'spikes': spk,
                'clusters': sort_idx_np,
                'artifacts': artifact_ids,
                'kept_mask': kept_mask.detach().cpu().numpy() if torch.is_tensor(kept_mask) else kept_mask,
                'selected_features': selected_features.detach().cpu().numpy() if torch.is_tensor(selected_features) else selected_features,
                'temperature': temperature.detach().cpu().numpy() if torch.is_tensor(temperature) else temperature,
            }

        return results
