import time

import torch
import numpy as np
import logging
import json
from torch import nn
from pathlib import Path

from torchbci.kilosort4.postprocessing import remove_duplicates
logger = logging.getLogger("rtsort_pipeline")
class SaveRTBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, all_times, all_clusters, out_dir):
        tic = time.time()
        dtype_times=np.int64
        dtype_clusters=np.int32
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_times = np.asarray(all_times, dtype=dtype_times)
        all_clusters = np.asarray(all_clusters, dtype=dtype_clusters)
        if all_times.size > 0:
            # Canonical ordering avoids order-only run-to-run differences and
            # makes duplicate removal stable.
            order = np.lexsort((all_clusters, all_times))
            all_times = all_times[order]
            all_clusters = all_clusters[order]
        spike_times, spike_clusters, _ = remove_duplicates(
            all_times.astype(np.int64, copy=False),
            all_clusters.astype(np.int32, copy=False),
            np.int32(15),
        )
        spike_times = spike_times.astype(dtype_times, copy=False)
        spike_clusters = spike_clusters.astype(dtype_clusters, copy=False)
        np.save(out_dir / "spike_times.npy", spike_times)
        np.save(out_dir / "spike_clusters.npy", spike_clusters)
        elapsed = time.time() - tic
        # Optional: save a small metadata json
        logger.info(f"Saved Kilosort export to: {out_dir} | n_spikes={spike_times.size}")
        return spike_times, spike_clusters, elapsed
