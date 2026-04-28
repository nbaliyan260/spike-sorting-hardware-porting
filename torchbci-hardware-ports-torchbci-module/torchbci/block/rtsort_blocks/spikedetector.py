import logging
import time
from torch import nn
from pathlib import Path
from torchbci.rtsort.rt_sort import detect_sequences, neuropixels_params
logger = logging.getLogger("rtsort_pipeline")

class DetectRTSortSequences(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inter_path: Path, recording, detection_model:str, recording_window_ms: tuple, return_spikes=False, device="cuda", verbose=True, optimize_memory=False, deterministic_mode=0) -> dict:
        inter_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Calibrating RTSort sequences on window_ms={recording_window_ms} "
            f"using detection_model={detection_model}"
        )
        tic = time.time()
        rt_sort = detect_sequences(
            recording=recording,
            inter_path=inter_path,
            detection_model=detection_model,
            recording_window_ms=recording_window_ms,
            return_spikes=return_spikes,
            device=device,
            deterministic_mode=deterministic_mode,
            optimize_memory=optimize_memory,
            verbose=verbose,
            **neuropixels_params,
        )
        elapsed = time.time() - tic

        logger.info(
            "RTSort calibration done | "
        )
        return rt_sort, elapsed
