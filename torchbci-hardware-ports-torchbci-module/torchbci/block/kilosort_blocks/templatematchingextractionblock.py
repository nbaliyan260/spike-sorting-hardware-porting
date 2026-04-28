import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort4 import io, template_matching
from torchbci.kilosort4.utils import (
    log_performance, get_performance, log_cuda_details, log_thread_count
)

logger = logging.getLogger("kilosort")

class TemplateMatchingExtractionBlock(nn.Module):
    """
    Extract spikes using cluster waveforms (template_matching.extract).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        optimize_memory: bool = False,
        deterministic_mode: int = 0,
    ):
        super().__init__()
        self.device = device
        self.optimize_memory = optimize_memory
        self.deterministic_mode = deterministic_mode

    def forward(
        self,
        ops: Dict[str, Any],
        bfile: io.BinaryFiltered,
        Wall3: torch.Tensor,
        progress_bar: Optional[Any] = None,
        tic0: float = np.nan,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
        device = self.device

        tic = time.time()
        logger.info(" ")
        logger.info("Extracting spikes using cluster waveforms")
        logger.info("-" * 40)

        st, tF, ops = template_matching.extract(
            ops, bfile, Wall3, device=device, progress_bar=progress_bar, optimize_memory=self.optimize_memory, deterministic_mode=self.deterministic_mode
        )

        log_thread_count(logger)

        elapsed = time.time() - tic
        total = time.time() - tic0
        ops["runtime_st"] = elapsed
        ops["usage_st"] = get_performance()
        if torch.cuda.is_available() and device == torch.device("cuda"):
            ops["cuda_st"] = torch.cuda.memory_stats(device)
        logger.info(
            f"{len(st)} spikes extracted in {elapsed:.2f}s; total {total:.2f}s"
        )
        logger.debug(f"st shape: {st.shape}")
        logger.debug(f"tF shape: {tF.shape}")
        logger.debug(f'iCC shape: {ops["iCC"].shape}')
        logger.debug(f'iU shape: {ops["iU"].shape}')

        log_cuda_details(logger)
        log_performance(
            logger,
            "info",
            "Resource usage after spike detect (learned)",
            reset=True,
        )

        return st, tF, ops