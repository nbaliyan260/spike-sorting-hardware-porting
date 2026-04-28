import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort4 import spikedetect, io
from torchbci.kilosort4.utils import (
    log_performance, get_performance, log_thread_count
)

logger = logging.getLogger("kilosort")
from torchbci.block.base.base_detection_block import BaseDetectionBlock

class SpikeTemplateDetectionBlock(BaseDetectionBlock):
    """
    Extract spikes using templates (spikedetect.run).
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        clear_cache: bool = False,
        verbose: bool = False,
        optimize_memory: bool = False,
        deterministic_mode: int = 0,
        seed: int = 1
    ):
        super().__init__()
        self.device = device
        self.clear_cache = clear_cache
        self.verbose = verbose
        self.optimize_memory = optimize_memory
        self.deterministic_mode = deterministic_mode
        self.seed = seed

    def forward(
        self,
        ops: Dict[str, Any],
        bfile: io.BinaryFiltered,
        progress_bar: Optional[Any] = None,
        tic0: float = np.nan,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
        device = self.device
        clear_cache = self.clear_cache
        verbose = self.verbose

        tic = time.time()
        logger.info(" ")
        logger.info("Extracting spikes using templates")
        logger.info("-" * 40)
        st0, tF, ops = spikedetect.run(
            ops,
            bfile,
            device=device,
            progress_bar=progress_bar,
            clear_cache=clear_cache,
            verbose=verbose,
            optimize_memory=self.optimize_memory, 
            deterministic_mode=self.deterministic_mode,
            seed=self.seed
        )
        if not self.optimize_memory:
            tF = torch.from_numpy(tF)

        elapsed = time.time() - tic
        total = time.time() - tic0
        ops["runtime_st0"] = elapsed
        ops["usage_st0"] = get_performance()
        if torch.cuda.is_available() and device == torch.device("cuda"):
            ops["cuda_st0"] = torch.cuda.memory_stats(device)
        logger.info(
            f"{len(st0)} spikes extracted in {elapsed:.2f}s; total {total:.2f}s"
        )
        logger.debug(f"st0 shape: {st0.shape}")
        logger.debug(f"tF shape: {tF.shape}")
        if len(st0) == 0:
            raise ValueError("No spikes detected, cannot continue sorting.")
        log_performance(
            logger, "info", "Resource usage after spike detect (univ)", reset=True
        )
        log_thread_count(logger)

        return st0, tF, ops