import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort3 import ks3_first_clustering
from torchbci.kilosort4 import template_matching, clustering_qr
from torchbci.kilosort4.utils import (
    log_performance, get_performance, log_thread_count
)

logger = logging.getLogger("kilosort")

from torchbci.block.base.base_clustering_block import BaseClusteringBlock

class FirstClusteringBlock(BaseClusteringBlock):

    def __init__(
        self,
        device: Optional[torch.device] = None,
        clear_cache: bool = False,
        verbose: bool = False,
        optimize_memory: bool = False,
    ):
        super().__init__()
        self.device = device
        self.clear_cache = clear_cache
        self.verbose = verbose
        self.optimize_memory = optimize_memory

    def forward(
        self,
        ops: Dict[str, Any],
        st0: np.ndarray,
        tF: torch.Tensor,
        progress_bar: Optional[Any] = None,
        tic0: float = np.nan,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        device = self.device
        clear_cache = self.clear_cache
        verbose = self.verbose

        tic = time.time()
        logger.info(" ")
        logger.info("First clustering")
        logger.info("-" * 40)

        # --- NEW (KS3/KS4 switch) ---
        if ops['settings'].get('algorithm', 'ks4') == 'ks3':
            clu, Wall = ks3_first_clustering(
                ops, st0, tF, device=device, progress_bar=progress_bar,
                optimize_memory=self.optimize_memory
            )
        else:
            clu, Wall = clustering_qr.run(
                ops, st0, tF, mode='spikes', device=device, progress_bar=progress_bar,
                clear_cache=clear_cache, verbose=verbose, optimize_memory=self.optimize_memory
                )
        Wall3 = template_matching.postprocess_templates(
            Wall, ops, clu, st0, tF, device=device, optimize_memory=self.optimize_memory
            )

        elapsed = time.time() - tic
        total = time.time() - tic0
        ops["runtime_clu0"] = elapsed
        ops["usage_clu0"] = get_performance()
        if torch.cuda.is_available() and device == torch.device("cuda"):
            ops["cuda_clu0"] = torch.cuda.memory_stats(device)
        logger.info(
            f"{clu.max() + 1} clusters found, in {elapsed:.2f}s; total {total:.2f}s"
        )
        logger.debug(f"clu shape: {clu.shape}")
        logger.debug(f"Wall shape: {Wall.shape}")
        log_performance(
            logger, "info", "Resource usage after first clustering", reset=True
        )
        log_thread_count(logger)

        return clu, Wall, Wall3, ops
