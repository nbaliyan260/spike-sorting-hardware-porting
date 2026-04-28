import numpy as np
import torch
import logging
import time
import random
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort3 import ks3_final_clustering
from torchbci.kilosort4 import clustering_qr
from torchbci.kilosort4.utils import (
    get_performance, log_thread_count
    )

logger = logging.getLogger("kilosort")
from torchbci.block.base.base_clustering_block import BaseClusteringBlock

class FinalGraphClusteringBlock(BaseClusteringBlock):
    """
    Final graph-based clustering on features from template re-extraction (mode='template').
    Wraps kilosort.clustering_qr.run.
    """
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
        st: np.ndarray,
        tF: np.ndarray | torch.Tensor,
        progress_bar: Optional[Any] = None,
        tic0=np.nan
    ) -> Tuple[np.ndarray, torch.Tensor]:
        # print("Running final graph-based clustering on re-extracted spikes...")
        tF_torch = torch.as_tensor(tF) if not torch.is_tensor(tF) else tF
        tic = time.time()
        logger.info(' ')
        logger.info('Final clustering')
        logger.info('-'*40)
        if ops['settings'].get('algorithm', 'ks4') == 'ks3':
            clu, Wall = ks3_final_clustering(
                ops, st, tF, device=self.device, progress_bar=progress_bar,
                optimize_memory=self.optimize_memory
            )
        else:
            clu, Wall = clustering_qr.run(
                ops,
                st,
                tF_torch,
                mode="template",
                device=self.device,
                progress_bar=progress_bar,
                clear_cache=self.clear_cache,
                verbose=self.verbose,
                optimize_memory=self.optimize_memory
            )
        elapsed = time.time() - tic
        total = time.time() - tic0
        ops['runtime_clu'] = elapsed
        ops['usage_clu'] = get_performance()
        if torch.cuda.is_available() and self.device == torch.device('cuda'):
            ops['cuda_clu'] = torch.cuda.memory_stats(self.device)
        logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' + 
                    f'total {total:.2f}s')
        logger.debug(f'clu shape: {clu.shape}')
        logger.debug(f'Wall shape: {Wall.shape}')
        log_thread_count(logger)

        return clu, Wall, ops
