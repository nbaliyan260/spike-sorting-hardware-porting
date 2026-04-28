import numpy as np
import torch
import logging
import time
import random
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort4 import template_matching
from torchbci.kilosort4.utils import (
    get_performance, log_cuda_details, log_performance
    )

logger = logging.getLogger("kilosort")
class MergeBlock(nn.Module):
    """
    Kilosort's template-based merge pass (refractory + template correlation checks).
    Wraps kilosort.template_matching.merging_function.
    """
    def __init__(self, device: Optional[torch.device] = None, check_dt: bool = True, optimize_memory: bool = False):
        super().__init__()
        self.device = device
        self.check_dt = check_dt
        self.optimize_memory = optimize_memory

    def forward(
        self,
        ops: Dict[str, Any],
        Wall: torch.Tensor,
        clu: np.ndarray,
        st: np.ndarray,
        tF: np.ndarray | torch.Tensor,
        tic0=np.nan
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        tic = time.time()
        logger.info(' ')
        logger.info('Merging clusters')
        logger.info('-'*40)
        Wall, clu, is_ref, st, tF = template_matching.merging_function(
            ops, Wall, clu, st, torch.as_tensor(tF), device=self.device, check_dt=self.check_dt, optimize_memory=self.optimize_memory
        )
        elapsed = time.time() - tic
        total = time.time() - tic0
        ops['runtime_merge'] = elapsed
        ops['usage_merge'] = get_performance()
        if torch.cuda.is_available() and self.device == torch.device('cuda'):
            ops['cuda_merge'] = torch.cuda.memory_stats(self.device)
        logger.info(f'{clu.max()+1} units found, in {elapsed:.2f}s; ' + 
                    f'total {total:.2f}s')
        logger.debug(f'clu shape: {clu.shape}')
        logger.debug(f'Wall shape: {Wall.shape}')

        log_cuda_details(logger)
        log_performance(logger, 'info', 'Resource usage after clustering',
                        reset=True)
        return Wall, clu.astype("int32"), is_ref, st, tF, ops