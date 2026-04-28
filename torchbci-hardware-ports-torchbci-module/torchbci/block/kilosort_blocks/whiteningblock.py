import numpy as np
import torch
import time
import logging
from typing import Any, Dict, Optional
from torchbci.block.kilosort_blocks.basepreprocessingblock import BasePreprocessingBlock

from torchbci.kilosort4 import preprocessing, io
from torchbci.kilosort4.utils import (
    log_performance, get_performance
)

logger = logging.getLogger("kilosort")

class WhiteningBlock(BasePreprocessingBlock):
    """
    Compute the whitening matrix from an already-filtered BinaryFiltered object.
    """
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device

    def compute_whitening(
        self,
        ops: Dict[str, Any],
        bfile: io.BinaryFiltered,
        device: torch.device,
        tic0: float = np.nan,
    ) -> Dict[str, Any]:
        tic = time.time()
        logger.info(' ')
        logger.info('Computing whitening matrix.')
        logger.info('-' * 40)
        (_, _, _, _, _, _, _, _, _, xc, yc,
         _, _, _, _, _, _) = self.get_run_parameters(ops)

        nskip = ops['settings']['nskip']
        whitening_range = ops['settings']['whitening_range']

        whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
                                                        nrange=whitening_range)


        # Save results
        ops.setdefault('preprocessing', {})
        ops['preprocessing']['whiten_mat'] = whiten_mat
        ops['Wrot'] = whiten_mat

        elapsed = time.time() - tic
        total = time.time() - tic0
        ops['runtime_whiten'] = elapsed
        # Keep a combined runtime with the old key name for compatibility
        ops['runtime_preproc'] = ops.get('runtime_filter', 0.0) + elapsed
        ops['usage_preproc'] = get_performance()

        logger.info(f'Whitening matrix computed in {elapsed:.2f}s; total {total:.2f}s')
        logger.debug(f'whiten_mat shape: {whiten_mat.shape}')

        log_performance(logger, 'info', 'Resource usage after preprocessing', reset=True)

        return ops

    def forward(
        self,
        ops: Dict[str, Any],
        bfile: io.BinaryFiltered,
        tic0: float = np.nan,
    ) -> Dict[str, Any]:
        return self.compute_whitening(ops, bfile, self.device, tic0=tic0)
