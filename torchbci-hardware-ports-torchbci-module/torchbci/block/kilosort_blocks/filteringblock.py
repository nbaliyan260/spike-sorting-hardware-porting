import numpy as np
import torch
import time
import logging
from typing import Any, Dict, Optional, Tuple
from torchbci.block.filter import JimsFilter
from torchbci.block.kilosort_blocks.basepreprocessingblock import BasePreprocessingBlock

from torchbci.kilosort4 import preprocessing, io

logger = logging.getLogger("kilosort")

class FilteringBlock(BasePreprocessingBlock):
    """
    Compute the high-pass filter and set up the BinaryFiltered reader.
    """
    def __init__(self, device: Optional[torch.device] = None, deterministic_mode: int = 0):
        super().__init__()
        self.device = device
        self.deterministic_mode = deterministic_mode

    def compute_filtering(
        self,
        ops: Dict[str, Any],
        device: torch.device,
        tic0: float = np.nan,
        file_object: Optional[Any] = None,
        use_jims: bool = False
    ) -> Tuple[Dict[str, Any], io.BinaryFiltered]:
        tic = time.time()
        logger.info(' ')
        logger.info('Computing filtering variables.')
        logger.info('-' * 40)

        (n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert,
         xc, yc, tmin, tmax, artifact, shift, scale,
         batch_downsampling) = self.get_run_parameters(ops)

        jims_filter = None
        hp_filter = None
        if use_jims:
            jims_filter = JimsFilter(window_size=21).to(device)
        else:
            # Compute high pass filter
            cutoff = ops['settings']['highpass_cutoff']
            hp_filter = preprocessing.get_highpass_filter(fs, cutoff, device=device)
            jims_filter = None

        bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
                                chan_map, hp_filter, device=device, do_CAR=do_CAR,
                                invert_sign=invert, dtype=dtype, tmin=tmin,
                                tmax=tmax, artifact_threshold=artifact,
                                shift=shift, scale=scale, file_object=file_object,
                                batch_downsampling=batch_downsampling, filter_fn=jims_filter, deterministic_mode=self.deterministic_mode)

        logger.info(f'N samples: {bfile.n_samples}')
        logger.info(f'N seconds: {bfile.n_samples / fs}')
        logger.info(f'N batches: {bfile.n_batches}')

        # Save results related to filtering
        ops.setdefault('preprocessing', {})
        if hp_filter is not None:
            ops['preprocessing']['hp_filter'] = hp_filter
        ops['Nbatches'] = bfile.n_batches
        ops['fwav'] = hp_filter  # for KS4 compatibility

        elapsed = time.time() - tic
        total = time.time() - tic0
        ops['runtime_filter'] = elapsed
        logger.info(f'Filtering computed in {elapsed:.2f}s; total {total:.2f}s')
        if not use_jims:
            logger.debug(f'hp_filter shape: {hp_filter.shape}')

        # Optional sanity check on first filtered batch
        b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
        logger.debug(f"First batch (filtered) min, max: {b1.min(), b1.max()}")

        return ops, bfile
    
    def forward(
        self,
        ops: Dict[str, Any],
        file_object: Optional[Any] = None,
        tic0: float = np.nan,
        use_jims: bool = False
    ) -> Tuple[Dict[str, Any], io.BinaryFiltered]:
        return self.compute_filtering(ops, self.device, tic0=tic0, file_object=file_object, use_jims=use_jims)
