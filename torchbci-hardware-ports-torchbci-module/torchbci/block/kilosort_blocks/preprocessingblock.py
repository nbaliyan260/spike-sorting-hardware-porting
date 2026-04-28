import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort4 import preprocessing, io 
from torchbci.block.kilosort_blocks.filteringblock import FilteringBlock
from torchbci.block.kilosort_blocks.whiteningblock import WhiteningBlock
from torchbci.kilosort4.utils import (
    log_performance, get_performance
    )
logger = logging.getLogger("kilosort")
from torchbci.block.base.base_preprocessing_block import BasePreprocessingBlock

class PreprocessingBlock(BasePreprocessingBlock):
    
    def __init__(self, device: Optional[torch.device] = None, deterministic_mode: int = 0):
        super().__init__()
        self.device = device
        self.filtering_block = FilteringBlock(device=device, deterministic_mode=deterministic_mode)
        self.whitening_block = WhiteningBlock(device=device)

    def forward(self,
                ops: Dict[str, Any],
                file_object: Optional[Any] = None,
                tic0=np.nan,
                use_jims: bool = False,
                ) -> Dict[str, Any]:
        ops, bfile = self.filtering_block(ops, file_object=file_object, tic0=tic0, use_jims=use_jims)
        ops = self.whitening_block(ops, bfile=bfile, tic0=tic0)
        return ops
# class PreprocessingBlock(nn.Module):
#     """
#     Wraps kilosort.run_kilosort.compute_preprocessing (HP filter + whitening).
#     """
#     def __init__(self, device: Optional[torch.device] = None):
#         super().__init__()
#         self.device = device

#     def compute_preprocessing(self, ops, device, tic0=np.nan, file_object=None):
#         """Compute preprocessing parameters and save them to `ops`.

#         Parameters
#         ----------
#         ops : dict
#             Dictionary storing settings and results for all algorithmic steps.
#         device : torch.device
#             Indicates whether `pytorch` operations should be run on cpu or gpu.
#         tic0 : float; default=np.nan
#             Start time of `run_kilosort`.
#         file_object : array-like file object; optional.
#             Must have 'shape' and 'dtype' attributes and support array-like
#             indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
#             array or memmap.

#         Returns
#         -------
#         ops : dict
        
#         """

#         tic = time.time()
#         logger.info(' ')
#         logger.info('Computing preprocessing variables.')
#         logger.info('-'*40)

#         n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
#             xc, yc, tmin, tmax, artifact, shift, scale, batch_downsampling = \
#                 self.get_run_parameters(ops)
#         nskip = ops['settings']['nskip']
#         whitening_range = ops['settings']['whitening_range']
        
#         # Compute high pass filter
#         cutoff = ops['settings']['highpass_cutoff']
#         hp_filter = preprocessing.get_highpass_filter(fs, cutoff, device=device)
#         # Compute whitening matrix
#         bfile = io.BinaryFiltered(ops['filename'], n_chan_bin, fs, NT, nt, twav_min,
#                                 chan_map, hp_filter, device=device, do_CAR=do_CAR,
#                                 invert_sign=invert, dtype=dtype, tmin=tmin,
#                                 tmax=tmax, artifact_threshold=artifact,
#                                 shift=shift, scale=scale, file_object=file_object,
#                                 batch_downsampling=batch_downsampling)

#         logger.info(f'N samples: {bfile.n_samples}')
#         logger.info(f'N seconds: {bfile.n_samples/fs}')
#         logger.info(f'N batches: {bfile.n_batches}')

#         whiten_mat = preprocessing.get_whitening_matrix(bfile, xc, yc, nskip=nskip,
#                                                         nrange=whitening_range)


#         # Save results
#         ops['Nbatches'] = bfile.n_batches
#         ops['preprocessing'] = {}
#         ops['preprocessing']['whiten_mat'] = whiten_mat
#         ops['preprocessing']['hp_filter'] = hp_filter
#         ops['Wrot'] = whiten_mat
#         ops['fwav'] = hp_filter

#         elapsed = time.time() - tic
#         total = time.time() - tic0
#         ops['runtime_preproc'] = elapsed
#         ops['usage_preproc'] = get_performance()
#         logger.info(f'Preprocessing filters computed in {elapsed:.2f}s; ' +
#                     f'total {total:.2f}s')
#         logger.debug(f'hp_filter shape: {hp_filter.shape}')
#         logger.debug(f'whiten_mat shape: {whiten_mat.shape}')
#         # Check scale of data for log file
#         b1 = bfile.padded_batch_to_torch(0).cpu().numpy()
#         logger.debug(f"First batch min, max: {b1.min(), b1.max()}")

#         log_performance(logger, 'info', 'Resource usage after preprocessing',
#                         reset=True)

#         return ops

#     def get_run_parameters(self, ops) -> list:
#         """Get `ops` dict values needed by `run_kilosort` subroutines."""

#         parameters = [
#             ops['settings']['n_chan_bin'],
#             ops['settings']['fs'],
#             ops['settings']['batch_size'],
#             ops['settings']['nt'],
#             ops['settings']['nt0min'],  # also called twav_min
#             ops['probe']['chanMap'],
#             ops['data_dtype'],
#             ops['do_CAR'],
#             ops['invert_sign'],
#             ops['probe']['xc'],
#             ops['probe']['yc'],
#             ops['settings']['tmin'],
#             ops['settings']['tmax'],
#             ops['settings']['artifact_threshold'],
#             ops['settings']['shift'],
#             ops['settings']['scale'],
#             ops['settings']['batch_downsampling']
#         ]

#         return parameters
#     def forward(
#         self,
#         ops: Dict[str, Any],
#         file_object: Optional[Any] = None,
#         tic0=np.nan
#     ) -> Dict[str, Any]:
#         # print("Computing preprocessing (filtering + whitening)...")
#         return self.compute_preprocessing(ops, self.device, tic0=tic0, file_object=file_object)
