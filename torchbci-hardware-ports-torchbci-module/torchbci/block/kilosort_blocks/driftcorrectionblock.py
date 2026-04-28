import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.block.filter import JimsFilter
from torchbci.kilosort4 import io, datashift
from torchbci.kilosort4.utils import (
    log_performance, get_performance, log_cuda_details
    )

logger = logging.getLogger("kilosort")

#TODO: base on base parameters like the preprocessing
class DriftCorrectionBlock(nn.Module):
    """
    Wraps kilosort.compute_drift_correction (computes ops['dshift'] and
    returns a drift-corrected BinaryFiltered for downstream steps).
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        clear_cache: bool = False,
        verbose: bool = False,
        deterministic_mode: int = 0,
        optimize_memory: bool = False,
        seed: int = 1
    ):
        super().__init__()
        self.device = device
        self.clear_cache = clear_cache
        self.verbose = verbose
        self.deterministic_mode = deterministic_mode
        self.optimize_memory = optimize_memory
        self.seed = seed

    def get_run_parameters(self, ops) -> list:
        """Get `ops` dict values needed by `run_kilosort` subroutines."""

        parameters = [
            ops['settings']['n_chan_bin'],
            ops['settings']['fs'],
            ops['settings']['batch_size'],
            ops['settings']['nt'],
            ops['settings']['nt0min'],  # also called twav_min
            ops['probe']['chanMap'],
            ops['data_dtype'],
            ops['do_CAR'],
            ops['invert_sign'],
            ops['probe']['xc'],
            ops['probe']['yc'],
            ops['settings']['tmin'],
            ops['settings']['tmax'],
            ops['settings']['artifact_threshold'],
            ops['settings']['shift'],
            ops['settings']['scale'],
            ops['settings']['batch_downsampling']
        ]

        return parameters
    def compute_drift_correction(self, ops, device, tic0=np.nan, progress_bar=None,
                             file_object=None, clear_cache=False, verbose=False, skip_drift=False, use_jims=False):
        """Compute drift correction parameters and save them to `ops`.

        Parameters
        ----------
        ops : dict
            Dictionary storing settings and results for all algorithmic steps.
        device : torch.device
            Indicates whether `pytorch` operations should be run on cpu or gpu.
        tic0 : float; default=np.nan.
            Start time of `run_kilosort`.
        progress_bar : TODO; optional.
            Informs `tqdm` package how to report progress, type unclear.
        file_object : array-like file object; optional.
            Must have 'shape' and 'dtype' attributes and support array-like
            indexing (e.g. [:100,:], [5, 7:10], etc). For example, a numpy
            array or memmap.
        clear_cache : bool; False.
            If True, force pytorch to clear cached cuda memory after some
            memory-intensive steps in the pipeline.
        verbose : bool; False.
            If true, include additional debug-level logging statements.

        Returns
        -------
        ops : dict
            Dictionary storing settings and results for all algorithmic steps.
        bfile : kilosort.io.BinaryFiltered
            Wrapped file object for handling data.
        st0 : np.ndarray.
            Intermediate spike times variable with 6 columns. This is only used
            for generating the 'Drift Scatter' plot through the GUI.
        
        """

        tic = time.time()
        logger.info(' ')
        logger.info('Computing drift correction.')
        logger.info('-'*40)

        n_chan_bin, fs, NT, nt, twav_min, chan_map, dtype, do_CAR, invert, \
            _, _, tmin, tmax, artifact, shift, scale, batch_downsampling = \
                self.get_run_parameters(ops)
        if use_jims:
            jims_filter = JimsFilter(window_size=21).to(device)
            hp_filter = None
        else:
            hp_filter = ops['preprocessing']['hp_filter']
            jims_filter = None
        whiten_mat = ops['preprocessing']['whiten_mat']
        bfile = io.BinaryFiltered(
            ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
            hp_filter=hp_filter, whiten_mat=whiten_mat, device=device, do_CAR=do_CAR,
            invert_sign=invert, dtype=dtype, tmin=tmin, tmax=tmax,
            artifact_threshold=artifact, shift=shift, scale=scale,
            file_object=file_object, batch_downsampling=batch_downsampling,
            filter_fn=jims_filter, deterministic_mode=self.deterministic_mode
            )
        
        if skip_drift:
            logger.info('Skipping drift correction as per user request.')
            ops['dshift'] = None
            ops["yblk"] = None
            ops['iKxx'] = None
            ops['runtime_drift'] = 0
            ops['usage_drift'] = None
            ops['cuda_drift'] = None
            return ops, bfile, None
        
        ops, st = datashift.run(ops, bfile, device=device, progress_bar=progress_bar,
                                clear_cache=clear_cache, verbose=verbose, optimize_memory=self.optimize_memory, deterministic_mode=self.deterministic_mode, seed=self.seed)
        
        elapsed = time.time() - tic
        total = time.time() - tic0
        ops['runtime_drift'] = elapsed
        ops['usage_drift'] = get_performance()
        if torch.cuda.is_available() and device == torch.device('cuda'):
            ops['cuda_drift'] = torch.cuda.memory_stats()
        logger.info(f'drift computed in {elapsed:.2f}s; total {total:.2f}s')

        if st is not None:
            logger.debug(f'st shape: {st.shape}')
            logger.debug(f'yblk shape: {ops["yblk"].shape}')
            logger.debug(f'dshift shape: {ops["dshift"].shape}')
            logger.debug(f'iKxx shape: {ops["iKxx"].shape}')
        
        # binary file with drift correction
        bfile = io.BinaryFiltered(
            ops['filename'], n_chan_bin, fs, NT, nt, twav_min, chan_map, 
            hp_filter=hp_filter, whiten_mat=whiten_mat, device=device,
            dshift=ops['dshift'], do_CAR=do_CAR, dtype=dtype, tmin=tmin, tmax=tmax,
            artifact_threshold=artifact, shift=shift, scale=scale,
            file_object=file_object, batch_downsampling=batch_downsampling,
            filter_fn=jims_filter, deterministic_mode=self.deterministic_mode
            )

        log_cuda_details(logger)
        log_performance(logger, 'info', 'Resource usage after drift correction',
                        reset=True)
        return ops, bfile, st

    def forward(
        self,
        ops: Dict[str, Any],
        file_object: Optional[Any] = None,
        progress_bar: Optional[Any] = None,
        tic0=np.nan,
        skip_drift: bool = False,
        use_jims: bool = False,
    ) -> Tuple[Dict[str, Any], io.BinaryFiltered, Optional[np.ndarray]]:
        ops, bfile, st_scatter = self.compute_drift_correction(
            ops,
            self.device,
            tic0=tic0,
            progress_bar=progress_bar,
            file_object=file_object,
            clear_cache=self.clear_cache,
            verbose=self.verbose,
            skip_drift=skip_drift,
            use_jims=use_jims,
        )
        return ops, bfile, st_scatter