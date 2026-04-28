import numpy as np
import torch
import time
import logging
from torch import nn
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from torchbci.kilosort4 import io

from torchbci.kilosort4.utils import (
    get_performance, log_cuda_details, log_performance
    )

logger = logging.getLogger("kilosort")
from torchbci.block.base.base_save_block import BaseSaveBlock

class SaveResultsBlock(BaseSaveBlock):
    """
    Wraps kilosort.run_kilosort.save_sorting to write Phy-compatible outputs.
    """
    def __init__(self, results_dir: Path, save_extra_vars: bool = False, save_preprocessed_copy: bool = False, optimize_memory: bool = False):
        super().__init__()
        self.results_dir = Path(results_dir)
        self.save_extra_vars = save_extra_vars
        self.save_preprocessed_copy = save_preprocessed_copy
        self.optimize_memory = optimize_memory
    def save_sorting(self, ops, results_dir, st, clu, tF, Wall, imin, tic0=np.nan,
                 save_extra_vars=False, save_preprocessed_copy=False,
                 skip_dat_path=False):  
        """Save sorting results, and format them for use with Phy

        Parameters
        -------
        ops : dict
            Dictionary storing settings and results for all algorithmic steps.
        results_dir : pathlib.Path
            Directory where results should be saved.
        st : np.ndarray
            3-column array of peak time (in samples), template, and thresold
            amplitude for each spike.
        clu : np.ndarray
            1D vector of cluster ids indicating which spike came from which cluster,
            same shape as `st[:,0]`.
        tF : torch.Tensor
            PC features for each spike, with shape
            (n_spikes, nearest_chans, n_pcs)
        Wall : torch.Tensor
            PC feature representation of spike waveforms for each cluster, with shape
            (n_clusters, n_channels, n_pcs).
        imin : int
            Minimum sample index used by BinaryRWFile, exported spike times will
            be shifted forward by this number.
        tic0 : float; default=np.nan.
            Start time of `run_kilosort`.
        save_extra_vars : bool; default=False.
            If True, save tF and Wall to disk along with copies of st, clu and
            amplitudes with no postprocessing applied.
        save_preprocessed_copy : bool; default=False.
            If True, save a pre-processed copy of the data (including drift
            correction) to `temp_wh.dat` in the results directory and format Phy
            output to use that copy of the data.
        skip_dat_path : bool; default=False.
            If True, will save `dat_path = 'no_path.bin'` in `params.py` in place
            of a real filename. This is done to prevent an error in Phy when filename
            has an unexpected format, like when using a `file_object` loaded from
            an external data format through SpikeInterface. The full filename(s) will
            still be included in `params.py` for reference, but will be commented out.

        Returns
        -------
        ops : dict
            Dictionary storing settings and results for all algorithmic steps.
        similar_templates : np.ndarray.
            Similarity score between each pair of clusters, computed as correlation
            between clusters. Shape (n_clusters, n_clusters).
        is_ref : np.ndarray.
            1D boolean array with shape (n_clusters,) indicating whether each
            cluster is refractory.
        est_contam_rate : np.ndarray.
            Contamination rate for each cluster, computed as fraction of refractory
            period violations relative to expectation based on a Poisson process.
            Shape (n_clusters,).
        kept_spikes : np.ndarray.
            Boolean mask with shape (n_spikes,) that is False for spikes that were
            removed by `kilosort.postprocessing.remove_duplicate_spikes`
            and True otherwise.

        """

        tic = time.time()
        logger.info(' ')
        logger.info('Saving to phy and computing refractory periods')
        logger.info('-'*40)
        results_dir, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            io.save_to_phy(
                st, clu, tF, Wall, ops['probe'], ops, imin, results_dir=results_dir,
                data_dtype=ops['data_dtype'], save_extra_vars=save_extra_vars,
                save_preprocessed_copy=save_preprocessed_copy,
                skip_dat_path=skip_dat_path, optimize_memory=self.optimize_memory
                )
        logger.info(f'{int(is_ref.sum())} units found with good refractory periods')
        
        ops['n_units_total'] = np.unique(clu).size
        ops['n_units_good'] = int(is_ref.sum())
        if torch.is_tensor(st):
            kept_spikes_t = torch.from_numpy(kept_spikes).to(st.device)
            ops['n_spikes'] = st[kept_spikes_t].shape[0]
        else:
            ops['n_spikes'] = st[kept_spikes].shape[0]
        if ops.get('dshift', None) is not None:
            ops['mean_drift'] = np.abs(ops['dshift']).mean(axis=0)[0]
        else:
            ops['mean_drift'] = np.nan

        elapsed = elapsed = time.time() - tic
        ops['runtime_postproc'] = elapsed
        ops['usage_postproc'] = get_performance()
        logger.info(f'Exporting to Phy took: {elapsed:.2f}s')

        runtime = time.time()-tic0
        seconds = runtime % 60
        mins = runtime // 60
        hrs = mins // 60
        mins = mins % 60

        logger.info(f'Total runtime: {runtime:.2f}s = {int(hrs):02d}:' +
                    f'{int(mins):02d}:{round(seconds)} h:m:s')
        ops['runtime'] = runtime 
        io.save_ops(ops, results_dir)
        logger.info(f'Sorting output saved in: {results_dir}.')

        log_cuda_details(logger)
        log_performance(logger, 'info', 'Resource usage after saving',
                        reset=True)
        return ops, similar_templates, is_ref, est_contam_rate, kept_spikes

    def forward(
        self,
        ops: Dict[str, Any],
        st: np.ndarray,
        clu: np.ndarray,
        tF: np.ndarray | torch.Tensor,
        Wall: torch.Tensor,
        bfile: Optional[io.BinaryFiltered] = None,
        skip_dat_path: bool = False,
        tic0=np.nan
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # print("Saving sorting results to disk...")
        # 'imin' is required by save_sorting; get from bfile if available else 0
        imin = getattr(bfile, "imin", 0) if bfile is not None else 0
        out = self.save_sorting(
            ops,
            self.results_dir,
            st,
            clu,
            torch.as_tensor(tF),
            Wall,
            imin,
            tic0=tic0,
            save_extra_vars=self.save_extra_vars,
            save_preprocessed_copy=self.save_preprocessed_copy,
            skip_dat_path=skip_dat_path,
        )
        # Returns: ops, similar_templates, is_ref, est_contam_rate, kept_spikes
        return out
