import hashlib
import numpy as np
import torch
import time
import logging
import random
from torch import nn
from typing import Any, Dict, Optional, Tuple

from torchbci.kilosort4 import spikedetect, io, template_matching, clustering_qr
from torchbci.kilosort4.utils import (
    log_performance, get_performance, log_cuda_details, log_thread_count
    )
from torchbci.block.kilosort_blocks.spiketemplatedetectionblock import SpikeTemplateDetectionBlock
from torchbci.block.kilosort_blocks.firstclusteringblock import FirstClusteringBlock
from torchbci.block.kilosort_blocks.templatematchingextractionblock import TemplateMatchingExtractionBlock

logger = logging.getLogger("kilosort")
def _sha256_sampled(x, sample_elems: int = 200_000) -> dict:
    if x is None:
        return {"shape": None, "dtype": None, "sha256": None}

    # --- torch ---
    if torch.is_tensor(x):
        t = x.detach()
        shape = list(t.shape)
        dtype = str(t.dtype)

        flat = t.reshape(-1)
        n = flat.numel()
        if n == 0:
            return {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(b"").hexdigest()}

        step = max(1, n // sample_elems)
        sample = flat[::step][:sample_elems]

        arr = np.ascontiguousarray(sample.contiguous().cpu().numpy())
        h = hashlib.sha256(arr.view(np.uint8)).hexdigest()
        return {"shape": shape, "dtype": dtype, "sha256": h}

    # --- cupy ---
    if isinstance(x, cp.ndarray):
        shape = list(x.shape)
        dtype = str(x.dtype)

        flat = x.ravel()
        n = flat.size
        if n == 0:
            return {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(b"").hexdigest()}

        step = max(1, n // sample_elems)
        sample = flat[::step][:sample_elems]

        arr = np.ascontiguousarray(cp.asnumpy(sample))
        h = hashlib.sha256(arr.view(np.uint8)).hexdigest()
        return {"shape": shape, "dtype": dtype, "sha256": h}

    # --- numpy / anything array-like ---
    arr0 = np.asarray(x)
    shape = list(arr0.shape)
    dtype = str(arr0.dtype)

    flat = arr0.reshape(-1)
    n = flat.size
    if n == 0:
        return {"shape": shape, "dtype": dtype, "sha256": hashlib.sha256(b"").hexdigest()}

    step = max(1, n // sample_elems)
    sample = flat[::step][:sample_elems]

    sample = np.ascontiguousarray(sample)
    h = hashlib.sha256(sample.view(np.uint8)).hexdigest()
    return {"shape": shape, "dtype": dtype, "sha256": h}

from torchbci.block.base.base_detection_block import BaseDetectionBlock

class SpikeDetectionBlock(BaseDetectionBlock):
    """
    Wraps kilosort.detect_spikes.
    """
    def __init__(
        self,
        device: Optional[torch.device] = None,
        optimize_memory: bool = False,
        deterministic_mode: int = 0,
        seed: int = 1
    ):
        super().__init__()
        self.device = device
        self.spike_template_detection_block = SpikeTemplateDetectionBlock(device=device, optimize_memory=optimize_memory, deterministic_mode=deterministic_mode, seed=seed)
        self.first_cluster_block = FirstClusteringBlock(device=device, optimize_memory=optimize_memory)
        self.extract_block = TemplateMatchingExtractionBlock(device=device, optimize_memory=optimize_memory, deterministic_mode=deterministic_mode)
        self.hash_trace = []
    def record_hash(self, stage: str, tic0: float, **named):
        rec = {"stage": stage, "t_rel_s": float(time.time() - tic0), "items": {}}
        for name, obj in named.items():
            if obj is None:
                continue
            try:
                rec["items"][name] = _sha256_sampled(obj, sample_elems=200_000)
            except Exception as e:
                rec["items"][name] = {"error": repr(e)}
        self.hash_trace.append(rec)

        # short log line
        short = ", ".join(
            f"{k}={v.get('sha256','')[:12]}"
            for k, v in rec["items"].items()
            if isinstance(v, dict) and v.get("sha256") is not None
        )
        if short:
            logger.info(f"HASH {stage}: {short}")
        else:
            logger.info(f"HASH {stage}: (no hashable items)")
    def forward(
        self,
        ops: Dict[str, Any],
        bfile: io.BinaryFiltered,
        progress_bar: Optional[Any] = None,
        tic0=np.nan
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        st0, tF, ops = self.spike_template_detection_block(ops, bfile, progress_bar=progress_bar, tic0=tic0)
        self.record_hash("spike_template_detection_block", tic0, st0=st0, tF=tF)
        clu, Wall, Wall3, ops = self.first_cluster_block(ops, st0, tF, progress_bar=progress_bar, tic0=tic0)
        self.record_hash("first_cluster_block", tic0, clu=clu, Wall=Wall, Wall3=Wall3)
        st, tF, ops = self.extract_block(ops, bfile, Wall3, progress_bar=progress_bar, tic0=tic0)
        self.record_hash("extract_block", tic0, st=st, tF=tF)
        return st, tF, ops, self.hash_trace
        
# class SpikeDetectionBlock(nn.Module):
#     """
#     Wraps kilosort.detect_spikes.
#     """
#     def __init__(
#         self,
#         device: Optional[torch.device] = None,
#         clear_cache: bool = False,
#         verbose: bool = False
#     ):
#         super().__init__()
#         self.device = device
#         self.clear_cache = clear_cache
#         self.verbose = verbose
        
#     def detect_spikes(self, ops, device, bfile, tic0=np.nan, progress_bar=None,
#                   clear_cache=False, verbose=False):
#         """Detect spikes via template deconvolution.
        
#         Parameters
#         ----------
#         ops : dict
#             Dictionary storing settings and results for all algorithmic steps.
#         device : torch.device
#             Indicates whether `pytorch` operations should be run on cpu or gpu.
#         bfile : kilosort.io.BinaryFiltered
#             Wrapped file object for handling data.
#         tic0 : float; default=np.nan.
#             Start time of `run_kilosort`.
#         progress_bar : TODO; optional.
#             Informs `tqdm` package how to report progress, type unclear.
#         clear_cache : bool; False.
#             If True, force pytorch to clear cached cuda memory after some
#             memory-intensive steps in the pipeline.
#         verbose : bool; False.
#             If true, include additional debug-level logging statements.

#         Returns
#         -------
#         st : np.ndarray
#             3-column array of peak time (in samples), template, and thresold
#             amplitude for each spike.
#         clu : np.ndarray
#             1D vector of cluster ids indicating which spike came from which cluster,
#             same shape as `st`.
#         tF : torch.Tensor
#             PC features for each spike, with shape
#             (n_spikes, nearest_chans, n_pcs)
#         Wall : torch.Tensor
#             PC feature representation of spike waveforms for each cluster, with shape
#             (n_clusters, n_channels, n_pcs).

#         """

#         tic = time.time()
#         logger.info(' ')
#         logger.info(f'Extracting spikes using templates')
#         logger.info('-'*40)
#         st0, tF, ops = spikedetect.run(
#             ops, bfile, device=device, progress_bar=progress_bar,
#             clear_cache=clear_cache, verbose=verbose
#             )
#         tF = torch.from_numpy(tF)

#         elapsed = time.time() - tic
#         total = time.time() - tic0
#         ops['runtime_st0'] = elapsed
#         ops['usage_st0'] = get_performance()
#         if torch.cuda.is_available() and device == torch.device('cuda'):
#             ops['cuda_st0'] = torch.cuda.memory_stats(device)
#         logger.info(f'{len(st0)} spikes extracted in {elapsed:.2f}s; ' + 
#                     f'total {total:.2f}s')
#         logger.debug(f'st0 shape: {st0.shape}')
#         logger.debug(f'tF shape: {tF.shape}')
#         if len(st0) == 0:
#             raise ValueError('No spikes detected, cannot continue sorting.')
#         log_performance(logger, 'info', 'Resource usage after spike detect (univ)',
#                     reset=True)
#         log_thread_count(logger)    
#         tic = time.time()
#         logger.info(' ')
#         logger.info('First clustering')
#         logger.info('-'*40)
#         clu, Wall = clustering_qr.run(
#             ops, st0, tF, mode='spikes', device=device, progress_bar=progress_bar,
#             clear_cache=clear_cache, verbose=verbose
#             )
#         Wall3 = template_matching.postprocess_templates(
#             Wall, ops, clu, st0, tF, device=device
#             )

#         elapsed = time.time() - tic
#         total = time.time() - tic0
#         ops['runtime_clu0'] = elapsed
#         ops['usage_clu0'] = get_performance()
#         if torch.cuda.is_available() and device == torch.device('cuda'):
#             ops['cuda_clu0'] = torch.cuda.memory_stats(device)
#         logger.info(f'{clu.max()+1} clusters found, in {elapsed:.2f}s; ' +
#                     f'total {total:.2f}s')
#         logger.debug(f'clu shape: {clu.shape}')
#         logger.debug(f'Wall shape: {Wall.shape}')
#         log_performance(logger, 'info', 'Resource usage after first clustering',
#                     reset=True) 
#         log_thread_count(logger)
#         tic = time.time()
#         logger.info(' ')
#         logger.info('Extracting spikes using cluster waveforms')
#         logger.info('-'*40)
#         st, tF, ops = template_matching.extract(
#             ops, bfile, Wall3, device=device, progress_bar=progress_bar
#             )
        
#         log_thread_count(logger)

#         elapsed = time.time() - tic
#         total = time.time() - tic0
#         ops['runtime_st'] = elapsed
#         ops['usage_st'] = get_performance()
#         if torch.cuda.is_available() and device== torch.device('cuda'):
#             ops['cuda_st'] = torch.cuda.memory_stats(device)
#         logger.info(f'{len(st)} spikes extracted in {elapsed:.2f}s; ' +
#                     f'total {total:.2f}s')
#         logger.debug(f'st shape: {st.shape}')
#         logger.debug(f'tF shape: {tF.shape}')
#         logger.debug(f'iCC shape: {ops["iCC"].shape}')
#         logger.debug(f'iU shape: {ops["iU"].shape}')

#         log_cuda_details(logger)
#         log_performance(logger, 'info', 'Resource usage after spike detect (learned)',
#                         reset=True)

#         return st, tF, Wall, clu

#     def forward(
#         self,
#         ops: Dict[str, Any],
#         bfile: io.BinaryFiltered,
#         progress_bar: Optional[Any] = None,
#         tic0=np.nan
#     ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
#         st0,tF, _, _ = self.detect_spikes(
#             ops, self.device, bfile, tic0=tic0, progress_bar=progress_bar,
#             clear_cache=self.clear_cache, verbose=self.verbose
#             )
#         return st0, tF, ops