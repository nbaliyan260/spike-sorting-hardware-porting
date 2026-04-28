# ks4_blocks.py
from __future__ import annotations
import hashlib
import json
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from torchbci.kilosort4.parameters import DEFAULT_SETTINGS

import numpy as np
import cupy as cp
import torch
import time
import platform
import logging
import random
logger = logging.getLogger("kilosort")

from torch import nn

from torchbci.kilosort4 import io, PROBE_DIR
from torchbci.block.kilosort_blocks.initializeopsblock import InitializeOpsBlock
from torchbci.block.kilosort_blocks.preprocessingblock import PreprocessingBlock
from torchbci.block.kilosort_blocks.driftcorrectionblock import DriftCorrectionBlock
from torchbci.block.kilosort_blocks.spikedetectionblock import SpikeDetectionBlock
from torchbci.block.kilosort_blocks.finalgraphclusteringblock import FinalGraphClusteringBlock
from torchbci.block.kilosort_blocks.mergeblock import MergeBlock
from torchbci.block.kilosort_blocks.saveresultsblock import SaveResultsBlock
from torchbci.kilosort4.utils import (
    log_performance, probe_as_string, ops_as_string, log_sorting_summary, log_thread_count
    )

_DET_STATE = {
    "env": {k: os.environ.get(k) for k in (
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )},
    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "num_threads": torch.get_num_threads(),
    "num_interop_threads": torch.get_num_interop_threads(),
}

def run_once(func):
    """Decorator to run a function only once."""
    has_run = False

    def wrapper(*args, **kwargs):
        nonlocal has_run
        if not has_run:
            has_run = True
            return func(*args, **kwargs)

    return wrapper

# TODO: FIX ISSUE WITH ENFORCING DETERMINISM BY A FLAG
# @run_once
def set_deterministic(seed: int = 1):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) 
    set_seeds(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
# @run_once
def set_semi_deterministic(seed: int = 1):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seeds(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

@run_once
def set_undeterministic(seed: int = 1):
    torch.use_deterministic_algorithms(False)
    torch.backends.cuda.matmul.allow_tf32 = _DET_STATE["allow_tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = _DET_STATE["allow_tf32_cudnn"]
    torch.backends.cudnn.benchmark = _DET_STATE["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = _DET_STATE["cudnn_deterministic"]

    for k, v in _DET_STATE["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    set_seeds(seed)
    try:
        torch.set_num_threads(_DET_STATE["num_threads"])
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(_DET_STATE["num_interop_threads"])
    except RuntimeError:
        pass

def set_seeds(seed: int = 1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    cp.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)

import hashlib
import numpy as np
import torch
import cupy as cp

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
    
class KS4Pipeline(nn.Module):
    """
    Orchestrates all blocks to run the full Kilosort4 pipeline in a single forward pass.

    Stages (exactly the Kilosort4 sequence):
      1) initialize_ops  2) compute_preprocessing  3) compute_drift_correction
      4) spikedetect.run (universal templates)  5) clustering_qr.run (mode='spikes')
      6) template_matching.postprocess_templates → Wall3
      7) template_matching.extract  8) clustering_qr.run (mode='template')
      9) template_matching.merging_function  10) save_sorting (optional)

    Deterministic mode:
        - 0: undeterministic (default)
        - 1: Semi deterministic (allows for non determinisim on big sized datasets)
        - 2: Fully deterministic (Slower)
    """

    def __init__(
        self,
        settings,
        probe,
        results_dir,
        device,
        disable_blocks: Optional[Dict[str, bool]] = None,
        use_jims_filter: bool = False,
        deterministic_mode: int = 0,
        optimize_memory: bool = False,
        seed: int = 1,
        kilosort_version: int = 4
    ):
        super().__init__()
        if deterministic_mode == 2:
            set_deterministic(seed=seed)
        elif deterministic_mode == 1:
            set_semi_deterministic(seed=seed)
        else:
            set_undeterministic(seed=seed)
        if settings is None or settings.get('n_chan_bin', None) is None:
            raise ValueError(
                '`n_chan_bin` is a required setting. This is the total number of '
                'channels in the binary file, which may or may not be equal to the '
                'number of channels specified by the probe.'
                )
        settings = {**DEFAULT_SETTINGS, **settings}
        self.settings = settings
        self.device = device
        self.filename, _data_dir, _results_dir, _probe = self.set_files(settings, None, probe, None, None, results_dir, None, 0)
        self.setup_logger(_results_dir)
        self.results_dir = Path(_results_dir)

        self.init_block = InitializeOpsBlock(self.settings, probe=probe, device=device, kilosort_version=kilosort_version)
        self.preproc_block = PreprocessingBlock(device=device, deterministic_mode=deterministic_mode)
        self.drift_block = DriftCorrectionBlock(device=device, deterministic_mode=deterministic_mode, optimize_memory=optimize_memory, seed=seed)
        self.detect_block = SpikeDetectionBlock(device=device, optimize_memory=optimize_memory, deterministic_mode=deterministic_mode, seed=seed)
        self.final_cluster_block =  FinalGraphClusteringBlock(device=device, optimize_memory=optimize_memory)
        self.merge_block = MergeBlock(device=device, optimize_memory=optimize_memory)
        self.save_block = SaveResultsBlock(results_dir=_results_dir, optimize_memory=optimize_memory) if not disable_blocks or not disable_blocks.get('save', False) else None
        self.disable_blocks = disable_blocks
        self.use_jims_filter = use_jims_filter
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

    def set_files(self, settings, filename, probe, probe_name, data_dir, results_dir,
              bad_channels, shank_idx):
        """Parse file and directory information for data, probe, and results."""

        # Check for filename 
        filename = settings.get('filename', None) if filename is None else filename 

        # Use data_dir if filename not available
        if filename is None:
            data_dir = settings.get('data_dir', None) if data_dir is None else data_dir
            if data_dir is None:
                raise ValueError('no path to data provided, set "data_dir=" or "filename="')
            data_dir = Path(data_dir).resolve()
            if not data_dir.exists():
                raise FileExistsError(f"data_dir '{data_dir}' does not exist")

            # Find binary file in the folder
            filename  = io.find_binary(data_dir=data_dir)
            filename = [filename]
        else:
            if not isinstance(filename, list):
                filename = [filename]
            filename = [Path(f) for f in filename]
            for f in filename:
                if not f.exists():
                    raise FileExistsError(f"filename '{filename}' does not exist")
            data_dir = filename[0].parent
            
        # Convert paths to strings when saving to ops, otherwise ops can only
        # be loaded on the operating system that originally ran the code.
        settings['filename'] = filename
        settings['data_dir'] = data_dir

        # Try to set results_dir based on settings, otherwise use default.
        results_dir = settings.get('results_dir', None) if results_dir is None else results_dir
        results_dir = Path(results_dir).resolve() if results_dir is not None else None
        if results_dir is None:
            results_dir = data_dir / 'kilosort4'

        # Make sure results directory exists
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # find probe configuration file and load
        if probe is None:
            if probe_name is not None:     probe_path = PROBE_DIR / probe_name
            elif 'probe_name' in settings: probe_path = PROBE_DIR / settings['probe_name']
            elif 'probe_path' in settings: probe_path = Path(settings['probe_path']).resolve()
            else: raise ValueError('no probe_name or probe_path provided, set probe_name=')
            if not probe_path.exists():
                raise FileExistsError(f"probe_path '{probe_path}' does not exist")
            
            probe  = io.load_probe(probe_path)
        else:
            # Make sure xc, yc are float32, otherwise there are casting problems
            # with some pytorch functions.
            probe['xc'] = probe['xc'].astype(np.float32)
            probe['yc'] = probe['yc'].astype(np.float32)

        # Let user know if there are too many dimensions in probe entries.
        # Don't want to automatically flatten them incase they've made assumptions
        # about higher-D ordering.
        for k in ['xc', 'yc', 'kcoords', 'chanMap']:
            if probe[k].ndim > 1:
                raise ValueError(f"Array-valued probe entries should have 1 dim, "
                                f"but key: {k} has ndim == {probe[k].ndim}.")

        if bad_channels is not None:
            probe = io.remove_bad_channels(probe, bad_channels)

        return filename, data_dir, results_dir, probe

    def setup_logger(self, results_dir, verbose_console=False):
        results_dir = Path(results_dir)
        
        # Get root logger for Kilosort application
        ks_log = logging.getLogger('kilosort')
        ks_log.setLevel(logging.DEBUG)

        # Add file handler at debug level, include timestamps and logging level
        # in text output.
        file = logging.FileHandler(results_dir / 'kilosort4.log', mode='w')
        file.setLevel(logging.DEBUG)
        text_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        file_formatter = logging.Formatter(text_format)
        file.setFormatter(file_formatter)

        # Skip this if the handlers were already added, like when running multiple
        # times in a single session.
        if not ks_log.handlers:
            # Add console handler at info level with shorter messages,
            # unless verbose is requested.
            console = logging.StreamHandler()
            if verbose_console:
                console.setLevel(logging.DEBUG)
                console.setFormatter(file_formatter)
            else:
                console.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(name)-12s: %(message)s')
                console.setFormatter(console_formatter)
            ks_log.addHandler(console)

        # Always add file handler since log file might change locations
        ks_log.addHandler(file)


    def close_logger(self):
        ks_log = logging.getLogger('kilosort')
        for handler in ks_log.handlers.copy():
            ks_log.removeHandler(handler)
            handler.close()

    def forward(
        self,
        *,
        file_object: Optional[Any] = None,
        progress_bar: Optional[Any] = None,
        skip_save: bool = False,
    ) -> Dict[str, Any]:
        with torch.inference_mode():
            logger.info(f"Kilosort version 4.1.1")
            logger.info(f"Python version {platform.python_version()}")
            logger.info('-'*40)
            logger.info('System information:')
            logger.info(f'{platform.platform()} {platform.machine()}')
            logger.info(platform.processor())
            if torch.cuda.is_available() and self.device == torch.device('cuda'):
                logger.info('Using GPU for PyTorch computations. '
                            'Specify `device` to change this.')
            else:
                logger.info('Using CPU for PyTorch computations. '
                            'Specify `device` to change this.')

            if self.device != torch.device('cpu'):
                memory = torch.cuda.get_device_properties(self.device).total_memory/1024**3
                logger.info(f'Using CUDA device: {torch.cuda.get_device_name()} {memory:.2f}GB')

            logger.info('-'*40)
            if len(self.filename) == 1:
                logger.info(f"Sorting {self.filename}")
            else:
                logger.info(f"Sorting {self.filename[0].parent}/... (multiple files)")

            tic0 = time.time()
            # 1) ops
            ops = self.init_block()
            probe = ops.get("probe", {})
            self.record_hash(
                "init_ops",
                tic0=tic0,
                chanMap=probe.get("chanMap", None),
                xc=probe.get("xc", None),
                yc=probe.get("yc", None),
            )
            # Pretty-print ops and probe for log
            logger.debug(f"Initial ops:\n\n{ops_as_string(ops)}\n")
            logger.debug(f"Probe dictionary:\n\n{probe_as_string(ops['probe'])}\n")

            # Baseline performance metrics
            log_performance(logger, 'info', 'Resource usage before sorting')
            log_thread_count(logger)
            # 2) preprocessing (HPF + whitening)
            ops = self.preproc_block(ops, file_object=file_object, tic0=tic0, use_jims=self.use_jims_filter)
            self.record_hash(
                "preproc",
                tic0=tic0,
                Wrot=ops.get("Wrot", None),
                Wwhite=ops.get("Wwhite", None),
                iC=ops.get("iC", None),
            )
            # 3) drift correction → drift-corrected bfile
            ops, bfile, _st_scatter = self.drift_block(ops, file_object=file_object, progress_bar=progress_bar, tic0=tic0, skip_drift=self.disable_blocks.get('drift', False) if self.disable_blocks else False, use_jims=self.use_jims_filter)
            self.record_hash(
                "drift",
                tic0=tic0,
                dshift=ops.get("dshift", None),
                _st_scatter=_st_scatter,
            )
            log_thread_count(logger)
            # 4) universal-template detection
            # 5) graph clustering (first pass) + template postprocessing
            # 6) re-extract spikes with learned templates
            st, tF, ops, hash_trace = self.detect_block(ops, bfile, progress_bar=progress_bar, tic0=tic0)
            for rec in hash_trace:
                self.hash_trace.append(rec)
            # self.record_hash("detect", tic0=tic0, st=st, tF=tF)

            log_thread_count(logger)
            # 7) final graph clustering
            clu, Wall_merge, ops = self.final_cluster_block(ops, st, tF, progress_bar=progress_bar, tic0=tic0)
            self.record_hash("final_cluster", tic0=tic0, clu=clu, Wall=Wall_merge)

            # 8) merges / cleanups
            Wall, clu, is_ref, st, tF, ops = self.merge_block(ops, Wall_merge, clu, st, tF, tic0=tic0)
            self.record_hash("merge", tic0=tic0, clu=clu, Wall=Wall, is_ref=is_ref)
            ops["hash_trace"] = self.hash_trace
            try:
                (self.results_dir / "hash_trace.json").write_text(json.dumps(self.hash_trace, indent=2))
            except Exception as e:
                logger.warning(f"Could not write hash_trace.json: {e}")
            result = {
                "ops": ops,
                "bfile": bfile,
                "st": st,
                "clu": clu,
                "tF": tF,
                "Wall": Wall,
                "is_ref": is_ref,
                "wall_merge": Wall_merge,
            }
            log_thread_count(logger)

            # 9) optional save
            if (self.save_block is not None) and (not skip_save):
                ops_out, similar_templates, is_ref2, est_contam_rate, kept_spikes = self.save_block(
                    ops, st, clu, tF, Wall, bfile=bfile, tic0=tic0
                )
                result.update(
                    dict(
                        ops_saved=ops_out,
                        similar_templates=similar_templates,
                        is_ref_saved=is_ref2,
                        est_contam_rate=est_contam_rate,
                        kept_spikes=kept_spikes,
                    )
                )
            if torch.cuda.is_available() and self.device == torch.device('cuda'):
                ops['cuda_postproc'] = torch.cuda.memory_stats(self.device)
            log_thread_count(logger)
            logger.info('Sorting finished.')
            log_sorting_summary(ops, log=logger, level='info')
            self.close_logger()
            return result
