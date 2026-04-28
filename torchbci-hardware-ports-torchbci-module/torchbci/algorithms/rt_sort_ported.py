import json
from pathlib import Path
import pickle
import time
from torchbci.block.rtsort_blocks.initialize import LoadRecordingBlock
from torchbci.block.rtsort_blocks.spikedetector import DetectRTSortSequences
from torchbci.block.rtsort_blocks.onlinestream import StreamingSortBlock
from torchbci.block.rtsort_blocks.save import SaveRTBlock
from torch import nn
import torch
import logging
import os
import numpy as np
import random
import cupy as cp

logger = logging.getLogger("rtsort_pipeline")
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

# @run_once
def set_deterministic(seed: int = 1):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
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

class RTSortPipeline(nn.Module):
    """
    A PyTorch-friendly pipeline that chains Blocks.
    Calling the pipeline runs: LoadProbe → LoadRecording → CalibrateRTSort → Stream → Dedup → Save
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.seed = cfg.get("seed", 1)
        self.deterministic_mode = cfg.get("deterministic_mode", 0)
        print(f"Running RTSortPipeline with seed {self.seed} and deterministic_mode {self.deterministic_mode}")
        if self.deterministic_mode == 2:
            set_deterministic(seed=self.seed)
        elif self.deterministic_mode == 1:
            set_semi_deterministic(seed=self.seed)
        else:
            set_undeterministic(seed=self.seed)
        self.load_recording = LoadRecordingBlock()
        self.detect_sequences = DetectRTSortSequences()
        self.stream = StreamingSortBlock()
        self.save_block = SaveRTBlock()
        self.bin_path = cfg["bin_path"] 
        self.probe_path = cfg["probe_path"]
        self.sampling_frequency = cfg["sampling_frequency"]
        self.num_channels = cfg["num_channels"]
        self.dtype = cfg["dtype"]
        self.time_axis = cfg["time_axis"]
        self.inter_path = cfg["inter_path"]
        self.reference_inter_path = cfg.get("reference_inter_path", self.inter_path)
        self.detection_model = cfg["detection_model"]
        self.recording_window_ms = cfg["recording_window_ms"]
        self.return_spikes = cfg["return_spikes"]
        self.device = cfg["device"]
        self.verbose = cfg["verbose"]
        self.out_dir = cfg["out_dir"]
        self.detect_initial_sequences = cfg["detect_initial_sequences"]
        self.stream_parallel_workers = int(cfg.get("stream_parallel_workers") or 1)
        self.stream_parallel_segment_frames = cfg.get("stream_parallel_segment_frames")
        self.stream_parallel_overlap_frames = cfg.get("stream_parallel_overlap_frames")
        self.optimize_memory = cfg.get("optimize_memory", False)
        

    def setup_logger(level: str = "INFO") -> None:
        """Configure a reasonable default logger."""
        # level_num = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    @torch.inference_mode()
    def forward(self):
        if self.verbose:
            self.setup_logger()
        tic0 = time.time()
        rec, elapsed_loading = self.load_recording(self.bin_path, self.probe_path, self.sampling_frequency, self.num_channels, self.dtype, self.time_axis)
        if self.detect_initial_sequences:
            rt_sort, elapsed_detection = self.detect_sequences(self.inter_path, rec, self.detection_model, self.recording_window_ms, return_spikes=self.return_spikes, device=self.device, verbose=self.verbose, optimize_memory=self.optimize_memory, deterministic_mode=self.deterministic_mode)
        else:
            print(f"Skipping initial sequence detection and loading from reference inter path: {self.reference_inter_path}")
            pkl_path = Path(self.reference_inter_path) / "rt_sort.pickle"
            with pkl_path.open("rb") as f:
                rt_sort = pickle.load(f)
            elapsed_detection = 0.0 

        rt_sort.deterministic_mode = self.deterministic_mode
        if self.deterministic_mode == 2:
            # Strict mode favors exact repeatability over throughput: run the
            # full model path on CPU + float32.
            # rt_sort.to_device("cpu")
            # rt_sort.to_dtype(torch.float32)
            model_inter_path = self.inter_path if self.detect_initial_sequences else self.reference_inter_path
            rt_sort.set_model(self.detection_model, model_inter_path=model_inter_path)
        elif not self.detect_initial_sequences:
            rt_sort.set_model(self.detection_model, model_inter_path=self.reference_inter_path)
        all_spike_times, all_spike_clusters, elapsed_streaming = self.stream(
            rec,
            rt_sort,
            verbose=self.verbose,
            num_workers=self.stream_parallel_workers,
            segment_frames=self.stream_parallel_segment_frames,
            right_overlap_frames=self.stream_parallel_overlap_frames,
            deterministic_mode=self.deterministic_mode,
        )
        spike_times, spike_clusters, elapsed_saving = self.save_block(all_spike_times, all_spike_clusters, self.out_dir)
        tic_final = time.time()
        meta = {
            "bin_path": str(Path(self.bin_path).expanduser().resolve()),
            "probe_path": str(Path(self.probe_path).expanduser().resolve()),
            "sampling_frequency_hz": float(self.sampling_frequency),
            "num_channels_in_file": int(self.num_channels),
            "inter_path": str(Path(self.inter_path).expanduser().resolve()),
            "detection_model": str(self.detection_model),
            "recording_window_ms": list(self.recording_window_ms),
            "device": self.device,
            "optimize_memory": bool(self.optimize_memory),
            "stream_parallel_workers": self.stream_parallel_workers,
            "stream_parallel_segment_frames": self.stream_parallel_segment_frames,
            "stream_parallel_overlap_frames": self.stream_parallel_overlap_frames,
            "n_spikes": int(spike_times.size),
            "total_time_s": tic_final - tic0,
            "loading_time_s": elapsed_loading,
            "detection_time_s": elapsed_detection,
            "streaming_time_s": elapsed_streaming,
            "saving_time_s": elapsed_saving,
        }
        (Path(self.out_dir) / "export_meta.json").write_text(json.dumps(meta, indent=2))
        return spike_times, spike_clusters
