"""
rtsort_pipeline_blocks.py

A block-based, PyTorch-friendly port of `tutorial_rtsort.ipynb`.

Goal
----
Turn the notebook into a maintainable "pipeline" where:
  * You have discrete Blocks.
  * Each Block calls the next one (chain-of-responsibility).
  * The whole pipeline can be run from a single `RTSortPipeline(...)()` call.

This file intentionally keeps the logic close to the notebook while adding:
  * A config dataclass
  * Logging
  * Safer unit conversions (ms -> samples)
  * Optional numba acceleration for duplicate removal (with a pure-Python fallback)
  * A CLI entry point (so you can run it like a script)

Notebook -> Blocks mapping
--------------------------
Cell 0: load_probe()                            -> LoadProbeBlock
Cell 1: recording_from_bin_and_probe()          -> LoadRecordingBlock
Cell 2: remove_duplicates(), save_kilosort_spikes() -> DeduplicateBlock / SaveKilosortExportBlock
Cell 3-5: detect_sequences(...)                 -> CalibrateRTSortBlock
Cell 6: streaming loop (rt_sort.running_sort)   -> StreamingSortBlock
Cell 7: export spike_times.npy/spike_clusters.npy -> SaveKilosortExportBlock

Dependencies
------------
This script assumes you already have these installed in the environment where you run it:
  - torch
  - numpy
  - scipy (for .mat probe files)
  - spikeinterface (for reading raw binary recordings)
  - torchbci (for RTSort: torchbci.rtsort.rt_sort.detect_sequences)
Optional:
  - numba (for faster duplicate removal)

Typical usage (CLI)
-------------------
python rtsort_pipeline_blocks.py \
  --bin-path path/to/recording.bin \
  --probe-path path/to/chanMap.mat \
  --sampling-frequency-hz 30000 \
  --num-channels-in-file 384 \
  --inter-path ./rt_sort_results/rtsort_cache_myrec \
  --detection-model path/to/detection_models/neuropixels \
  --export-dir ./rt_sort_results/kilosort_export \
  --device cuda

Or use it as a module from Python:
    from rtsort_pipeline_blocks import RTSortConfig, RTSortPipeline
    cfg = RTSortConfig(...)
    pipeline = RTSortPipeline(cfg)
    state = pipeline()

The pipeline returns a `state` dict containing objects and outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

from torchbci.block.base.base_block import BaseBlock
from torchbci.kilosort4.io import load_probe, remove_duplicates


# ---------------------------
# Logging
# ---------------------------

logger = logging.getLogger("rtsort_pipeline")


def setup_logger(level: str = "INFO") -> None:
    """Configure a reasonable default logger."""
    level_num = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_num,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def recording_from_bin_and_probe(
    *,
    bin_path: Union[str, Path],
    probe_path: Union[str, Path],
    sampling_frequency: float,
    num_channels_in_file: int,
    dtype: str = "int16",
    time_axis: int = 0,
):
    """
    Return a SpikeInterface Recording with correct channel order and locations set,
    using a Kilosort chanMap(.mat) / .prb / .json loaded by `load_probe()`.
    """
    import spikeinterface as si  # local import

    probe = load_probe(probe_path)

    # 1) Load raw binary (channels in FILE ORDER 0..num_channels_in_file-1)
    rec = si.read_binary(
        file_paths=str(bin_path),
        sampling_frequency=sampling_frequency,
        num_channels=num_channels_in_file,
        dtype=dtype,
        time_axis=time_axis,
    )

    # 2) Reorder/select channels to match probe['chanMap'] order (connected-only)
    ch_ids = rec.get_channel_ids()
    sel_ids = ch_ids[probe["chanMap"]]
    rec = rec.select_channels(sel_ids)

    # 3) Attach locations as a SpikeInterface property named "location"
    locs = np.c_[probe["xc"], probe["yc"]].astype(np.float32)
    rec.set_property("location", locs)

    # Sanity checks
    assert rec.get_num_channels() == locs.shape[0], (rec.get_num_channels(), locs.shape)
    _ = rec.get_channel_locations()

    return rec


def save_kilosort_spikes(
    spike_trains_ms: Sequence[np.ndarray],
    samp_freq_khz: float,
    out_dir: Union[str, Path],
    *,
    add_start_ms: float = 0.0,
    dtype_times=np.int64,
    dtype_clusters=np.int32,
    remove_dups: bool = True,
    dup_dt: int = 15,
):
    """
    Save Kilosort-like spike_times.npy and spike_clusters.npy.

    Parameters
    ----------
    spike_trains_ms : sequence of 1D arrays
        spike_trains_ms[unit_id] = array of spike times in milliseconds.
    samp_freq_khz : float
        Sampling frequency in kHz (samples per ms). For 30 kHz data, this is 30.0.
    out_dir : str or Path
        Folder to write spike_times.npy and spike_clusters.npy
    add_start_ms : float
        If spike times are relative to a windowed segment, add the window start (ms)
        to make them absolute in the original recording.
    remove_dups : bool
        If True, remove same-cluster spikes that occur within `dup_dt` samples.
    dup_dt : int
        Duplicate window in samples for same-cluster spikes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_times: List[np.ndarray] = []
    all_clusters: List[np.ndarray] = []

    for unit_id, times_ms in enumerate(spike_trains_ms):
        if times_ms is None:
            continue
        times_ms = np.asarray(times_ms)
        if times_ms.size == 0:
            continue

        times_samp = np.round((times_ms + add_start_ms) * samp_freq_khz).astype(dtype_times)
        all_times.append(times_samp)
        all_clusters.append(np.full(times_samp.shape, unit_id, dtype=dtype_clusters))

    if len(all_times) == 0:
        spike_times = np.zeros((0,), dtype=dtype_times)
        spike_clusters = np.zeros((0,), dtype=dtype_clusters)
    else:
        spike_times = np.concatenate(all_times)
        spike_clusters = np.concatenate(all_clusters)

        order = np.argsort(spike_times, kind="mergesort")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]

    if remove_dups and spike_times.size:
        spike_times, spike_clusters, _ = remove_duplicates(
            spike_times.astype(np.int64, copy=False),
            spike_clusters.astype(np.int32, copy=False),
            np.int32(dup_dt),
        )
        spike_times = spike_times.astype(dtype_times, copy=False)
        spike_clusters = spike_clusters.astype(dtype_clusters, copy=False)

    np.save(out_dir / "spike_times.npy", spike_times)
    np.save(out_dir / "spike_clusters.npy", spike_clusters)

    return spike_times, spike_clusters


def ms_to_samples(t_ms: float, samp_freq: float) -> int:
    """
    Convert milliseconds to sample index with a best-effort unit inference.

    RTSort pipelines sometimes store:
      - samp_freq in Hz (samples/second), OR
      - samp_freq in kHz (samples/ms)

    Heuristic:
      - if samp_freq > 1e3 => treat as Hz and divide by 1000 to get samples/ms
      - else treat as kHz already (samples/ms)
    """
    samples_per_ms = samp_freq / 1000.0 if samp_freq > 1_000 else samp_freq
    return int(round(float(t_ms) * samples_per_ms))


# ---------------------------
# Config
# ---------------------------

@dataclass
class RTSortConfig:
    # Inputs
    bin_path: Union[str, Path]
    probe_path: Union[str, Path]
    sampling_frequency_hz: float
    num_channels_in_file: int

    # SpikeInterface read_binary
    dtype: str = "int16"
    time_axis: int = 0

    # RTSort calibration + runtime
    inter_path: Union[str, Path] = "./rt_sort_results/rtsort_cache"
    detection_model: Union[str, Path] = ""
    recording_window_ms: Tuple[float, float] = (0.0, 2.0 * 60_000.0)  # 2 minutes by default
    return_spikes: bool = False
    device: str = "cuda"
    verbose: bool = True

    # Streaming
    stream_start_frame: int = 0
    stream_end_frame: Optional[int] = None  # None => full recording

    # Post-processing / export
    export_dir: Optional[Union[str, Path]] = None
    dtype_times: str = "int64"
    dtype_clusters: str = "int32"
    remove_dups: bool = True
    dup_dt_samples: int = 15

    # Extra params for detect_sequences (defaults to neuropixels_params when available)
    rtsort_params: Dict[str, Any] = field(default_factory=dict)

    def resolved_export_dir(self) -> Path:
        if self.export_dir is not None:
            return Path(self.export_dir).expanduser().resolve()
        return Path(self.inter_path).expanduser().resolve() / "kilosort_export"

    def to_json(self) -> str:
        def _coerce(o: Any) -> Any:
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, tuple):
                return list(o)
            return o

        d = asdict(self)
        return json.dumps({k: _coerce(v) for k, v in d.items()}, indent=2)


# ---------------------------
# Block base class
# ---------------------------

State = Dict[str, Any]


class RTSortBlock(BaseBlock):
    """
    Base class for a chained RT-Sort block.

    Subclasses should implement `run_block(state)` and return the updated `state`.

    If `next_block` is set, this block will call it automatically.
    """

    def __init__(self) -> None:
        super().__init__()
        self.next_block: Optional["RTSortBlock"] = None

    def set_next(self, next_block: "RTSortBlock") -> "RTSortBlock":
        self.next_block = next_block
        return next_block

    def run_block(self, state: State) -> State:
        raise NotImplementedError

    def forward(self, state: Optional[State] = None) -> State:
        state = {} if state is None else state
        state = self.run_block(state)
        if self.next_block is not None:
            return self.next_block(state)
        return state


# ---------------------------
# Blocks
# ---------------------------

class LoadProbeBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        probe = load_probe(self.cfg.probe_path)
        state["probe"] = probe
        logger.info(
            f"Loaded probe from {Path(self.cfg.probe_path).expanduser().resolve()} "
            f"(connected channels: {probe['chanMap'].size}, n_chan_bin: {probe['n_chan']})"
        )
        return state


class LoadRecordingBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        rec = recording_from_bin_and_probe(
            bin_path=self.cfg.bin_path,
            probe_path=self.cfg.probe_path,
            sampling_frequency=self.cfg.sampling_frequency_hz,
            num_channels_in_file=self.cfg.num_channels_in_file,
            dtype=self.cfg.dtype,
            time_axis=self.cfg.time_axis,
        )
        state["recording"] = rec
        state["total_samples"] = int(rec.get_total_samples())
        state["sampling_frequency_hz"] = float(self.cfg.sampling_frequency_hz)
        logger.info(
            f"Loaded recording from {Path(self.cfg.bin_path).expanduser().resolve()} | "
            f"samples={state['total_samples']} | channels={rec.get_num_channels()} | "
            f"fs={self.cfg.sampling_frequency_hz} Hz"
        )
        return state


class CalibrateRTSortBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        from torchbci.rtsort.rt_sort import detect_sequences  # local import

        inter_path = Path(self.cfg.inter_path).expanduser().resolve()
        inter_path.mkdir(parents=True, exist_ok=True)
        state["inter_path"] = inter_path

        params = dict(self.cfg.rtsort_params)
        if not params:
            try:
                from torchbci.rtsort.rt_sort import neuropixels_params  # type: ignore
                params.update(neuropixels_params)
            except Exception:
                logger.warning(
                    "Could not import torchbci.rtsort.rt_sort.neuropixels_params. "
                    "Proceeding with empty params. If detect_sequences requires params, "
                    "provide them via cfg.rtsort_params."
                )

        logger.info(
            f"Calibrating RTSort sequences on window_ms={self.cfg.recording_window_ms} "
            f"using detection_model={self.cfg.detection_model}"
        )

        rt_sort = detect_sequences(
            recording=state["recording"],
            inter_path=inter_path,
            detection_model=str(self.cfg.detection_model),
            recording_window_ms=self.cfg.recording_window_ms,
            return_spikes=self.cfg.return_spikes,
            device=self.cfg.device,
            verbose=self.cfg.verbose,
            **params,
        )
        state["rt_sort"] = rt_sort

        state["buffer_size"] = int(getattr(rt_sort, "buffer_size", 0) or 0)
        state["rt_samp_freq"] = float(
            getattr(rt_sort, "samp_freq", state.get("sampling_frequency_hz", 0.0))
        )

        logger.info(
            "RTSort calibration done | "
            f"buffer_size={state['buffer_size']} | rt_samp_freq={state['rt_samp_freq']}"
        )
        return state


class StreamingSortBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        rec = state["recording"]
        rt_sort = state["rt_sort"]

        if hasattr(rt_sort, "reset"):
            rt_sort.reset()

        buffer_size = int(getattr(rt_sort, "buffer_size", 0) or 0)
        if buffer_size <= 0:
            raise ValueError(
                "RTSort object does not expose a valid `buffer_size`. Cannot stream."
            )

        total_samples = int(state["total_samples"])
        start_frame = int(self.cfg.stream_start_frame)
        end_frame = (
            int(self.cfg.stream_end_frame)
            if self.cfg.stream_end_frame is not None
            else total_samples
        )
        end_frame = min(end_frame, total_samples)

        if start_frame < 0 or start_frame >= end_frame:
            raise ValueError(
                f"Invalid stream range: start={start_frame}, end={end_frame}, total={total_samples}"
            )

        logger.info(f"Streaming sort frames [{start_frame}, {end_frame}) in chunks of {buffer_size}...")

        all_times: List[int] = []
        all_clusters: List[int] = []
        latest_frame = start_frame

        for start in range(start_frame, end_frame, buffer_size):
            end = min(start + buffer_size, end_frame)
            obs = rec.get_traces(start_frame=start, end_frame=end)
            latest_frame = end

            if self.cfg.verbose:
                logger.info(f"Processing frames {start} to {end}...")

            det = rt_sort.running_sort(obs, latest_frame=latest_frame)
            for c, t_ms in det:
                all_times.append(
                    ms_to_samples(
                        float(t_ms),
                        float(getattr(rt_sort, "samp_freq", state["sampling_frequency_hz"])),
                    )
                )
                all_clusters.append(int(c))

        state["all_times"] = np.asarray(all_times, dtype=np.int64)
        state["all_clusters"] = np.asarray(all_clusters, dtype=np.int32)
        logger.info(f"Streaming done | detections={state['all_times'].size}")
        return state


class DeduplicateBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        times = np.asarray(state.get("all_times", np.zeros((0,), dtype=np.int64)))
        clusters = np.asarray(state.get("all_clusters", np.zeros((0,), dtype=np.int32)))

        dtype_times = np.dtype(self.cfg.dtype_times)
        dtype_clusters = np.dtype(self.cfg.dtype_clusters)

        times = times.astype(dtype_times, copy=False)
        clusters = clusters.astype(dtype_clusters, copy=False)

        if self.cfg.remove_dups and times.size:
            logger.info(f"Removing duplicates within dt={self.cfg.dup_dt_samples} samples...")
            times2, clusters2, keep = remove_duplicates(
                times.astype(np.int64, copy=False),
                clusters.astype(np.int32, copy=False),
                np.int32(self.cfg.dup_dt_samples),
            )
            times = times2.astype(dtype_times, copy=False)
            clusters = clusters2.astype(dtype_clusters, copy=False)
            state["dedup_keep_mask"] = keep
            logger.info(f"Dedup complete | kept={times.size} of {state['all_times'].size}")
        else:
            logger.info("Skipping duplicate removal (remove_dups=False or empty spikes).")

        if times.size:
            order = np.argsort(times, kind="mergesort")
            times = times[order]
            clusters = clusters[order]

        state["spike_times"] = times
        state["spike_clusters"] = clusters
        return state


class SaveKilosortExportBlock(RTSortBlock):
    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def run_block(self, state: State) -> State:
        out_dir = self.cfg.resolved_export_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        spike_times = np.asarray(state.get("spike_times", np.zeros((0,), dtype=np.int64)))
        spike_clusters = np.asarray(state.get("spike_clusters", np.zeros((0,), dtype=np.int32)))

        np.save(out_dir / "spike_times.npy", spike_times)
        np.save(out_dir / "spike_clusters.npy", spike_clusters)

        meta = {
            "bin_path": str(Path(self.cfg.bin_path).name),
            "probe_path": str(Path(self.cfg.probe_path).name),
            "sampling_frequency_hz": float(self.cfg.sampling_frequency_hz),
            "num_channels_in_file": int(self.cfg.num_channels_in_file),
            "inter_path": str(Path(self.cfg.inter_path).name),
            "detection_model": str(Path(self.cfg.detection_model).name) if self.cfg.detection_model else "",
            "recording_window_ms": list(self.cfg.recording_window_ms),
            "device": self.cfg.device,
            "n_spikes": int(spike_times.size),
        }
        (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

        state["export_dir"] = out_dir
        logger.info(f"Saved Kilosort export to: {out_dir} | n_spikes={spike_times.size}")
        return state


# ---------------------------
# Pipeline
# ---------------------------

class RTSortPipeline(nn.Module):
    """
    A PyTorch-friendly pipeline that chains Blocks.
    Calling the pipeline runs:
    LoadProbe -> LoadRecording -> CalibrateRTSort -> Stream -> Dedup -> Save
    """

    def __init__(self, cfg: RTSortConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.load_probe = LoadProbeBlock(cfg)
        self.load_recording = LoadRecordingBlock(cfg)
        self.calibrate = CalibrateRTSortBlock(cfg)
        self.stream = StreamingSortBlock(cfg)
        self.dedup = DeduplicateBlock(cfg)
        self.save = SaveKilosortExportBlock(cfg)

        self.entry = self.load_probe
        self.load_probe.set_next(self.load_recording) \
                      .set_next(self.calibrate) \
                      .set_next(self.stream) \
                      .set_next(self.dedup) \
                      .set_next(self.save)

    @torch.inference_mode()
    def forward(self, state: Optional[State] = None) -> State:
        state = {} if state is None else state
        state.setdefault("config_json", self.cfg.to_json())
        return self.entry(state)


# ---------------------------
# CLI
# ---------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Block-based RTSort pipeline (ported from notebook).")

    # Required
    p.add_argument("--bin-path", required=True, type=str, help="Path to raw .bin file")
    p.add_argument("--probe-path", required=True, type=str, help="Path to probe file (.mat/.prb/.json)")
    p.add_argument("--sampling-frequency-hz", required=True, type=float, help="Sampling frequency in Hz")
    p.add_argument("--num-channels-in-file", required=True, type=int, help="Total number of channels in raw binary")

    # Optional read_binary
    p.add_argument("--dtype", default="int16", type=str, help="Binary dtype (default: int16)")
    p.add_argument("--time-axis", default=0, type=int, help="time_axis arg for spikeinterface.read_binary")

    # RTSort calibration/runtime
    p.add_argument("--inter-path", default="./rt_sort_results/rtsort_cache", type=str, help="RTSort cache folder")
    p.add_argument("--detection-model", default="", type=str, help="Path to RTSort detection_model folder")
    p.add_argument("--recording-window-ms", default="0,120000", type=str, help="Calibration window 'start_ms,end_ms'")
    p.add_argument("--device", default="cuda", type=str, help="cuda or cpu")
    p.add_argument("--verbose", action="store_true", help="Verbose progress logging")

    # Streaming
    p.add_argument("--stream-start-frame", default=0, type=int, help="Start frame for streaming")
    p.add_argument("--stream-end-frame", default=None, type=int, help="End frame (exclusive). Omit for full recording.")

    # Export
    p.add_argument("--export-dir", default=None, type=str, help="Output folder (default: inter_path/kilosort_export)")
    p.add_argument("--no-remove-dups", action="store_true", help="Disable duplicate removal")
    p.add_argument("--dup-dt-samples", default=15, type=int, help="Duplicate window (samples) for same cluster")
    p.add_argument("--log-level", default="INFO", type=str, help="Logging level (INFO/DEBUG/...)")

    # Optional params for detect_sequences
    p.add_argument(
        "--rtsort-params-json",
        default=None,
        type=str,
        help="JSON string or path-to-json-file with extra params passed to detect_sequences",
    )

    return p.parse_args()


def _load_rtsort_params(arg: Optional[str]) -> Dict[str, Any]:
    if not arg:
        return {}
    path = Path(arg)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(arg)


def main() -> None:
    args = _parse_args()
    setup_logger(args.log_level)

    parts = [p.strip() for p in args.recording_window_ms.split(",")]
    if len(parts) != 2:
        raise ValueError("--recording-window-ms must be 'start_ms,end_ms'")
    window_ms = (float(parts[0]), float(parts[1]))

    cfg = RTSortConfig(
        bin_path=args.bin_path,
        probe_path=args.probe_path,
        sampling_frequency_hz=float(args.sampling_frequency_hz),
        num_channels_in_file=int(args.num_channels_in_file),
        dtype=args.dtype,
        time_axis=int(args.time_axis),
        inter_path=args.inter_path,
        detection_model=args.detection_model,
        recording_window_ms=window_ms,
        device=args.device,
        verbose=bool(args.verbose),
        stream_start_frame=int(args.stream_start_frame),
        stream_end_frame=args.stream_end_frame,
        export_dir=args.export_dir,
        remove_dups=not bool(args.no_remove_dups),
        dup_dt_samples=int(args.dup_dt_samples),
        rtsort_params=_load_rtsort_params(args.rtsort_params_json),
    )

    logger.info("Running RTSort pipeline with config:\n" + cfg.to_json())

    pipeline = RTSortPipeline(cfg)
    state = pipeline()

    out_dir = state.get("export_dir", None)
    n_spikes = int(np.asarray(state.get("spike_times", np.zeros((0,), dtype=np.int64))).size)
    logger.info(f"Done. Export dir: {out_dir} | n_spikes={n_spikes}")


if __name__ == "__main__":
    main()