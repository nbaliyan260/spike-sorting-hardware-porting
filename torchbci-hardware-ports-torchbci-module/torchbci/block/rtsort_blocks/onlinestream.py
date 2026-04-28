import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn
import torch

logger = logging.getLogger("rtsort_pipeline")


@dataclass(frozen=True)
class _StreamSegment:
    idx: int
    primary_start: int
    primary_stop: int
    warmup_start: int
    worker_stop: int


class StreamingSortBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def ms_to_samples(self, t_ms: float, samp_freq: float) -> int:
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

    def _clone_rt_sort(self, rt_sort):
        """
        Clone RTSort state for a worker while reusing the same inference model.

        The sorter state is mutable across batches, but the model is used only for
        read-only inference. Sharing the model avoids copying large compiled
        modules for every worker.
        """
        model = getattr(rt_sort, "model", None)
        if model is None:
            raise RuntimeError(
                "Parallel RTSort streaming requires rt_sort.model to be initialized."
            )

        # Reuse the inference module directly instead of clearing it from the
        # shared sorter, which is unsafe once multiple worker threads clone the
        # same rt_sort instance concurrently.
        worker_sort = deepcopy(rt_sort, {id(model): model})
        worker_sort.model = model
        worker_sort.reset()
        return worker_sort

    def _run_serial(
        self,
        rec,
        rt_sort,
        verbose=True,
        deterministic_mode=0,
    ):
        rt_sort.reset()
        buffer_size = rt_sort.buffer_size
        total_samples = rec.get_total_samples()
        all_times = []
        all_clusters = []

        logger.info(
            f"Streaming sort frames 0 to {total_samples} with buffer_size={buffer_size}..."
        )

        for start in range(0, total_samples, buffer_size):
            end = min(start + buffer_size, total_samples)
            obs = rec.get_traces(start_frame=start, end_frame=end)

            if verbose:
                logger.info(f"Processing frames {start} to {end}...")
            det = rt_sort.running_sort(obs, latest_frame=end)
            for c, t_ms in det:
                all_times.append(self.ms_to_samples(t_ms, rt_sort.samp_freq))
                all_clusters.append(int(c))

        return all_times, all_clusters

    def _warmup_serial_kernel(self, rec, rt_sort):
        """
        Run and discard one warm-up pass so deterministic_mode runs start from
        the same warmed execution state.
        """
        total_samples = int(rec.get_total_samples())
        if total_samples <= 0:
            return
        warmup_end = min(int(rt_sort.buffer_size) * 3, total_samples)
        rt_sort.reset()
        for start in range(0, warmup_end, int(rt_sort.buffer_size)):
            end = min(start + int(rt_sort.buffer_size), warmup_end)
            obs = rec.get_traces(start_frame=start, end_frame=end)
            _ = rt_sort.running_sort(obs, latest_frame=end)
        rt_sort.reset()

    def _resolve_parallel_layout(
        self,
        total_samples: int,
        rt_sort,
        num_workers: int,
        segment_frames=None,
        right_overlap_frames=None,
    ):
        buffer_size = int(rt_sort.buffer_size)
        warmup_frames = int(getattr(rt_sort, "total_num_pre_median_frames", 0) or 0)
        input_size = int(getattr(rt_sort, "input_size", 0) or 0)
        end_buffer = int(getattr(rt_sort, "end_buffer", 0) or 0)
        seq_n_after = int(getattr(rt_sort, "seq_n_after", 0) or 0)

        if warmup_frames <= 0:
            raise ValueError(
                "Parallel RTSort streaming requires `total_num_pre_median_frames`."
            )
        if input_size > warmup_frames:
            raise ValueError(
                "Parallel RTSort streaming requires input_size <= total_num_pre_median_frames."
            )

        alignment = math.lcm(buffer_size, warmup_frames)

        if segment_frames is None:
            target_segments = max(num_workers * 2, 1)
            segment_frames = math.ceil(total_samples / target_segments / alignment) * alignment
        else:
            segment_frames = int(segment_frames)

        segment_frames = max(segment_frames, alignment)
        segment_frames = math.ceil(segment_frames / alignment) * alignment

        if right_overlap_frames is None:
            # Each segment needs enough future context to emit detections near
            # its right edge before we discard the overlap.
            right_overlap_frames = max(input_size, buffer_size + end_buffer + seq_n_after)
        else:
            right_overlap_frames = int(right_overlap_frames)

        right_overlap_frames = max(right_overlap_frames, input_size)
        right_overlap_frames = math.ceil(right_overlap_frames / buffer_size) * buffer_size

        segments = []
        for idx, primary_start in enumerate(range(0, total_samples, segment_frames)):
            primary_stop = min(primary_start + segment_frames, total_samples)
            warmup_start = max(0, primary_start - warmup_frames)
            worker_stop = min(total_samples, primary_stop + right_overlap_frames)
            segments.append(
                _StreamSegment(
                    idx=idx,
                    primary_start=primary_start,
                    primary_stop=primary_stop,
                    warmup_start=warmup_start,
                    worker_stop=worker_stop,
                )
            )
        return segments, right_overlap_frames

    def _run_parallel_segment(self, rec, rt_sort, segment: _StreamSegment, verbose=False):
        worker_sort = self._clone_rt_sort(rt_sort)
        buffer_size = int(worker_sort.buffer_size)
        all_times = []
        all_clusters = []

        if verbose:
            logger.info(
                "Parallel segment %s | primary=[%s, %s) | context=[%s, %s)",
                segment.idx,
                segment.primary_start,
                segment.primary_stop,
                segment.warmup_start,
                segment.worker_stop,
            )

        for start in range(segment.warmup_start, segment.worker_stop, buffer_size):
            end = min(start + buffer_size, segment.worker_stop)
            obs = rec.get_traces(start_frame=start, end_frame=end)
            det = worker_sort.running_sort(obs, latest_frame=end)

            for c, t_ms in det:
                sample = self.ms_to_samples(t_ms, worker_sort.samp_freq)
                if segment.primary_start <= sample < segment.primary_stop:
                    all_times.append(sample)
                    all_clusters.append(int(c))

        return segment.idx, all_times, all_clusters

    def forward(
        self,
        rec,
        rt_sort,
        verbose=True,
        num_workers=1,
        segment_frames=None,
        right_overlap_frames=None,
        deterministic_mode=0,
        model_traces_path=None,
        model_outputs_path=None,
    ):
        tic = time.time()
        buffer_size = rt_sort.buffer_size
        if buffer_size <= 0:
            raise ValueError(
                "Cannot stream buffer size <= 0."
            )

        num_workers = max(1, int(num_workers))
        if deterministic_mode == 2 and num_workers > 1:
            logger.info(
                "deterministic_mode=2 forces serial RTSort streaming; overriding num_workers=%s to 1.",
                num_workers,
            )
            num_workers = 1

        if num_workers == 1:
            if deterministic_mode == 2:
                self._warmup_serial_kernel(rec, rt_sort)
            all_times, all_clusters = self._run_serial(
                rec,
                rt_sort,
                verbose=verbose,
                deterministic_mode=deterministic_mode,
            )
        else:
            if str(getattr(rt_sort, "device", "")).lower().startswith("cuda"):
                logger.warning(
                    "Parallel RTSort streaming shares one model instance across worker threads; "
                    "CUDA or TensorRT execution may serialize or fail to scale."
                )
            segments, resolved_overlap = self._resolve_parallel_layout(
                rec.get_total_samples(),
                rt_sort,
                num_workers,
                segment_frames=segment_frames,
                right_overlap_frames=right_overlap_frames,
            )
            if len(segments) <= 1:
                all_times, all_clusters = self._run_serial(rec, rt_sort, verbose=verbose)
            else:
                logger.info(
                    "Parallel streaming sort across %s segments with %s workers "
                    "(segment_frames=%s, overlap_frames=%s)...",
                    len(segments),
                    min(num_workers, len(segments)),
                    segments[0].primary_stop - segments[0].primary_start,
                    resolved_overlap,
                )

                results = [None] * len(segments)
                with ThreadPoolExecutor(max_workers=min(num_workers, len(segments))) as pool:
                    futures = [
                        pool.submit(
                            self._run_parallel_segment,
                            rec,
                            rt_sort,
                            segment,
                            verbose,
                        )
                        for segment in segments
                    ]
                    for future in futures:
                        idx, seg_times, seg_clusters = future.result()
                        results[idx] = (seg_times, seg_clusters)

                all_times = []
                all_clusters = []
                for seg_times, seg_clusters in results:
                    all_times.extend(seg_times)
                    all_clusters.extend(seg_clusters)

        elapsed = time.time() - tic
        logger.info(f"Streaming done | detections={len(all_times)}")
        return all_times, all_clusters, elapsed
