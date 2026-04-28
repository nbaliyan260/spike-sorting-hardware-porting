"""Compare algorithm spike times against ground truth spikes."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import scipy
import pandas as pd

@dataclass(frozen=True)
class ComparisonResult:
    template_index: int
    matched_indices: np.ndarray
    num_matched: int
    tp: int
    fp: int
    fn: int
    accuracy: float
    recall: float
    precision: float
    false_discovery: float


def count_matching_events(times1, times2, delta: int = 10, record_matches : bool = False, false_neg = False):
    """
    Counts matching events.

    Parameters
    ----------
    times1 : list
        List of spike train 1 frames
    times2 : list
        List of spike train 2 frames
    delta : int
        Number of frames for considering matching events

    Returns
    -------
    matching_count : int
        Number of matching events
    """

    t1_len = times1.shape[0]
    times_concat = np.concatenate((times1, times2))
    membership_i = np.array(range(times_concat.shape[0]))
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    mi_sorted = membership_i[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
    if len(inds) == 0:
        return None, 0
    inds_adj = np.array(sorted(np.concatenate([inds[membership_sorted[inds] == 2] + 1, inds[membership_sorted[inds] == 1]])))
    inds_template = np.array(sorted(np.concatenate([inds[membership_sorted[inds] == 1] + 1, inds[membership_sorted[inds] == 2]])))
    fn_inds = list(set(range(times1.shape[0], times_concat.shape[0])) - set(inds_template))
    if false_neg:
        remove_dups = list(set(times_concat_sorted[fn_inds]))
        return np.array(remove_dups), len(remove_dups)
    if record_matches:
        remove_dups = list(set(mi_sorted[inds_adj]))
        return np.array(remove_dups), len(remove_dups)
    else:
        return len(inds_adj)


def get_false_negatives(
    true_sp_triggers: np.ndarray, sorting_res: np.ndarray, delta_frames: int
) -> Tuple[np.ndarray, int]:
    return count_matching_events(true_sp_triggers, sorting_res, delta_frames, false_neg=True)


def count_match_spikes(
    times1: np.ndarray, all_times2: Sequence[np.ndarray], delta_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes matching spikes between one spike train and a list of others.
    """
    matching_event_counts = np.zeros(len(all_times2), dtype="int64")
    matching_events = np.zeros(len(all_times2), dtype=object)
    for template, times2 in enumerate(all_times2):
        matches, num_matches = count_matching_events(
            times1, times2, delta=delta_frames, record_matches=True
        )
        matching_event_counts[template] = num_matches
        matching_events[template] = np.array(matches)
    return matching_event_counts, matching_events

def get_matching(ks_res, true_sp_trig, delta):
    '''
    ks res: ks st data set by template
    true_sp_trig: true spike triggers
    delta: max num windows for which a match will be found
    '''
    # creating new st by template
        
    counts, inds = count_match_spikes(np.array(true_sp_trig), ks_res, delta)
    
    t = np.argmax(counts)
    print(f'best template: {t}')
    matches, num = count_matching_events(np.array(true_sp_trig), ks_res[t], delta, True)
    if matches is None:
        matches = np.array([], dtype=int)
        num = 0
    return t, np.asarray(matches, dtype=int), int(num)


def create_st_by_template(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    *,
    n_templates: Optional[int] = None,
    time_scale: float = 1.0,
    remap_clusters: bool = True,
    pc_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build per-template spike times.
    """
    st = np.asarray(spike_times)
    clu = np.asarray(spike_clusters)
    # if remap_clusters:
    #     unique = np.unique(clu)
    #     mapping = {cid: i for i, cid in enumerate(unique)}
    #     remapped = np.vectorize(mapping.get)(clu)
    #     clu = remapped
    #     if n_templates is None:
    #         n_templates = len(unique)
    if n_templates is None:
        n_templates = int(clu.max()) + 1 if clu.size else 0

    st_by_template = np.zeros(n_templates, dtype=object)
    pc_features_by_template = np.zeros(n_templates, dtype=object)
    for t in range(n_templates):
        which_spikes = np.flatnonzero(clu == t)
        # st_by_template[t] = st[which_spikes] * time_scale
        st_by_template[t] = st[which_spikes]
        if pc_features is not None:
            pc_features_by_template[t] = pc_features[which_spikes]
    return st_by_template, pc_features_by_template


def compare_to_ground_truth(
    true_spike_times: np.ndarray,
    st_by_template: Sequence[np.ndarray],
    delta_frames: int,
    *,
    template_index: Optional[int] = None,
    pc_features_by_template: Optional[Sequence[np.ndarray]] = None,
) -> ComparisonResult:
    """
    Compare per-template spikes against ground truth and compute metrics.
    """
    true_spike_times = np.asarray(true_spike_times)
    gt_match_idx = np.array([], dtype=int)
    num_matched_gt = 0
    if template_index is None:
        template_index, gt_match_idx, num_matched_gt = get_matching(
            st_by_template, true_spike_times, delta_frames
        )
    else:
        if template_index < 0 or template_index >= len(st_by_template):
            raise IndexError(
                f"template_index {template_index} is out of range for {len(st_by_template)} templates"
            )
        gt_match_idx, num_matched_gt = count_matching_events(
            true_spike_times, st_by_template[template_index], delta_frames, record_matches=True
        )
        if gt_match_idx is None:
            gt_match_idx = np.array([], dtype=int)
            num_matched_gt = 0

    template_match_idx = np.array([], dtype=int)
    selected_template_spikes = np.array([], dtype=float)
    selected_template_features = np.array([], dtype=float)
    if template_index >= 0:
        selected_template_spikes = np.asarray(st_by_template[template_index])
        template_match_idx, _ = count_matching_events(
            selected_template_spikes, true_spike_times, delta_frames, record_matches=True
        )
        if template_match_idx is None:
            template_match_idx = np.array([], dtype=int)
        else:
            template_match_idx = np.asarray(template_match_idx, dtype=int)
        if pc_features_by_template is not None:
            selected_template_features = np.asarray(pc_features_by_template[template_index])

    matched_spike_times = (
        selected_template_spikes[template_match_idx]
        if template_index >= 0
        else np.array([], dtype=float)
    )
    non_matched_spike_times = (
        np.setdiff1d(selected_template_spikes, matched_spike_times)
        if template_index >= 0
        else np.array([], dtype=float)
    )
    print(f"Template spikes: {len(selected_template_spikes)}, Matched: {len(matched_spike_times)}, Non-matched: {len(non_matched_spike_times)}")
    # matched_spike_features = (
    #     selected_template_features[template_match_idx]
    #     if (template_index >= 0 and pc_features_by_template is not None)
    #     else np.array([], dtype=float)
    # )
    # non_matched_spike_features = (
    #     selected_template_features[
    #         np.setdiff1d(np.arange(len(selected_template_spikes)), template_match_idx)
    #     ]
    #     if (template_index >= 0 and pc_features_by_template is not None)
    #     else np.array([], dtype=float)
    # )
    print(len(st_by_template[template_index]), len(matched_spike_times), len(non_matched_spike_times))
    np.save(f'matched_spike_times_template_{template_index}.npy', matched_spike_times)
    np.save(f'non_matched_spike_times_template_{template_index}.npy', non_matched_spike_times)
    # np.save(f'matched_spike_features_template_{template_index}.npy', matched_spike_features)
    # np.save(f'non_matched_spike_features_template_{template_index}.npy', non_matched_spike_features)
    
    tp = int(num_matched_gt)
    fp = int(len(selected_template_spikes) - tp) if template_index >= 0 else 0
    fn = int(len(true_spike_times) - tp) if template_index >= 0 else len(true_spike_times)
    denom = tp + fn + fp

    accuracy = (tp / denom) if denom else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    false_discovery = (fp / (tp + fp)) if (tp + fp) else 0.0

    return ComparisonResult(
        template_index=template_index,
        matched_indices=np.asarray(template_match_idx, dtype=int),
        num_matched=tp,
        tp=tp,
        fp=fp,
        fn=fn,
        accuracy=accuracy,
        recall=recall,
        precision=precision,
        false_discovery=false_discovery,
    )


def _load_npy_1d(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array in {path}, got shape {arr.shape}")
    return arr

def _load_npy_3d(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {arr.shape}")
    return arr

def _append_results_csv(path: str, row: dict) -> None:
    results_path = Path(path)
    df_new = pd.DataFrame([row])
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(results_path, index=False)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compare algorithm spike times to ground truth spikes."
    )
    # parser.add_argument("--patch-filename", required=False, help="Path to patch file ground truth .bin")
    parser.add_argument("--filename", required=False, help="Path to extracellular file .npy")
    parser.add_argument("--spike-times", required=True, help="Path to spike_times.npy")
    parser.add_argument("--spike-clusters", required=True, help="Path to spike_clusters.npy")
    parser.add_argument("--delta", type=int, default=30, help="Match window in frames")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=None,
        help=(
            "Scale factor applied to spike_times (e.g. ground_truth_fs / spike_fs). "
            "If set, --spike-fs/--ground-truth-fs are ignored."
        ),
    )
    parser.add_argument(
        "--spike-fs",
        type=float,
        default=None,
        help="Sampling rate (Hz) of spike_times.npy (dataset rate).",
    )
    parser.add_argument(
        "--ground-truth-fs",
        type=float,
        default=None,
        help=(
            "Sampling rate (Hz) of ground-truth spikes. Defaults to --spike-fs "
            "if not provided."
        ),
    )
    parser.add_argument(
        "--template-index",
        type=int,
        default=None,
        help="Template index to score (default: best matching template)",
    )
    parser.add_argument(
        "--no-remap-clusters",
        action="store_true",
        help="Do not remap cluster ids to 0..N-1",
    )
    parser.add_argument(
        "--result-label",
        required=True,
        help="Label for this experiment row in results.csv",
    )

    args = parser.parse_args()
    if args.time_scale is not None:
        if args.spike_fs is not None or args.ground_truth_fs is not None:
            parser.error("Use --time-scale or --spike-fs/--ground-truth-fs, not both.")
        time_scale = args.time_scale
    else:
        if args.spike_fs is None:
            parser.error("Provide --spike-fs (dataset sampling rate) or --time-scale.")
        ground_truth_fs = args.ground_truth_fs or args.spike_fs
        time_scale =  args.spike_fs / ground_truth_fs

    # from https://spikeinterface.github.io/blog/marques-smith-neuropixel-384ch-paired-recording/
    def detect_peak_on_patch_sig(patch_sig, sample_rate):
        # filter because some traces have drift
        sos = scipy.signal.iirfilter(5, 200./sample_rate*2, analog=False, btype = 'highpass', ftype = 'butter', output = 'sos')
        patch_sig_f = scipy.signal.sosfiltfilt(sos, patch_sig, axis=0)
        
        med = np.median(patch_sig_f)
        mad = np.median(np.abs(patch_sig_f-med))*1.4826
        thresh = med - 12 * mad
        
        # 1 ms aounrd peak
        d = int(sample_rate * 0.001)
        spike_indexes, prop = scipy.signal.find_peaks(-patch_sig_f, height=-thresh, distance=d)
        
        return spike_indexes


    def loadPatchRawData(patch_path):
        patch_recording = np.fromfile(patch_path, dtype='float64')
        return patch_recording
    
    # if args.patch_filename is not None:
    #     patch_filename = args.patch_filename

    #     # loading juxta data
    #     juxta = loadPatchRawData(patch_filename)

    #     sample_rate = args.spike_fs

    #     spike_triggers = detect_peak_on_patch_sig(juxta, sample_rate)
    #     # true_spikes = [s for s in spike_triggers if s <= 600 * sample_rate]
    #     true_spikes = [s for s in spike_triggers]
    if args.filename is not None:
        gt = np.load(args.filename)
        true_spikes = [s for s in gt]
        print(f"Loaded {len(true_spikes)} ")
    else:
        raise ValueError("Either --patch-filename or --filename must be provided")

    # true_spikes = _load_npy_1d(args.ground_truth)
    st = _load_npy_1d(args.spike_times)
    clu = _load_npy_1d(args.spike_clusters).astype(int)
    # Shape (n_spikes, n_pcs, nearest_chans)
    # pc_features = _load_npy_3d(args.spike_times.replace("spike_times.npy", "pc_features.npy"))
    st_by_template, pc_features_by_template = create_st_by_template(
        st,
        clu,
        time_scale=time_scale,
        remap_clusters=not args.no_remap_clusters,
        pc_features=None,
    )

    result = compare_to_ground_truth(
        true_spikes,
        st_by_template,
        args.delta,
        template_index=args.template_index,
        pc_features_by_template=pc_features_by_template,
    )

    print("template_index:", result.template_index)
    print("num_matched:", result.num_matched)
    print("tp:", result.tp)
    print("fp:", result.fp)
    print("fn:", result.fn)
    print("accuracy:", result.accuracy)
    print("recall:", result.recall)
    print("precision:", result.precision)
    print("false_discovery:", result.false_discovery)

    row = {
        "result_label": args.result_label,
        "GT_Spikes_count": len(true_spikes),
        "num_matched": result.num_matched,
        "tp": result.tp,
        "fp": result.fp,
        "fn": result.fn,
        "accuracy": result.accuracy,
        "recall": result.recall,
        "precision": result.precision,
        "false_discovery": result.false_discovery,
    }
    _append_results_csv("results_c14.csv", row)
    print("Saved results to results_c46_ks3.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
