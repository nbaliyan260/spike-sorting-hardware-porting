import copy
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from torchbci.algorithms.kilosort_paper_attempt import KS4Pipeline


REFERENCE_DIR = Path(
    os.environ.get(
        "KS4_REFERENCE_DIR",
        "path/to/results_tests",
    )
)
REFERENCE_OPS_PATH = REFERENCE_DIR / "ops.npy"

if not REFERENCE_OPS_PATH.exists():
    pytest.skip(
        f"Reference ops not found at {REFERENCE_OPS_PATH}; set KS4_REFERENCE_DIR to a valid run.",
        allow_module_level=True,
    )


def _to_numpy(value: np.ndarray | torch.Tensor):
    """Convert torch tensors to cpu numpy arrays for comparisons."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _set_block_seeds():
    """Mirror the seeding used by the pipeline forward method."""
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
        torch.random.manual_seed(1)


@pytest.fixture(scope="session")
def reference_ops():
    ops = np.load(REFERENCE_OPS_PATH, allow_pickle=True).item()
    data_files = ops["settings"]["filename"]
    for fname in data_files:
        if not Path(fname).exists():
            pytest.skip(
                f"Required data file missing for reference comparison: {fname}",
                allow_module_level=True,
            )
    return ops


@pytest.fixture(scope="session")
def device(reference_ops):
    requested = reference_ops.get("torch_device", "cpu")
    requested_device = os.environ.get("KS4_TEST_DEVICE", requested)
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip(
                "CUDA not available; cannot reproduce reference CUDA run.",
                allow_module_level=True,
            )
        return torch.device("cuda")
    return torch.device(requested_device)


@pytest.fixture(scope="session")
def ks4_outputs(tmp_path_factory, reference_ops, device):
    """
    Run the KS4 pipeline once and snapshot outputs after each block.

    A temporary results directory is used to avoid mutating the reference run.
    """
    # mktemp only takes (basename, numbered=True/False)
    results_dir = tmp_path_factory.mktemp("ks4_outputs")
    results_dir = Path(results_dir)  # optional, if you prefer pathlib

    print(f"Running KS4 pipeline for tests: {reference_ops['settings']}")
    print(f"Using probe: {reference_ops['probe']}")
    settings = dict(reference_ops["settings"])
    settings["results_dir"] = str(results_dir)
    settings["filename"] = [str(Path(f)) for f in settings["filename"]]

    _set_block_seeds()
    pipeline = KS4Pipeline(
        settings=settings,
        probe=reference_ops["probe"],
        results_dir=results_dir,
        device=device,
    )

    with torch.inference_mode():
        tic0 = time.time()
        ops_init = pipeline.init_block()
        ops_after_init = copy.deepcopy(ops_init)

        ops_preproc = pipeline.preproc_block(ops_init, tic0=tic0)
        ops_after_preproc = copy.deepcopy(ops_preproc)

        ops_drift, bfile, _st_scatter = pipeline.drift_block(
            ops_preproc, progress_bar=None, tic0=tic0
        )
        ops_after_drift = copy.deepcopy(ops_drift)

        st, tF, ops_detect = pipeline.detect_block(
            ops_drift, bfile, progress_bar=None, tic0=tic0
        )
        detection_snapshot = {
            k: copy.deepcopy(ops_detect[k])
            for k in ["iC", "iC2", "iCC", "iCC_mask", "iU"]
        }

        clu, Wall, ops_cluster = pipeline.final_cluster_block(
            ops_detect, st, tF, progress_bar=None, tic0=tic0
        )
        Wall_m, clu_m, is_ref, st_m, tF_m, ops_merge = pipeline.merge_block(
            ops_cluster, Wall, clu, st, tF, tic0=tic0
        )

        ops_saved, similar_templates, is_ref_saved, est_contam_rate, kept_spikes = (
            pipeline.save_block(
                ops_merge, st_m, clu_m, tF_m, Wall_m, bfile=bfile, tic0=tic0
            )
        )

        del st, tF, Wall, clu
        del st_m, tF_m, Wall_m, clu_m

    yield {
        "results_dir": Path(results_dir),
        "ops_init": ops_after_init,
        "ops_preproc": ops_after_preproc,
        "ops_drift": ops_after_drift,
        "detect_ops": detection_snapshot,
        "ops_saved": ops_saved,
        "similar_templates": similar_templates,
        "is_ref": is_ref_saved,
        "kept_spikes": kept_spikes,
        "est_contam_rate": est_contam_rate,
    }

    shutil.rmtree(results_dir, ignore_errors=True)



def test_initialize_ops_matches_reference(ks4_outputs, reference_ops):
    ops_init = ks4_outputs["ops_init"]
    ref = reference_ops

    assert ops_init["NTbuff"] == ref["NTbuff"]
    assert ops_init["Nchan"] == ref["Nchan"]
    assert ops_init["n_chan_bin"] == ref["n_chan_bin"]
    assert ops_init["duplicate_spike_bins"] == ref["duplicate_spike_bins"]
    np.testing.assert_array_equal(ops_init["chanMap"], ref["chanMap"])
    np.testing.assert_allclose(ops_init["xc"], ref["xc"])
    np.testing.assert_allclose(ops_init["yc"], ref["yc"])
    np.testing.assert_array_equal(ops_init["kcoords"], ref["kcoords"])


def test_preprocessing_matches_reference(ks4_outputs, reference_ops):
    ops_pre = ks4_outputs["ops_preproc"]

    hp_filter = _to_numpy(ops_pre["preprocessing"]["hp_filter"])
    ref_filter = reference_ops["fwav"]
    np.testing.assert_allclose(hp_filter, ref_filter, rtol=1e-6, atol=1e-7)

    whitening = _to_numpy(ops_pre["preprocessing"]["whiten_mat"])
    ref_whitening = np.load(REFERENCE_DIR / "whitening_mat.npy", allow_pickle=False)
    np.testing.assert_allclose(whitening, ref_whitening, rtol=1e-6, atol=1e-7)


def test_drift_correction_matches_reference(ks4_outputs, reference_ops):
    ops_drift = ks4_outputs["ops_drift"]

    np.testing.assert_allclose(
        _to_numpy(ops_drift["dshift"]),
        reference_ops["dshift"],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        _to_numpy(ops_drift["yblk"]), reference_ops["yblk"], rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        _to_numpy(ops_drift["iKxx"]),
        reference_ops["iKxx"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_detection_metadata_matches_reference(ks4_outputs, reference_ops):
    detect_ops = ks4_outputs["detect_ops"]

    for key in ["iC", "iC2", "iCC", "iCC_mask", "iU"]:
        np.testing.assert_array_equal(
            _to_numpy(detect_ops[key]),
            reference_ops[key],
            err_msg=f"Detection key {key} diverged from reference",
        )


def test_saved_outputs_match_reference_files(ks4_outputs):
    results_dir = ks4_outputs["results_dir"]

    file_checks = {
        "spike_times.npy": {"rtol": 0, "atol": 0},
        "spike_clusters.npy": {"rtol": 0, "atol": 0},
        "spike_templates.npy": {"rtol": 0, "atol": 0},
        "spike_detection_templates.npy": {"rtol": 0, "atol": 0},
        "kept_spikes.npy": {"rtol": 0, "atol": 0},
        "templates.npy": {"rtol": 1e-6, "atol": 1e-7},
        "templates_ind.npy": {"rtol": 0, "atol": 0},
        "pc_features.npy": {"rtol": 1e-5, "atol": 1e-6},
        "pc_feature_ind.npy": {"rtol": 0, "atol": 0},
        "similar_templates.npy": {"rtol": 1e-6, "atol": 1e-7},
        "amplitudes.npy": {"rtol": 1e-6, "atol": 1e-7},
        "spike_positions.npy": {"rtol": 1e-6, "atol": 1e-7},
        "channel_map.npy": {"rtol": 0, "atol": 0},
        "channel_positions.npy": {"rtol": 0, "atol": 0},
        "channel_shanks.npy": {"rtol": 0, "atol": 0},
        "whitening_mat.npy": {"rtol": 1e-6, "atol": 1e-7},
        "whitening_mat_inv.npy": {"rtol": 1e-6, "atol": 1e-7},
    }

    for fname, tolerances in file_checks.items():
        ref_arr = np.load(REFERENCE_DIR / fname, allow_pickle=False, mmap_mode="r")
        new_arr = np.load(results_dir / fname, allow_pickle=False, mmap_mode="r")
        assert ref_arr.shape == new_arr.shape, f"{fname} shape mismatch"
        if np.issubdtype(ref_arr.dtype, np.integer) or np.issubdtype(
            ref_arr.dtype, np.bool_
        ):
            np.testing.assert_array_equal(ref_arr, new_arr, err_msg=f"{fname} mismatch")
        else:
            np.testing.assert_allclose(
                ref_arr,
                new_arr,
                rtol=tolerances["rtol"],
                atol=tolerances["atol"],
                err_msg=f"{fname} mismatch",
            )


def test_spike_arrays_and_clusters_match_reference(ks4_outputs):
    """Explicitly verify detected spikes and cluster assignments."""
    results_dir = ks4_outputs["results_dir"]

    ref_spike_times = np.load(REFERENCE_DIR / "spike_times.npy", mmap_mode="r")
    ref_spike_clusters = np.load(REFERENCE_DIR / "spike_clusters.npy", mmap_mode="r")
    ref_spike_templates = np.load(REFERENCE_DIR / "spike_templates.npy", mmap_mode="r")

    spike_times = np.load(results_dir / "spike_times.npy", mmap_mode="r")
    spike_clusters = np.load(results_dir / "spike_clusters.npy", mmap_mode="r")
    spike_templates = np.load(results_dir / "spike_templates.npy", mmap_mode="r")

    np.testing.assert_array_equal(ref_spike_times, spike_times)
    np.testing.assert_array_equal(ref_spike_clusters, spike_clusters)
    np.testing.assert_array_equal(ref_spike_templates, spike_templates)

    assert np.unique(spike_clusters).size == np.unique(ref_spike_clusters).size
    assert spike_times.shape == ref_spike_times.shape
    assert spike_clusters.shape == ref_spike_clusters.shape
    assert spike_templates.shape == ref_spike_templates.shape


def test_final_ops_summary_matches_reference(ks4_outputs, reference_ops):
    ops_saved = ks4_outputs["ops_saved"]

    for key in ["n_units_total", "n_units_good", "n_spikes"]:
        assert ops_saved[key] == reference_ops[key]

    np.testing.assert_allclose(
        ops_saved["mean_drift"], reference_ops["mean_drift"], rtol=1e-6, atol=1e-7
    )
