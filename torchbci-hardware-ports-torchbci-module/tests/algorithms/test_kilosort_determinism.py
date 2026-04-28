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
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _set_block_seeds():
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
                f"Required data file missing for deterministic test: {fname}",
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


def _run_pipeline_once(settings, probe, device, results_dir: Path):
    settings = dict(settings)
    settings["results_dir"] = str(results_dir)
    settings["filename"] = [str(Path(f)) for f in settings["filename"]]

    _set_block_seeds()
    pipeline = KS4Pipeline(
        settings=settings, probe=probe, results_dir=results_dir, device=device
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

    return {
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


def _compare_saved_files(run_a: Path, run_b: Path):
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
        arr_a = np.load(run_a / fname, allow_pickle=False, mmap_mode="r")
        arr_b = np.load(run_b / fname, allow_pickle=False, mmap_mode="r")
        assert arr_a.shape == arr_b.shape, f"{fname} shape mismatch between runs"
        if np.issubdtype(arr_a.dtype, np.integer) or np.issubdtype(arr_a.dtype, np.bool_):
            np.testing.assert_array_equal(arr_a, arr_b, err_msg=f"{fname} mismatch")
        else:
            np.testing.assert_allclose(
                arr_a,
                arr_b,
                rtol=tolerances["rtol"],
                atol=tolerances["atol"],
                err_msg=f"{fname} mismatch",
            )


def test_two_runs_are_deterministic(tmp_path_factory, reference_ops, device):
    """
    Run the torchbci KS4 pipeline twice on the c46 dataset and ensure all
    artifacts, spikes, and cluster assignments are identical.
    """
    run1_dir = Path(tmp_path_factory.mktemp("ks4_run1"))
    run2_dir = Path(tmp_path_factory.mktemp("ks4_run2"))

    run1 = _run_pipeline_once(reference_ops["settings"], reference_ops["probe"], device, run1_dir)
    run2 = _run_pipeline_once(reference_ops["settings"], reference_ops["probe"], device, run2_dir)

    try:
        # Initialization consistency
        for key in ["NTbuff", "Nchan", "n_chan_bin", "duplicate_spike_bins"]:
            assert run1["ops_init"][key] == run2["ops_init"][key]
        np.testing.assert_array_equal(run1["ops_init"]["chanMap"], run2["ops_init"]["chanMap"])
        np.testing.assert_allclose(run1["ops_init"]["xc"], run2["ops_init"]["xc"])
        np.testing.assert_allclose(run1["ops_init"]["yc"], run2["ops_init"]["yc"])
        np.testing.assert_array_equal(run1["ops_init"]["kcoords"], run2["ops_init"]["kcoords"])

        # Preprocessing consistency
        np.testing.assert_allclose(
            _to_numpy(run1["ops_preproc"]["preprocessing"]["hp_filter"]),
            _to_numpy(run2["ops_preproc"]["preprocessing"]["hp_filter"]),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _to_numpy(run1["ops_preproc"]["preprocessing"]["whiten_mat"]),
            _to_numpy(run2["ops_preproc"]["preprocessing"]["whiten_mat"]),
            rtol=1e-6,
            atol=1e-7,
        )

        # Drift correction consistency
        np.testing.assert_allclose(
            _to_numpy(run1["ops_drift"]["dshift"]),
            _to_numpy(run2["ops_drift"]["dshift"]),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _to_numpy(run1["ops_drift"]["yblk"]),
            _to_numpy(run2["ops_drift"]["yblk"]),
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _to_numpy(run1["ops_drift"]["iKxx"]),
            _to_numpy(run2["ops_drift"]["iKxx"]),
            rtol=1e-6,
            atol=1e-7,
        )

        # Detection metadata consistency
        for key in ["iC", "iC2", "iCC", "iCC_mask", "iU"]:
            np.testing.assert_array_equal(
                _to_numpy(run1["detect_ops"][key]),
                _to_numpy(run2["detect_ops"][key]),
                err_msg=f"{key} differs between runs",
            )

        # Saved files consistency (spikes, clusters, templates, features, etc.)
        _compare_saved_files(run1["results_dir"], run2["results_dir"])

        # Final summary consistency
        for key in ["n_units_total", "n_units_good", "n_spikes"]:
            assert run1["ops_saved"][key] == run2["ops_saved"][key]
        np.testing.assert_allclose(
            run1["ops_saved"]["mean_drift"],
            run2["ops_saved"]["mean_drift"],
            rtol=1e-6,
            atol=1e-7,
        )
    finally:
        shutil.rmtree(run1_dir, ignore_errors=True)
        shutil.rmtree(run2_dir, ignore_errors=True)

