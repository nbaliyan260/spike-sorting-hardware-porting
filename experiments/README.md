# experiments/

Standalone experiment scripts for the Tenstorrent PCA porting feasibility study.

All scripts are self-contained — they can be run directly with Python and require only `torch` and `numpy` (no full pipeline setup needed).

---

## Scripts

| File | Purpose | Tests | Run It |
|------|---------|-------|--------|
| [`test_pca_module.py`](test_pca_module.py) | **Unit tests** — 7 tests for `Kilosort4PCFeatureConversion` | 7/7 ✅ | `python3 test_pca_module.py` |
| [`cross_validate_pca.py`](cross_validate_pca.py) | **Cross-validation** — 9 tests verifying all report numbers | 9/9 ✅ | `python3 cross_validate_pca.py` |
| [`test_pca_simulated_recordings.py`](test_pca_simulated_recordings.py) | **Structured synthetic validation** — realistic Neuropixels data | 8/8 ✅ | `python3 test_pca_simulated_recordings.py` |
| [`test_pca_allen_real.py`](test_pca_allen_real.py) | **Real Allen ground truth** — 1,437 spikes, 9 clusters | 8/8 ✅ | `python3 test_pca_allen_real.py` |
| [`pca_quantitative_comparison.py`](pca_quantitative_comparison.py) | **Benchmark** — Before/after PCA: dimension, memory, timing | ✅ | `python3 pca_quantitative_comparison.py` |
| [`test_filter_module.py`](test_filter_module.py) | **Filter unit tests** — confirms scipy CPU-only blocker | 4/4 ✅ | `python3 test_filter_module.py` |
| [`test_pca_tenstorrent.py`](test_pca_tenstorrent.py) | **TT-NN analysis** — operator compatibility, ONNX export | ✅ | `python3 test_pca_tenstorrent.py` |
| [`test_pca_ttnn_real.py`](test_pca_ttnn_real.py) | **Real TT-NN execution** — runs PCA on Blackhole hardware | ✅ PASS | Requires TT machine |
| [`run_on_tenstorrent.exp`](run_on_tenstorrent.exp) | **Automation** — SSH + run scripts on TT machine | — | `expect run_on_tenstorrent.exp` |
| [`run_full_remote.exp`](run_full_remote.exp) | **Full remote runner** — all experiments on TT machine | — | `expect run_full_remote.exp` |
| [`ssh_connect.exp`](ssh_connect.exp) | **SSH helper** — basic connection to TT machine | — | `expect ssh_connect.exp` |
| [`run_tt_machine.sh`](run_tt_machine.sh) | **Shell runner** — runs experiments on TT machine | — | `bash run_tt_machine.sh` |

---

## Execution Order (Recommended)

```
1. test_pca_module.py                    # verify PCA module standalone (7/7)
2. cross_validate_pca.py                 # verify all report numbers (9/9)
3. test_pca_simulated_recordings.py      # structured synthetic validation (8/8)
4. test_pca_allen_real.py                # real Allen ground truth validation (8/8)
5. pca_quantitative_comparison.py        # get before/after benchmark numbers
6. test_filter_module.py                 # verify filter + confirm scipy blocker (4/4)
7. test_pca_tenstorrent.py              # TT-NN compatibility analysis + ONNX export
8. test_pca_ttnn_real.py                 # run on TT machine (needs ttnn + device)
```

---

## Expected Results

```
test_pca_module.py                    → 7/7  ✅
cross_validate_pca.py                 → 9/9  ✅
test_pca_simulated_recordings.py      → 8/8  ✅  (61.7% variance on structured data)
test_pca_allen_real.py                → 8/8  ✅  (100% variance on 1,437 real spikes)
pca_quantitative_comparison.py        → 10.2x reduction, ~0.006ms transform  ✅
test_filter_module.py                 → 4/4  ✅  (scipy confirmed as blocker)
test_pca_tenstorrent.py              → ✅    (ONNX export, TT-NN compatibility analysis)
test_pca_ttnn_real.py                 → ✅ PASS on tt-blackhole-01 AND tt-blackhole-02
                                        Max diff: 0.034, MSE: 4.89e-05 (bfloat16)
```

---

## Hardware Execution Summary

PCA transform verified on **two** Tenstorrent Blackhole machines:
- `tt-blackhole-01` (2026-04-30): ttnn.sub 747.7ms + ttnn.matmul 647.1ms = 1394.8ms total
- `tt-blackhole-02` (2026-05-01): ttnn.sub 681.8ms + ttnn.matmul 657.6ms = 1339.4ms total
- **Numerical results identical:** max diff 0.034, MSE 4.89e-05

---

## Dependencies

```
torch        ≥ 2.0
numpy        ≥ 1.24
scipy        (only for test_filter_module.py)
torchaudio   (optional — guarded import in kilosort.py)
ttnn         (only for test_pca_ttnn_real.py — TT machine only)
```

Output JSON files are saved to `../notes/`.
