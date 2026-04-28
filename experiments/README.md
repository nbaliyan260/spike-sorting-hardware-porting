# experiments/

Standalone experiment scripts for the Tenstorrent PCA porting feasibility study.

All scripts are self-contained — they can be run directly with Python and require only `torch` and `numpy` (no full pipeline setup needed).

---

## Scripts

| File | Purpose | Run It |
|------|---------|--------|
| [`cross_validate_pca.py`](cross_validate_pca.py) | **Cross-validation** — 9 tests verifying all report numbers are correct | `python3 cross_validate_pca.py` |
| [`pca_quantitative_comparison.py`](pca_quantitative_comparison.py) | **Benchmark** — Before/after PCA: dimension, memory, timing, MSE | `python3 pca_quantitative_comparison.py` |
| [`test_pca_module.py`](test_pca_module.py) | **Unit tests** — 7 tests for `Kilosort4PCFeatureConversion` | `python3 test_pca_module.py` |
| [`test_filter_module.py`](test_filter_module.py) | **Unit tests** — 4 tests for `Kilosort4Filtering` (scipy, CPU-only) | `python3 test_filter_module.py` |
| [`test_pca_tenstorrent.py`](test_pca_tenstorrent.py) | **TT-NN analysis** — operator compatibility, ONNX export, pseudocode | `python3 test_pca_tenstorrent.py` |
| [`test_pca_ttnn_real.py`](test_pca_ttnn_real.py) | **Real TT-NN execution** — runs on `tt-blackhole-01` with real `ttnn` | Requires TT machine + board reset |
| [`run_on_tenstorrent.exp`](run_on_tenstorrent.exp) | **Automation** — `expect` script: SSH → activate venv → clone → run | `expect run_on_tenstorrent.exp` |
| [`ssh_connect.exp`](ssh_connect.exp) | **SSH helper** — basic connection to `tt-blackhole-01` | `expect ssh_connect.exp` |

---

## Execution Order (Recommended)

```
1. test_pca_module.py              # verify the module works standalone (7/7)
2. test_filter_module.py           # verify filter module + confirm scipy blocker (4/4)
3. test_pca_tenstorrent.py         # run TT-NN compatibility analysis + ONNX export
4. pca_quantitative_comparison.py  # get before/after benchmark numbers
5. cross_validate_pca.py           # verify all report numbers are correct (9/9)
6. test_pca_ttnn_real.py           # run on TT machine (needs board reset)
```

---

## Expected Results

```
cross_validate_pca.py        → 9/9  ✅
test_pca_module.py           → 7/7  ✅
test_filter_module.py        → 4/4  ✅ (but scipy confirmed as blocker)
pca_quantitative_comparison  → 10.2x reduction, 0.0054ms transform ✅
test_pca_ttnn_real.py        → BLOCKED (Ethernet core timeout on tt-blackhole-01)
```

---

## Dependencies

```
torch        ≥ 2.0
numpy        ≥ 1.24
scipy        (only for test_filter_module.py)
ttnn         (only for test_pca_ttnn_real.py — TT machine only)
```

Output JSON files are saved to `../notes/`.
