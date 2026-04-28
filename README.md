# Spike Sorting Hardware Porting — Tenstorrent Exploratory Lane

> **Contributor:** Nazish Baliyan  
> **Role:** Tenstorrent Feasibility (Non-AMD Lane)  
> **Hardware Target:** Tenstorrent Blackhole (TT-NN / TT-XLA)  
> **Status:** PCA ported ✅ | TT device blocked by hardware reset ⚠️

---

## Overview

This repository documents the feasibility study and implementation work for porting the **Kilosort4 spike sorting pipeline** from its original PyTorch/CUDA baseline to **Tenstorrent hardware** using the TT-NN API.

The work covers:
- Full pipeline analysis and operator inventory
- Isolation and testing of the PCA feature compression module
- Integration of PCA into the live pipeline (replaced bypass)
- Attempted real TT-NN execution on `tt-blackhole-01`
- Quantitative benchmarks and clear blocker documentation

---

## Key Result in 30 Seconds

| Claim | Evidence |
|-------|----------|
| PCA transform is TT-portable | Uses only `ttnn.sub` + `ttnn.matmul` (both supported) |
| PCA bypass fixed | `kilosort.py` line ~544: replaced `spike_pc_features = spike_features` with fit+transform |
| 10.2× dimension reduction | 61-dim → 6-dim, 48,800 → 4,800 bytes/batch |
| Transform overhead | 0.0054 ms per batch (negligible) |
| All tests pass | `experiments/cross_validate_pca.py`: **9/9 ✅** |
| TT device blocked | Ethernet core timeout on `tt-blackhole-01` — needs board reset |

---

## Repository Structure

```
spike-sorting-hardware-porting/
│
├── README.md                          ← You are here
├── Nazish_Complete_Progress_Report.md ← Full project report (read this first)
├── Nazish_Day_By_Day_Task_Tracker.md  ← Daily execution plan
├── s41592-024-02232-7.pdf             ← Kilosort4 reference paper
│
├── experiments/                       ← All experiment scripts
│   ├── cross_validate_pca.py          ← Cross-validation: 9/9 tests ✅
│   ├── pca_quantitative_comparison.py ← Before/after benchmark
│   ├── test_pca_module.py             ← PCA unit tests (7/7 ✅)
│   ├── test_pca_tenstorrent.py        ← TT-NN compatibility analysis
│   ├── test_pca_ttnn_real.py          ← Real TT-NN execution script
│   ├── test_filter_module.py          ← Filter module tests
│   ├── run_on_tenstorrent.exp         ← SSH automation script for TT machine
│   └── ssh_connect.exp                ← SSH connection helper
│
├── notes/                             ← Daily logs, results, exported models
│   ├── day0_alignment.md              ← Target stack confirmation
│   ├── day1_baseline_log.md           ← Codebase setup
│   ├── day2_pipeline_map.md           ← Full Kilosort4 pipeline trace
│   ├── day3_module_test.md            ← PCA + filter test results
│   ├── day4_backend_attempt.md        ← TT-NN compatibility analysis
│   ├── day5_tenstorrent_execution.md  ← Real hardware SSH session
│   ├── final_week_summary.md          ← Consolidated summary
│   ├── pca_transform.onnx             ← Exported ONNX model
│   ├── pca_test_results.json          ← PCA test output
│   ├── filter_test_results.json       ← Filter test output
│   ├── backend_attempt_results.json   ← TT-NN analysis output
│   ├── pca_quantitative_comparison.json ← Benchmark numbers
│   └── ttnn_real_results.json         ← Real hardware results from tt-blackhole-01
│
└── torchbci-hardware-ports-torchbci-module/   ← Modified pipeline codebase
    └── torchbci/algorithms/kilosort.py        ← ⭐ PCA integrated here
```

---

## Quick Start

### 1. Run cross-validation (no dependencies needed except torch)
```bash
python3 experiments/cross_validate_pca.py
# Expected: 9/9 tests pass
```

### 2. Run before/after benchmark
```bash
python3 experiments/pca_quantitative_comparison.py
# Expected: 10.2x dimension reduction, 0.0054ms transform
```

### 3. Run full PCA unit test suite
```bash
python3 experiments/test_pca_module.py
# Expected: 7/7 tests pass
```

### 4. Run TT-NN execution (requires Tenstorrent machine)
```bash
ssh nazishbaliyan@10.127.30.197
source ~/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal/python_env/bin/activate
cd ~/spike-sorting-hardware-porting
python3 experiments/test_pca_ttnn_real.py
# Note: requires board reset first (Ethernet core timeout blocker)
```

---

## Pipeline Architecture

```
Input Signal [C, N]
    │
    ▼
Kilosort4CAR           ← Common Average Referencing
    │
    ▼
Kilosort4Filtering     ← 300Hz Butterworth high-pass  ⚠️ scipy (not portable)
    │
    ▼
Kilosort4Whitening     ← ZCA decorrelation
    │
    ▼
Kilosort4Detection     ← Iterative template matching  ⚠️ control-flow (not portable)
    │
    ▼
Kilosort4PCFeatureConversion   ← PCA 61→6 dims  ✅ PORTED (sub + matmul)
    │                              ← Previously bypassed, NOW INTEGRATED
    ▼
SimpleOnlineKMeansClustering   ← Online K-Means
    │
    ▼
Cluster labels [N_spikes]
```

---

## Portability Summary

| Stage | Portable to TT-NN? | Blocker |
|-------|-------------------|---------|
| CAR | ✅ Yes | None — basic arithmetic |
| Filtering | ❌ No | `scipy.signal` (CPU-only) |
| Whitening | ⚠️ Partial | `torch.linalg.svd` in a loop |
| Detection | ❌ No | Iterative data-dependent control flow |
| **PCA transform** | ✅ **Yes** | None — `sub + matmul` only |
| PCA fit | ❌ No | `torch.pca_lowrank` not on TT |
| Clustering | ⚠️ Partial | Dynamic Python lists |

**Strategy:** Fit on CPU (one-time) → Transfer weights → Transform on TT hardware

---

## Hardware Execution Status

| Check | Status |
|-------|--------|
| `ttnn` importable on `tt-blackhole-01` | ✅ Yes |
| 4x Blackhole chips detected | ✅ Yes |
| `ttnn.open_device(0)` succeeds | ❌ Blocked |
| Blocker | Ethernet core timeout (firmware 19.4.2 > tested 19.4.0) |
| Fix | Admin board reset of `tt-blackhole-01` |
| TT-NN code ready to run | ✅ Yes (`experiments/test_pca_ttnn_real.py`) |

---

## Environment

**Local (macOS):** Python 3.x, PyTorch, torchaudio, scipy, numpy  
**Remote (TT machine):**
```
Host    : tt-blackhole-01 (10.127.30.197)
OS      : Ubuntu 22.04.5 LTS
Python  : 3.10.12
torch   : 2.7.1+cpu
ttnn    : 0.68.0 (TT-Metal venv)
Hardware: 4x Tenstorrent Blackhole chips
```

Activate TT environment:
```bash
source ~/assignment-1-single-core-matrix-multiplication-nbaliyan260/tt-metal/python_env/bin/activate
```

---

## Key Engineering Contribution

**File:** `torchbci-hardware-ports-torchbci-module/torchbci/algorithms/kilosort.py`

```diff
- # Old: PCA bypass — module existed but was never used
- spike_pc_features = spike_features

+ # New: lazy-fit + transform
+ if not self._pca_fitted:
+     self.pc_featuring.fit(spike_features)   # one-time, ~23ms
+     self._pca_fitted = True
+ spike_pc_features = self.pc_featuring.transform(spike_features)
```

**Effect:** Clustering receives 6-dimensional PCA-compressed features instead of raw 61-dimensional waveforms. Memory per batch reduced from 48,800 → 4,800 bytes (10.2×).

---

## Reference

- **Paper:** Pachitariu et al., *Kilosort4: fast spike sorting with a graph-based algorithm*, Nature Methods 2024. [`s41592-024-02232-7.pdf`](./s41592-024-02232-7.pdf)
- **Codebase:** Based on `torchbci` — PyTorch-based Brain-Computer Interface framework
- **Dataset:** C46 (Neuropixels recording, 384 channels, ~50 kHz)

---

*This work is part of a System Design course project evaluating hardware portability of neural signal processing pipelines.*
