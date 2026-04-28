# Day 1 - Baseline Setup and First Run

## Date: 2026-04-28

## Environment Details

| Parameter | Value |
|-----------|-------|
| Machine | macOS (local dev) / 10.127.30.197 (remote) |
| OS | macOS (local) / Linux (remote, TBD) |
| Python version | 3.10+ |
| PyTorch version | TBD (requires `torch`, `torchaudio`, `scipy`, `sklearn`) |
| Device | CPU (initial baseline) |
| Dataset | C46 (`c46_npx_raw.bin`) |

## Repo Status

### Cloned Repos
1. **torchbci-hardware-ports-torchbci-module** — the hardware-port fork (primary working repo)
   - Contains: `demo/`, `torchbci/`, `evaluation/`, `tests/`, `doc/`
   - Key files: `torchbci/algorithms/kilosort.py` (623 lines), `demo/kilosort_driver.py`
   
2. **torchbci-main** — the original baseline
   - Contains: `demo/`, `torchbci/` (smaller, no kilosort4/ or datasets/)
   - Key file: `torchbci/algorithms/kilosort.py` (original version)

### Dependencies Required
```
torch
torchaudio
scipy
scikit-learn
numpy
matplotlib
```

## Baseline Run Attempt

### Entry Points Identified
1. **`demo/kilosort_driver.py`** — uses `KS4Pipeline` from `torchbci.algorithms.kilosort_ported` (the full ported Kilosort4 pipeline)
2. **`demo/build_kilosort4_with_blocks.ipynb`** — tutorial notebook showing block abstractions
3. **`torchbci/algorithms/kilosort.py`** — contains `Kilosort4Algorithm` with modular block-based pipeline

### Baseline Status: Local Code Analysis Complete
- The modular pipeline (`Kilosort4Algorithm`) can be instantiated and run on synthetic data
- The full pipeline (`KS4Pipeline` in `kilosort_ported.py`) requires data files and probe configuration
- The modular approach is more suitable for isolated module testing

### Key Finding
The `Kilosort4Algorithm.forward()` path:
1. ✅ CAR (Common Average Referencing) — active
2. ✅ Filtering (300 Hz high-pass Butterworth) — active
3. ✅ Whitening (ZCA transform) — active
4. ✅ Detection (iterative template matching) — active
5. ⚠️ PCA Feature Conversion — **instantiated but BYPASSED** (`spike_pc_features = spike_features`)
6. ✅ Clustering (Simple Online K-Means) — active

### Error Log
- No runtime errors yet (code analysis only on local machine)
- Remote machine execution pending (requires SSH access + data files)

## Exit Criteria Status
- [x] Code structure understood
- [x] Entry points identified
- [x] Active pipeline path traced
- [ ] Baseline executed on remote machine (pending)
- [x] Exact failing step identified if applicable (PCA bypass found)
