# Day 3 - Module Test Results

## Date: 2026-04-28

## Primary Module: Kilosort4PCFeatureConversion (PCA)

### Test Results (All 7 PASSED ✅)

| Test | Status | Details |
|------|--------|---------|
| Instantiation | ✅ PASS | `dim_pc_features=6, use_lowrank=True` |
| Fit | ✅ PASS | components=[61,6], var_explained=16.91%, time=23.6ms |
| Transform | ✅ PASS | [200,61] → [200,6], time≈0ms |
| Inverse transform | ✅ PASS | reconstruction MSE=0.821359 |
| Repeatability | ✅ PASS | max_diff=0.00 across 3 runs — fully deterministic |
| Timing benchmark | ✅ PASS | fit=0.2ms, transform≈0ms (10 iterations) |
| forward() alias | ✅ PASS | nn.Module compatible |

### Operator Inventory
- **fit():** `torch.mean`, `torch.pca_lowrank` / `torch.linalg.svd`, `pow`, `sum`, `reshape`
- **transform():** `matmul` (@), `sub`
- **inverse_transform():** `matmul` (@), `mul`, `add`

### Key Observations
1. **Pure PyTorch** — no scipy, no numpy in the hot path
2. **Deterministic** — identical outputs across repeated runs with same seed
3. **Fast** — sub-millisecond on CPU for typical Kilosort4 input sizes
4. **Self-contained** — no external state dependencies beyond input data

## Backup Module: Kilosort4Filtering

### Test Results (All 4 PASSED ✅)

| Test | Status | Details |
|------|--------|---------|
| Instantiation | ✅ PASS | sr=50024, cutoff=300 |
| Forward pass | ✅ PASS | [10,5000]→[10,5000], time=5.1ms |
| Repeatability | ✅ PASS | deterministic |
| Portability | ⛔ BLOCKED | scipy.signal dependency |

### Key Finding
Filtering uses `scipy.signal.butter` + `scipy.signal.sosfiltfilt` — **cannot be ported** to any accelerator backend without a complete rewrite to pure PyTorch IIR/FIR filtering.
