# Day 4 - Backend Attempt: Tenstorrent Compatibility

## Date: 2026-04-28

## Target
- **Backend:** Tenstorrent (TT-NN / TT-XLA)
- **Module:** Kilosort4PCFeatureConversion (PCA)

## Approach
1. Created a minimal `PCATransformModule` wrapping only the inference path
2. Analyzed all PyTorch ops for TT-NN compatibility
3. Attempted ONNX export as portability proxy
4. Wrote TT-NN pseudocode for the transform path

## Operator Compatibility Analysis

| Operation | Where Used | TT-NN Status | Notes |
|-----------|-----------|--------------|-------|
| `torch.sub` | transform (centering) | ✅ SUPPORTED | Basic element-wise op |
| `torch.matmul` | transform (projection) | ✅ SUPPORTED | Core GEMM op |
| `torch.mean` | fit (centering) | ✅ SUPPORTED | Reduction op |
| `torch.pca_lowrank` | fit (decomposition) | ❌ NOT SUPPORTED | Complex decomposition |
| `torch.linalg.svd` | fit (fallback) | ❌ NOT SUPPORTED | Complex linear algebra |
| `torch.Tensor.pow` | fit (variance) | ✅ SUPPORTED | Element-wise |
| `torch.Tensor.sum` | fit (variance) | ✅ SUPPORTED | Reduction |

**Result: 5/7 ops supported. The 2 unsupported ops are fit-only (offline calibration).**

## ONNX Export

| Metric | Value |
|--------|-------|
| Status | ✅ SUCCESS |
| File | `notes/pca_transform.onnx` |
| Size | 2608 bytes |
| Opset | 18 (auto-upgraded from requested 13) |
| Dynamic axes | batch_size dimension |

## Key Finding: Split Architecture Strategy

```
┌─────────────────────────────────┐
│       CPU / Host Side           │
│                                 │
│  1. Load data                   │
│  2. PCA fit() → learn basis     │
│     └─ pca_lowrank / SVD        │
│  3. Export components_ & mu_    │
│                                 │
└────────────┬────────────────────┘
             │ Transfer parameters
             ▼
┌─────────────────────────────────┐
│    Tenstorrent / TT-NN Side     │
│                                 │
│  4. Load components & mu        │
│     └─ ttnn.from_torch()        │
│  5. PCA transform() → project   │
│     └─ ttnn.sub + ttnn.matmul   │
│  6. Return reduced features     │
│                                 │
└─────────────────────────────────┘
```

**This is a valid split:** fit on CPU (one-time calibration), deploy transform to accelerator (repeated inference).

## Blockers

### For Transform (Inference) Path
- **None.** Only uses `sub` and `matmul`, both supported by TT-NN.

### For Fit (Calibration) Path  
- `torch.pca_lowrank` — not available on TT hardware
- `torch.linalg.svd` — not available on TT hardware
- **Mitigation:** Fit on CPU, which is the standard approach for PCA-based feature engineering

### For Full Pipeline Integration
- Filtering stage uses scipy (not PyTorch) — separate blocker
- Detection stage has iterative data-dependent loops — hard to accelerate
- Whitening uses per-channel SVD loops — moderate difficulty

## Status
- ✅ Minimal export module created and validated
- ✅ ONNX export successful
- ✅ TT-NN pseudocode documented
- ✅ Split architecture strategy defined
- ✅ Blockers clearly identified
