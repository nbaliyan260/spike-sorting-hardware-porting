#!/usr/bin/env python3
"""
Improvement 2: Quantitative PCA Comparison
==========================================
Produces a clean before-vs-after table showing the concrete benefit
of integrating PCA into the Kilosort4 pipeline.

Author: Nazish Baliyan | Date: 2026-04-29
"""
import sys, os, time, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torchbci-hardware-ports-torchbci-module'))
from torchbci.algorithms.kilosort import Kilosort4PCFeatureConversion

torch.manual_seed(42)

# Simulate what detection stage outputs: N spike waveforms, each 61 time points
N_SPIKES   = 200   # number of detected spikes
D_FEATURES = 61    # raw feature dimension (lookbehind + lookahead + 1)
K_COMPONENTS = 6   # PCA output dimension

print("=" * 62)
print("QUANTITATIVE COMPARISON: Before PCA vs After PCA")
print("=" * 62)

spike_features = torch.randn(N_SPIKES, D_FEATURES)

# ── BEFORE PCA ────────────────────────────────────────────────
print("\n[BEFORE PCA — Bypass Mode]")
t0 = time.perf_counter()
for _ in range(1000):
    spike_pc_features_before = spike_features   # bypass
t_before = (time.perf_counter() - t0) / 1000 * 1000

mem_before = spike_features.numel() * 4  # float32 bytes
print(f"  Feature dimension : {spike_features.shape[1]} (raw, unreduced)")
print(f"  Output shape      : {list(spike_features.shape)}")
print(f"  Memory per batch  : {mem_before} bytes ({mem_before/1024:.2f} KB)")
print(f"  Time (bypass)     : {t_before:.4f} ms  (1000 runs avg)")
print(f"  Reconstruction MSE: N/A (no compression)")

# ── AFTER PCA ─────────────────────────────────────────────────
print("\n[AFTER PCA — Integrated Mode]")
pca = Kilosort4PCFeatureConversion(dim_pc_features=K_COMPONENTS, use_lowrank=True)

t_fit_start = time.perf_counter()
pca.fit(spike_features)
t_fit = (time.perf_counter() - t_fit_start) * 1000

t_tr_runs = []
for _ in range(1000):
    t0 = time.perf_counter()
    spike_pc_features_after = pca.transform(spike_features)
    t_tr_runs.append((time.perf_counter() - t0) * 1000)
t_transform = np.mean(t_tr_runs)

mem_after = spike_pc_features_after.numel() * 4
recon = pca.inverse_transform(spike_pc_features_after)
mse = torch.mean((spike_features - recon) ** 2).item()
variance_kept = pca.variance_explained

print(f"  Feature dimension : {spike_pc_features_after.shape[1]} (PCA-compressed)")
print(f"  Output shape      : {list(spike_pc_features_after.shape)}")
print(f"  Memory per batch  : {mem_after} bytes ({mem_after/1024:.2f} KB)")
print(f"  Time (fit, once)  : {t_fit:.2f} ms")
print(f"  Time (transform)  : {t_transform:.4f} ms  (1000 runs avg)")
print(f"  Reconstruction MSE: {mse:.6f}")
print(f"  Variance explained: {variance_kept:.4f} ({variance_kept*100:.1f}%)")

# ── COMPARISON TABLE ──────────────────────────────────────────
dim_reduction = spike_features.shape[1] / spike_pc_features_after.shape[1]
mem_reduction = mem_before / mem_after
speed_overhead = t_transform / t_before if t_before > 0 else 0

print("\n" + "=" * 62)
print("RESULTS TABLE")
print("=" * 62)
print(f"{'Metric':<30} {'Before PCA':>14} {'After PCA':>14}")
print(f"{'─'*30} {'─'*14} {'─'*14}")
print(f"{'Feature dimension':<30} {D_FEATURES:>14} {K_COMPONENTS:>14}")
print(f"{'Output shape':<30} {'[200, 61]':>14} {'[200, 6]':>14}")
print(f"{'Memory (bytes/batch)':<30} {mem_before:>14} {mem_after:>14}")
print(f"{'Memory reduction':<30} {'1x':>14} {f'{mem_reduction:.1f}x less':>14}")
print(f"{'Dimension reduction':<30} {'1x':>14} {f'{dim_reduction:.1f}x less':>14}")
print(f"{'Transform time (ms)':<30} {t_before:>14.4f} {t_transform:>14.4f}")
print(f"{'Reconstruction MSE':<30} {'—':>14} {mse:>14.6f}")
print(f"{'Variance explained':<30} {'100% (raw)':>14} {f'{variance_kept*100:.1f}%':>14}")
print("=" * 62)

print(f"""
KEY INSIGHT:
  PCA reduces spike feature dimension from {D_FEATURES} → {K_COMPONENTS} ({dim_reduction:.1f}x)
  Memory reduced {mem_reduction:.1f}x: {mem_before} bytes → {mem_after} bytes
  Variance preserved: {variance_kept*100:.1f}%
  Reconstruction MSE: {mse:.4f} (expected — 6 of 61 dims kept)
  Transform overhead: {t_transform:.4f} ms per batch (negligible)
""")

# Save results
results = {
    "before_pca": {
        "feature_dim": D_FEATURES,
        "shape": [N_SPIKES, D_FEATURES],
        "memory_bytes": mem_before,
        "time_ms": round(t_before, 6),
    },
    "after_pca": {
        "feature_dim": K_COMPONENTS,
        "shape": [N_SPIKES, K_COMPONENTS],
        "memory_bytes": mem_after,
        "fit_time_ms": round(t_fit, 3),
        "transform_time_ms": round(t_transform, 6),
        "reconstruction_mse": round(mse, 6),
        "variance_explained": round(variance_kept, 4),
    },
    "improvement": {
        "dimension_reduction": f"{dim_reduction:.1f}x",
        "memory_reduction": f"{mem_reduction:.1f}x",
        "variance_preserved_pct": round(variance_kept * 100, 1),
    }
}

out = os.path.join(os.path.dirname(__file__), '..', 'notes', 'pca_quantitative_comparison.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {out}")
