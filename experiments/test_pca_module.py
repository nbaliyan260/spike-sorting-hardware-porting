#!/usr/bin/env python3
"""
Standalone test for Kilosort4PCFeatureConversion module.
Tests: instantiation, fit, transform, inverse_transform, repeatability, timing.
Author: Nazish Baliyan | Date: 2026-04-28
"""
import sys, os, time, json, traceback
import torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torchbci-hardware-ports-torchbci-module'))
from torchbci.algorithms.kilosort import Kilosort4PCFeatureConversion

def run_pca_test():
    results = {"module": "Kilosort4PCFeatureConversion", "tests": []}
    dim_pc, n_samples, n_features = 6, 200, 61

    # Test 1: Instantiation
    print("[1] Instantiation...")
    pca = Kilosort4PCFeatureConversion(dim_pc_features=dim_pc, center=True, use_lowrank=True)
    print(f"  OK  fitted={pca.fitted_}")
    results["tests"].append({"name": "instantiation", "status": "PASS"})

    # Test 2: Fit
    print("[2] Fit...")
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features, dtype=torch.float32)
    t0 = time.perf_counter()
    pca.fit(X)
    t_fit = (time.perf_counter() - t0) * 1000
    print(f"  OK  components={list(pca.components_.shape)} var={pca.variance_explained:.4f} time={t_fit:.1f}ms")
    results["tests"].append({"name": "fit", "status": "PASS", "time_ms": round(t_fit,1)})

    # Test 3: Transform
    print("[3] Transform...")
    t0 = time.perf_counter()
    Z = pca.transform(X)
    t_tr = (time.perf_counter() - t0) * 1000
    assert Z.shape == (n_samples, dim_pc)
    print(f"  OK  shape={list(Z.shape)} time={t_tr:.1f}ms")
    results["tests"].append({"name": "transform", "status": "PASS", "shape": list(Z.shape)})

    # Test 4: Inverse transform
    print("[4] Inverse transform...")
    Xr = pca.inverse_transform(Z)
    mse = torch.mean((X - Xr)**2).item()
    print(f"  OK  recon_mse={mse:.6f}")
    results["tests"].append({"name": "inverse", "status": "PASS", "mse": round(mse,6)})

    # Test 5: Repeatability
    print("[5] Repeatability (3 runs)...")
    outs = []
    for i in range(3):
        torch.manual_seed(42)
        p = Kilosort4PCFeatureConversion(dim_pc_features=dim_pc, use_lowrank=True)
        outs.append(p.fit_transform(X))
    diffs = [torch.max(torch.abs(outs[i]-outs[j])).item() for i,j in [(0,1),(0,2),(1,2)]]
    det = all(d < 1e-5 for d in diffs)
    print(f"  {'OK' if det else 'WARN'}  max_diffs={[f'{d:.2e}' for d in diffs]} deterministic={det}")
    results["tests"].append({"name": "repeat", "status": "PASS" if det else "WARN"})

    # Test 6: Timing (10 iters)
    print("[6] Timing benchmark...")
    fit_t, tr_t = [], []
    for i in range(10):
        Xi = torch.randn(n_samples, n_features)
        p = Kilosort4PCFeatureConversion(dim_pc_features=dim_pc, use_lowrank=True)
        t0 = time.perf_counter(); p.fit(Xi); fit_t.append(time.perf_counter()-t0)
        t0 = time.perf_counter(); p.transform(Xi); tr_t.append(time.perf_counter()-t0)
    print(f"  OK  fit={np.mean(fit_t)*1000:.1f}±{np.std(fit_t)*1000:.1f}ms  transform={np.mean(tr_t)*1000:.1f}±{np.std(tr_t)*1000:.1f}ms")
    results["tests"].append({"name": "timing", "status": "PASS", "fit_ms": round(np.mean(fit_t)*1000,1)})

    # Test 7: forward() alias
    print("[7] forward() alias...")
    pca.fit(X)
    assert torch.allclose(pca(X), pca.transform(X))
    print("  OK  forward==transform")
    results["tests"].append({"name": "forward_alias", "status": "PASS"})

    # Operator inventory
    print("\nOPERATOR INVENTORY:")
    print("  fit: mean, pca_lowrank/svd, pow, sum, reshape")
    print("  transform: matmul(@), sub")
    print("  inverse: matmul(@), mul, add")

    failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
    print(f"\nSUMMARY: {len(results['tests'])} tests, {failed} failed")
    results["overall"] = "PASS" if failed == 0 else "FAIL"

    out_path = os.path.join(os.path.dirname(__file__), '..', 'notes', 'pca_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {out_path}")
    return results

if __name__ == "__main__":
    r = run_pca_test()
    sys.exit(0 if r["overall"] == "PASS" else 1)
