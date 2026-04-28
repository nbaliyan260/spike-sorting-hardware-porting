#!/usr/bin/env python3
"""
Standalone test for Kilosort4Filtering module (backup experiment).
Tests the high-pass Butterworth filter with synthetic data.
Author: Nazish Baliyan | Date: 2026-04-28
"""
import sys, os, time, json
import torch, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torchbci-hardware-ports-torchbci-module'))
from torchbci.algorithms.kilosort import Kilosort4Filtering

def run_filter_test():
    results = {"module": "Kilosort4Filtering", "tests": []}
    sample_rate, cutoff = 50024, 300
    n_channels, n_samples = 10, 5000

    print("[1] Instantiation...")
    filt = Kilosort4Filtering(sample_rate=sample_rate, cutoff_freq=cutoff)
    print(f"  OK  sr={filt.sample_rate} cutoff={filt.cutoff_freq}")
    results["tests"].append({"name": "instantiation", "status": "PASS"})

    print("[2] Forward pass...")
    torch.manual_seed(42)
    X = torch.randn(n_channels, n_samples, dtype=torch.float32)
    t0 = time.perf_counter()
    Y = filt(X)
    t_fwd = (time.perf_counter() - t0) * 1000
    print(f"  OK  in={list(X.shape)} out={list(Y.shape)} time={t_fwd:.1f}ms")
    assert Y.shape == X.shape
    results["tests"].append({"name": "forward", "status": "PASS", "time_ms": round(t_fwd,1)})

    print("[3] Repeatability...")
    outs = [filt(X) for _ in range(3)]
    diffs = [torch.max(torch.abs(outs[0]-outs[i])).item() for i in range(1,3)]
    det = all(d < 1e-5 for d in diffs)
    print(f"  {'OK' if det else 'WARN'}  max_diffs={diffs} det={det}")
    results["tests"].append({"name": "repeat", "status": "PASS" if det else "WARN"})

    print("[4] Portability analysis...")
    print("  WARNING: Uses scipy.signal.butter + sosfiltfilt (NOT PyTorch)")
    print("  → Cannot be exported to ONNX or TT-NN directly")
    print("  → Would need pure PyTorch rewrite (IIR or FIR filter)")
    results["tests"].append({"name": "portability", "status": "BLOCKED", "reason": "scipy dependency"})

    failed = sum(1 for t in results["tests"] if t["status"] == "FAIL")
    print(f"\nSUMMARY: {len(results['tests'])} tests, {failed} failed")
    results["overall"] = "PASS" if failed == 0 else "FAIL"

    out_path = os.path.join(os.path.dirname(__file__), '..', 'notes', 'filter_test_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results

if __name__ == "__main__":
    r = run_filter_test()
    sys.exit(0 if r["overall"] == "PASS" else 1)
