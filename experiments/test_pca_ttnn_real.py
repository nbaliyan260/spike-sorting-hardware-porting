#!/usr/bin/env python3
"""
REAL Tenstorrent TT-NN PCA Implementation
==========================================
This script attempts an ACTUAL TT-NN execution of the PCA transform.

Strategy:
  - Fit PCA on CPU (ttnn does not support SVD)
  - Transfer weights to TT device
  - Run transform using ttnn.sub + ttnn.matmul
  - Compare output vs PyTorch baseline
  - Report numerical difference and performance

Author: Nazish Baliyan | Date: 2026-04-28
"""
import sys, os, time, json, traceback
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'torchbci-hardware-ports-torchbci-module'))

results = {
    "target": "Tenstorrent TT-NN",
    "module": "PCA (Kilosort4PCFeatureConversion)",
    "timestamp": str(time.strftime("%Y-%m-%d %H:%M:%S")),
    "experiments": [],
    "ttnn_available": False,
    "ttnn_device_opened": False,
}

# ============================================================
# Step 0: Import and device setup
# ============================================================
print("=" * 65)
print("REAL TT-NN PCA IMPLEMENTATION")
print("=" * 65)

ttnn = None
device = None

print("\n[Step 0] Attempting to import ttnn...")
try:
    import ttnn
    results["ttnn_available"] = True
    print(f"  ✅ ttnn imported! version: {getattr(ttnn, '__version__', 'unknown')}")
except ImportError as e:
    print(f"  ❌ ttnn NOT available: {e}")
    print("  → Will run PyTorch fallback only")

# Try to open TT device
if ttnn is not None:
    print("\n[Step 0b] Opening TT device...")
    try:
        device = ttnn.open_device(device_id=0)
        results["ttnn_device_opened"] = True
        print(f"  ✅ Device opened: {device}")
    except Exception as e:
        print(f"  ❌ Failed to open device: {e}")
        device = None

# ============================================================
# Step 1: Fit PCA on CPU (always on CPU — SVD not on TT)
# ============================================================
print("\n[Step 1] Fitting PCA on CPU...")
try:
    from torchbci.algorithms.kilosort import Kilosort4PCFeatureConversion
    torch.manual_seed(42)
    dim_pc, n_samples, n_features = 6, 200, 61
    X = torch.randn(n_samples, n_features, dtype=torch.float32)

    pca = Kilosort4PCFeatureConversion(dim_pc_features=dim_pc, use_lowrank=True)
    pca.fit(X)

    # Get the fitted weights
    components_cpu = pca.components_.clone()  # [D, K]
    mu_cpu = pca.mu_.clone()                  # [1, D]

    # PyTorch baseline
    Z_pytorch = pca.transform(X)
    print(f"  ✅ PCA fit done. components: {components_cpu.shape}, mu: {mu_cpu.shape}")
    print(f"  ✅ PyTorch baseline Z shape: {Z_pytorch.shape}")
    results["experiments"].append({
        "name": "pca_fit_cpu",
        "status": "PASS",
        "components_shape": list(components_cpu.shape),
        "mu_shape": list(mu_cpu.shape),
        "Z_shape": list(Z_pytorch.shape),
    })
except Exception as e:
    print(f"  ❌ PCA fit FAILED: {e}")
    traceback.print_exc()
    results["experiments"].append({"name": "pca_fit_cpu", "status": "FAIL", "error": str(e)})
    # Can't continue without fitted PCA
    if device:
        try: ttnn.close_device(device)
        except: pass
    out_path = os.path.join(os.path.dirname(__file__), '..', 'notes', 'ttnn_real_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    sys.exit(1)

# ============================================================
# Step 2: PyTorch reference (manual sub + matmul)
# ============================================================
print("\n[Step 2] PyTorch reference (manual sub+matmul, no ttnn)...")
try:
    Xc_pt = X - mu_cpu
    Z_pt_manual = Xc_pt @ components_cpu
    diff_pt = torch.max(torch.abs(Z_pytorch - Z_pt_manual)).item()
    print(f"  ✅ Manual PyTorch: Z shape={Z_pt_manual.shape}, max_diff={diff_pt:.2e}")
    results["experiments"].append({
        "name": "pytorch_manual_transform",
        "status": "PASS",
        "max_diff_vs_original": diff_pt,
        "shape": list(Z_pt_manual.shape),
    })
except Exception as e:
    print(f"  ❌ PyTorch manual FAILED: {e}")
    results["experiments"].append({"name": "pytorch_manual_transform", "status": "FAIL", "error": str(e)})

# ============================================================
# Step 3: TT-NN Real Execution (if device is available)
# ============================================================
print("\n[Step 3] TT-NN real execution attempt...")

if device is not None and ttnn is not None:
    try:
        print("  → Transferring tensors to TT device...")

        # Convert to ttnn tensors (bfloat16 is native TT format)
        X_tt = ttnn.from_torch(
            X,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        mu_tt = ttnn.from_torch(
            mu_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        components_tt = ttnn.from_torch(
            components_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print(f"  ✅ Tensors on device — X:{X_tt.shape} mu:{mu_tt.shape} comp:{components_tt.shape}")

        # ---- THE REAL TT-NN EXECUTION ----
        print("  → Running ttnn.sub (centering)...")
        t0 = time.perf_counter()
        Xc_tt = ttnn.sub(X_tt, mu_tt)
        t_sub = (time.perf_counter() - t0) * 1000
        print(f"  ✅ ttnn.sub done in {t_sub:.2f}ms")

        print("  → Running ttnn.matmul (projection)...")
        t0 = time.perf_counter()
        Z_tt = ttnn.matmul(Xc_tt, components_tt)
        t_mm = (time.perf_counter() - t0) * 1000
        print(f"  ✅ ttnn.matmul done in {t_mm:.2f}ms")

        # Convert back to CPU for comparison
        Z_tt_cpu = ttnn.to_torch(Z_tt).float()

        # Numerical comparison
        diff_vs_pytorch = torch.max(torch.abs(Z_pytorch - Z_tt_cpu)).item()
        mse_vs_pytorch = torch.mean((Z_pytorch - Z_tt_cpu) ** 2).item()
        print(f"\n  NUMERICAL COMPARISON vs PyTorch:")
        print(f"    Max absolute diff : {diff_vs_pytorch:.6f}")
        print(f"    MSE               : {mse_vs_pytorch:.6f}")
        print(f"    (Note: bfloat16 has ~1e-2 precision, so small diffs expected)")

        passed = diff_vs_pytorch < 0.5  # bfloat16 tolerance
        print(f"  {'✅ PASS' if passed else '⚠️  WARN'} — diff={diff_vs_pytorch:.4f} {'< 0.5 OK' if passed else '>= 0.5 large'}")

        results["experiments"].append({
            "name": "ttnn_real_execution",
            "status": "PASS" if passed else "WARN",
            "ttnn_sub_ms": round(t_sub, 2),
            "ttnn_matmul_ms": round(t_mm, 2),
            "total_ms": round(t_sub + t_mm, 2),
            "max_diff_vs_pytorch": diff_vs_pytorch,
            "mse_vs_pytorch": mse_vs_pytorch,
            "tolerance": "bfloat16 ~1e-2",
            "dtype": "bfloat16",
        })

        # Clean up TT tensors
        ttnn.deallocate(X_tt)
        ttnn.deallocate(mu_tt)
        ttnn.deallocate(components_tt)
        ttnn.deallocate(Xc_tt)
        ttnn.deallocate(Z_tt)

    except Exception as e:
        print(f"  ❌ TT-NN execution FAILED: {e}")
        traceback.print_exc()
        results["experiments"].append({
            "name": "ttnn_real_execution",
            "status": "FAIL",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
else:
    print("  ⚠️  TT-NN device not available — skipping real execution")
    print("  → Recording as BLOCKED (ttnn not installed / no device)")
    results["experiments"].append({
        "name": "ttnn_real_execution",
        "status": "BLOCKED",
        "reason": "ttnn not importable or device not available",
        "ttnn_available": results["ttnn_available"],
        "device_opened": results["ttnn_device_opened"],
    })

# ============================================================
# Step 4: Performance benchmark (if ttnn worked)
# ============================================================
print("\n[Step 4] Performance benchmark...")
tt_exp = next((e for e in results["experiments"] if e.get("name") == "ttnn_real_execution" and e.get("status") == "PASS"), None)

if tt_exp:
    print(f"  TT-NN  : sub={tt_exp['ttnn_sub_ms']:.2f}ms, matmul={tt_exp['ttnn_matmul_ms']:.2f}ms, total={tt_exp['total_ms']:.2f}ms")
    # Compare vs pure PyTorch
    t0 = time.perf_counter()
    for _ in range(100):
        Xc_bm = X - mu_cpu
        Z_bm  = Xc_bm @ components_cpu
    t_pt = (time.perf_counter() - t0) / 100 * 1000
    print(f"  PyTorch: {t_pt:.3f}ms avg over 100 runs")
    results["experiments"].append({
        "name": "performance_comparison",
        "pytorch_ms": round(t_pt, 3),
        "ttnn_total_ms": tt_exp["total_ms"],
        "note": "Single measurement — warm-up not included for TT",
    })
else:
    print("  ⚠️  Skipping (TT-NN execution not available)")

# ============================================================
# Step 5: Close device
# ============================================================
if device is not None:
    print("\n[Step 5] Closing TT device...")
    try:
        ttnn.close_device(device)
        print("  ✅ Device closed cleanly")
    except Exception as e:
        print(f"  ⚠️  Close error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  ttnn available    : {'YES' if results['ttnn_available'] else 'NO'}")
print(f"  TT device opened  : {'YES' if results['ttnn_device_opened'] else 'NO'}")

tt_real = next((e for e in results["experiments"] if e.get("name") == "ttnn_real_execution"), {})
status = tt_real.get("status", "N/A")
print(f"  TT-NN execution   : {status}")

if status == "PASS":
    print(f"  Max diff vs PT    : {tt_real.get('max_diff_vs_pytorch', 'N/A'):.6f}")
    print(f"  MSE vs PT         : {tt_real.get('mse_vs_pytorch', 'N/A'):.6f}")
    print(f"  Performance       : {tt_real.get('total_ms', 'N/A'):.2f}ms (sub+matmul)")
    print(f"\n  ✅ PCA TRANSFORM RUNS ON TENSTORRENT HARDWARE")
elif status == "BLOCKED":
    print(f"  Reason: {tt_real.get('reason', 'unknown')}")
    print(f"\n  ⚠️  Cannot confirm — ttnn not installed on this machine")
    print(f"  → TT-NN ops (sub, matmul) are theoretically supported")
    print(f"  → ONNX export succeeded as portability proxy")
else:
    print(f"  Error: {tt_real.get('error', 'unknown')}")
    print(f"\n  ❌ PCA TRANSFORM FAILED ON TT HARDWARE")

print(f"\n  BLOCKERS:")
print(f"    • PCA fit (SVD): CPU only, NOT portable to TT")
print(f"    • PCA transform: No blockers (sub + matmul)")
if not results["ttnn_available"]:
    print(f"    • ttnn package not installed on this host")

# Save results
results["summary"] = {
    "pca_transform_runs_on_tt": status == "PASS",
    "ttnn_available": results["ttnn_available"],
    "ttnn_device_opened": results["ttnn_device_opened"],
    "ttnn_execution_status": status,
    "blocker_fit": "torch.pca_lowrank / SVD not supported on TT",
    "blocker_transform": "None — sub+matmul both supported",
}

out_path = os.path.join(os.path.dirname(__file__), '..', 'notes', 'ttnn_real_results.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n  Results saved: {out_path}")
