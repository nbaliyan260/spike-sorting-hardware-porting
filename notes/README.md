# notes/

Daily research logs, experiment results, and exported artifacts from the Tenstorrent PCA porting project.

---

## Daily Logs

| File | Day | Contents |
|------|-----|----------|
| [`day0_alignment.md`](day0_alignment.md) | Day 0 | Confirmed target stack (Tenstorrent TT-NN), clarified my exploratory role, and secured machine access. |
| [`day1_baseline_log.md`](day1_baseline_log.md) | Day 1 | Read through the codebase to map out entry points. Discovered that PCA was instantiated but completely bypassed. |
| [`day2_pipeline_map.md`](day2_pipeline_map.md) | Day 2 | Traced the complete forward path. Identified all 6 stages, cataloged operators, and assessed what would break on accelerators. |
| [`day3_module_test.md`](day3_module_test.md) | Day 3 | Built standalone harnesses for PCA and filtering. Benchmarked PCA and confirmed filtering is blocked by `scipy`. |
| [`day4_backend_attempt.md`](day4_backend_attempt.md) | Day 4 | Analyzed TT-NN op compatibility. Exported ONNX to confirm PCA only uses universally supported `Sub` and `MatMul`. |
| [`day5_tenstorrent_execution.md`](day5_tenstorrent_execution.md) | Day 5 | Logged into `tt-blackhole-01`. Got all scripts running but hit an Ethernet core timeout. |
| [`final_week_summary.md`](final_week_summary.md) | Final | Consolidated the week's findings, drafted the team update, and recommended next steps for hardware engineers. |

---

## Experiment Results (JSON)

| File | Description |
|------|-------------|
| [`pca_test_results.json`](pca_test_results.json) | 7-test PCA unit test output |
| [`filter_test_results.json`](filter_test_results.json) | 4-test filter module output |
| [`backend_attempt_results.json`](backend_attempt_results.json) | TT-NN operator analysis output |
| [`pca_quantitative_comparison.json`](pca_quantitative_comparison.json) | Before/after benchmark: dimension, memory, timing |
| [`pca_simulated_recordings_shaped_validation.json`](pca_simulated_recordings_shaped_validation.json) | simulated recordings realistic dataset validation results |
| [`pca_allen_real_validation.json`](pca_allen_real_validation.json) | Real Allen Institute Kilosort4 ground truth validation |
| [`ttnn_real_results.json`](ttnn_real_results.json) | Real results from `tt-blackhole-01` |

---

## Exported Models

| File | Description |
|------|-------------|
| [`pca_transform.onnx`](pca_transform.onnx) | ONNX export of `PCATransformModule` (2,608 bytes) |
| [`pca_transform.onnx.data`](pca_transform.onnx.data) | ONNX external weights |

The ONNX export confirms the PCA transform uses only universally supported operations (`Sub`, `MatMul`).

---

## Key Findings (Quick Reference)

```
PCA fit time       : ~23–27 ms (one-time, CPU)
PCA transform time : 0.0056 ms per batch
Dimension before   : 61 (raw spike waveform)
Dimension after    : 6  (PCA-compressed)
Reduction ratio    : 10.2x
Reconstruction MSE : 0.821359
Variance explained : 16.9%
Deterministic      : YES (diff = 0.00e+00 across runs)
TT device status   : BLOCKED (Ethernet core timeout — needs board reset)
```
