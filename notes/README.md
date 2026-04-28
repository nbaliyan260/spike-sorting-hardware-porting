# notes/

Daily research logs, experiment results, and exported artifacts from the Tenstorrent PCA porting project.

---

## Daily Logs

| File | Day | Contents |
|------|-----|----------|
| [`day0_alignment.md`](day0_alignment.md) | Day 0 | Target stack confirmation (Tenstorrent, not TensorRT), role definition, machine access |
| [`day1_baseline_log.md`](day1_baseline_log.md) | Day 1 | Codebase reading, entry points, PCA bypass discovery |
| [`day2_pipeline_map.md`](day2_pipeline_map.md) | Day 2 | Complete forward path trace — all 6 stages, operators, portability assessment |
| [`day3_module_test.md`](day3_module_test.md) | Day 3 | PCA and filter module standalone test results |
| [`day4_backend_attempt.md`](day4_backend_attempt.md) | Day 4 | TT-NN operator compatibility analysis, ONNX export, split-architecture design |
| [`day5_tenstorrent_execution.md`](day5_tenstorrent_execution.md) | Day 5 | Real SSH session on `tt-blackhole-01` — full execution log and hardware blockers |
| [`final_week_summary.md`](final_week_summary.md) | Final | Consolidated week summary, team update, recommended next steps |

---

## Experiment Results (JSON)

| File | Description |
|------|-------------|
| [`pca_test_results.json`](pca_test_results.json) | 7-test PCA unit test output |
| [`filter_test_results.json`](filter_test_results.json) | 4-test filter module output |
| [`backend_attempt_results.json`](backend_attempt_results.json) | TT-NN operator analysis output |
| [`pca_quantitative_comparison.json`](pca_quantitative_comparison.json) | Before/after benchmark: dimension, memory, timing |
| [`ttnn_real_results.json`](ttnn_real_results.json) | Real results from `tt-blackhole-01` (2026-04-28) |

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
PCA transform time : 0.0054 ms per batch
Dimension before   : 61 (raw spike waveform)
Dimension after    : 6  (PCA-compressed)
Reduction ratio    : 10.2x
Reconstruction MSE : 0.821359
Variance explained : 16.9%
Deterministic      : YES (diff = 0.00e+00 across runs)
TT device status   : BLOCKED (Ethernet core timeout — needs board reset)
```
