# Day 2 - Pipeline Mapping

## Date: 2026-04-28

## Active Pipeline Path Analysis

### Source File
`torchbci/algorithms/kilosort.py` — `Kilosort4Algorithm` class (lines 474–623)

### Forward Path (Batch Mode — `forward()`)
The main execution path follows `forward(x, batch_no, is_last)`:

```
Input: x [C, N] (channels × samples)
  │
  ├── 1. CAR (Kilosort4CAR.forward)
  │       x = x - mean(x, dim=0)
  │       Ops: torch.mean, subtraction
  │       File: kilosort.py:14-22
  │       Status: ✅ ACTIVE
  │
  ├── 2. Filtering (Kilosort4Filtering.forward)
  │       4th-order Butterworth high-pass (300 Hz)
  │       Ops: scipy.signal.butter, scipy.signal.sosfiltfilt, np↔torch conversion
  │       File: kilosort.py:24-42
  │       Status: ✅ ACTIVE
  │       ⚠️ Uses scipy (not pure PyTorch!) — major portability blocker
  │
  ├── 3. Whitening (Kilosort4Whitening.forward)
  │       Per-channel ZCA whitening using N nearest channels
  │       Ops: torch.cov, torch.linalg.svd, torch.diag, matrix multiply
  │       File: kilosort.py:44-72
  │       Status: ✅ ACTIVE
  │       ⚠️ Uses torch.linalg.svd — may need special handling
  │
  ├── 4. Detection (Kilosort4Detection.iterative_spike_detection)
  │       Iterative template matching + spike extraction + residual subtraction
  │       Ops: F.conv1d, torch.linalg.vector_norm, cosine similarity, data slicing
  │       File: kilosort.py:126-325
  │       Status: ✅ ACTIVE
  │       Complex: iterative loop with data-dependent control flow
  │
  ├── 5. PCA Feature Conversion (Kilosort4PCFeatureConversion)
  │       PCA-based dimensionality reduction
  │       Ops: torch.pca_lowrank / torch.linalg.svd, matrix multiply
  │       File: kilosort.py:341-472
  │       Status: ⚠️ INSTANTIATED BUT BYPASSED
  │       Line 544: spike_pc_features = spike_features (skips PCA)
  │       Line 525: self.pc_featuring = Kilosort4PCFeatureConversion(...)
  │
  └── 6. Clustering (SimpleOnlineKMeansClustering.forward)
          Online K-Means with EMA centroid updates
          Ops: distance computation, argmin, EMA update
          File: clustering.py:17-157
          Status: ✅ ACTIVE (runs on last batch only)
```

### Data Flow Through Batching
```
run(dataloader) → run_one_batch(batch, i, total) → forward(x, batch_no, is_last)

Batching uses KilosortDataset:
- window_samples: sliding window size
- hop_samples: stride between windows  
- margin: overlap for boundary effects
- Core spike indices adjusted: spikes[:, 1].add_(batch_no * H)

Clustering: accumulated across all batches, runs only on is_last=True
```

## Pipeline Map Table

| # | Stage | Class | File:Line | Active? | Compute-Heavy? | Export-Friendly? | Likely Blocker |
|---|-------|-------|-----------|---------|----------------|------------------|----------------|
| 1 | CAR | `Kilosort4CAR` | kilosort.py:14 | ✅ Yes | Low | ✅ Very easy | None |
| 2 | High-Pass Filter | `Kilosort4Filtering` | kilosort.py:24 | ✅ Yes | Medium | ❌ scipy dependency | scipy.signal not portable |
| 3 | Whitening | `Kilosort4Whitening` | kilosort.py:44 | ✅ Yes | High | ⚠️ Moderate | torch.linalg.svd, per-channel loop |
| 4 | Detection | `Kilosort4Detection` | kilosort.py:126 | ✅ Yes | Very High | ❌ Hard | Iterative loops, data-dependent flow |
| 5 | PCA Features | `Kilosort4PCFeatureConversion` | kilosort.py:341 | ❌ Bypassed | Medium | ✅ Good | torch.pca_lowrank |
| 6 | Clustering | `SimpleOnlineKMeansClustering` | clustering.py:17 | ✅ Yes | Medium | ⚠️ Moderate | Dynamic state, Python lists |

## Supporting Modules (Not in Main Path but Available)

| Module | Class | File | Status | Notes |
|--------|-------|------|--------|-------|
| Feature Selection | `JimsFeatureSelection` | featureselection.py | Available | Used by Jim's pipeline, not Kilosort4 |
| Template Matching | `JimsTemplateMatching` | templatematching.py | Available | Used by Jim's pipeline |
| Alignment | `JimsAlignment` | alignment.py | Available | Used by Jim's pipeline |
| Delay & Decay | `delay_and_decay()` | functional.py | ✅ Active | Used by detection for spike propagation |

## Key Operator Inventory

| Operator | Used By | TT-NN Likely Support | Notes |
|----------|---------|---------------------|-------|
| `torch.mean` | CAR | ✅ Yes | Basic reduce op |
| `scipy.signal.butter/sosfiltfilt` | Filtering | ❌ No | Not PyTorch at all |
| `torch.cov` | Whitening | ⚠️ Maybe | Compound op |
| `torch.linalg.svd` | Whitening, PCA | ⚠️ Maybe | Linear algebra |
| `torch.pca_lowrank` | PCA | ⚠️ Maybe | Decomposition |
| `F.conv1d` | Detection | ✅ Likely | Standard convolution |
| `torch.linalg.vector_norm` | Detection | ⚠️ Maybe | Norm computation |
| `F.cosine_similarity` | Detection | ⚠️ Maybe | Compound op |

## Module Selection Decision

### Primary Target: `Kilosort4PCFeatureConversion` (PCA)
**Why:**
1. Self-contained — has clear `fit()` / `transform()` / `forward()` API
2. Pure PyTorch ops — no scipy dependency unlike filtering
3. Standard linear algebra — uses `torch.pca_lowrank` or `torch.linalg.svd`
4. Easy to isolate — takes `[N, D]` input, produces `[N, K]` output
5. Well-documented internal structure with buffers for learned state
6. Even though bypassed in pipeline, it's the cleanest test target

### Backup Target: `Kilosort4Filtering` (High-Pass Filter)
**Why:**
1. Active in pipeline (gives more relevance)
2. But uses scipy — would need a pure PyTorch rewrite first
3. Conceptually simple (Butterworth filter)
4. Harder to port due to scipy dependency

### Decision
> **Primary module: PCA Feature Conversion (`Kilosort4PCFeatureConversion`)**
> **Backup module: Filtering (`Kilosort4Filtering`)**
> **Reason: PCA is the smallest, most self-contained, pure-PyTorch module suitable for backend conversion experiments.**

## Exit Criteria Status
- [x] Main algorithm file read and understood
- [x] Forward path traced completely
- [x] All stages documented with active/inactive status
- [x] PCA bypass identified and documented
- [x] One primary module chosen (PCA)
- [x] One backup module chosen (Filtering)
- [x] Clear rationale provided
