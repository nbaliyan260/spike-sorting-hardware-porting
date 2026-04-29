# Day 1 - Baseline Setup and First Run

## Date: 2026-04-28

## Environment Details

| Parameter | Value |
|-----------|-------|
| Machine | macOS (local dev) / 10.127.30.197 (remote `tt-blackhole-01`) |
| Python version | 3.10+ |
| Dependencies | `torch`, `torchaudio`, `scipy`, `sklearn` |
| Dataset | Simulated recordings (`simulated_recordings_npx_raw.bin`) |

## Codebase Orientation

### Cloned Repos
1. **torchbci-hardware-ports-torchbci-module** — This is our primary fork where all the porting work will live. It contains the modularized Kilosort4 logic.
   - Key files: `torchbci/algorithms/kilosort.py` (this is the massive 600+ line pipeline definition I need to untangle), `demo/kilosort_driver.py`
   
2. **torchbci-main** — The original baseline repository for reference.

### Initial Read-Through and Discovery
I spent today tracing execution from the `demo/kilosort_driver.py` entry point down into the actual algorithmic blocks in `Kilosort4Algorithm`. 

Here is what I found regarding the active pipeline path:
1. ✅ **CAR (Common Average Referencing)** — active
2. ✅ **Filtering** — active (uses a 300 Hz high-pass Butterworth)
3. ✅ **Whitening** (ZCA transform) — active
4. ✅ **Detection** (iterative template matching) — active
5. ⚠️ **PCA Feature Conversion** — **Instantiated but completely BYPASSED**. On line 544, it just assigns `spike_pc_features = spike_features`. This seems like a perfect entry point for me to fix and port.
6. ✅ **Clustering** (Simple Online K-Means) — active

### Current Blockers / Next Steps
- No major runtime errors yet since I'm just running code analysis locally.
- Next step is to map out exactly what operators each of these stages uses so I can determine what is actually portable to Tenstorrent.

## Exit Criteria Status
- [x] Code structure mapped out
- [x] Entry points found
- [x] PCA bypass identified (excellent candidate for my work)
- [ ] Remote execution on TT machine (pending access)
