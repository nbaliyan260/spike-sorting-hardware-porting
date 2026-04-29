# Day 0 - Alignment and Access

## Date: 2026-04-28

## Decision Log

### Target Stack Confirmation
- **Target stack:** Tenstorrent (TT-NN / TT-XLA / TT-Metalium)
- **Reasoning:** Based on team discussions, the project report explicitly mentions "Tenstorrent-class hardware" as a target. The exploratory non-AMD lane is intended to investigate Tenstorrent feasibility, not NVIDIA TensorRT (which would be redundant with the existing CUDA baseline).
- **Clarification sent to team:** "Please confirm my exact target for this week: NVIDIA TensorRT or actual Tenstorrent stack (TT-NN / TT-XLA). I will proceed on one target only."
- **Assumption (if no response):** Proceeding with Tenstorrent as the primary target, since the project report and course scope explicitly name it.

### Active Repo / Branch
- **Active repo:** `torchbci-hardware-ports-torchbci-module` (the hardware-port fork)
- **Main repo:** `torchbci-main` (original baseline)
- **Branch:** `main` (default branch in both repos)
- **Note:** The hardware-port fork contains additional modules (`kilosort4/`, `datasets/`, `rtsort/`, `kilosort3/`) not present in `torchbci-main`. All Kilosort work should reference the hardware-port fork.

### Dataset Path
- **Dataset:** Simulated recordings (`simulated_recordings_npx_raw.bin`)
- **Expected location:** `../data/simulated_recordings/simulated_recordings_npx_raw.bin` (relative to demo/)
- **Status:** Data files need to be verified on the Tenstorrent machine
- **Sampling rate:** 50023.87552924 Hz

### Primary Machine
- **Local machine:** macOS (Apple Silicon / Intel) — for code reading, pipeline mapping, script development
- **Remote machine:** `nazishbaliyan@10.127.30.197` — Tenstorrent-connected server for actual experiments
- **Access method:** SSH with password authentication

### Expected First Deliverable
- Baseline run log (success or failure with exact error)
- Pipeline map of active forward path
- Standalone module test for PCA feature conversion

## Summary
| Item | Value |
|------|-------|
| Target stack | Tenstorrent (TT-NN / TT-XLA) |
| Active repo/branch | torchbci-hardware-ports-torchbci-module / main |
| Dataset path | ../data/simulated_recordings/simulated_recordings_npx_raw.bin |
| Primary machine | 10.127.30.197 (remote) + macOS (local dev) |
| Expected first deliverable | Baseline run + pipeline map + module test |
