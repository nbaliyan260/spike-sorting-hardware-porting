# import pickle
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# from torchbci.kilosort4.io import load_probe
# from torchbci.algorithms.kilosort_ported import KS4Pipeline

# # ---- config ----
# n_runs = 1  # Number of runs per dataset

# # File paths and sampling rates
# file_configs = [
#     # {
#     #     "path": "../data/c14/c14_npx_raw.bin",
#     #     "label": "C14",
#     #     "fs": 50023.89868747
#     # },
#     # {
#     #     "path": "../data/c27/c27_npx_raw.bin",
#     #     "label": "C27",
#     #     "fs": 50023.88788029
#     # },
#     {
#         "path": "../data/c46/c46_npx_raw.bin",
#         "label": "C46",
#         "fs": 50023.87552924
#     }
# ]
# stages = ["preproc", "drift", "st0", "clu0", "st", "clu", "merge", "postproc"]

# # ---- storage for all recordings (ported KS4) ----
# all_results = {}  # will mirror all_recordings_results from original KS4

# # Run the pipeline multiple times for each dataset
# for config in file_configs:
#     print(f"\n{'='*60}")
#     print(f"Processing Recording: {config['label']}")
#     print(f"Path: {config['path']}")
#     print(f"Running {n_runs} iterations...")
#     print(f"{'='*60}")
    
#     # Per-recording storage (per run)
#     runtime_total = []
#     runtime_stage = {s: [] for s in stages}
#     cpu_util = {s: [] for s in stages}
#     gpu_mem_pct = {s: [] for s in stages}
#     cpu_mem_used = {s: [] for s in stages}
#     gpu_mem_used = {s: [] for s in stages}

#     settings = {
#         "n_chan_bin": 384,
#         "fs": config['fs'],
#         "filename": config['path'],
#     }

#     for run_idx in range(n_runs):
#         print(f"\nRun {run_idx + 1}/{n_runs} for {config['label']}")
        
#         probe = load_probe("../data/chanMap.mat")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # device = torch.device("cpu")
#         results_dir = Path("./results_temp") / f"{config['label']}_run_{run_idx}"
#         results_dir.mkdir(parents=True, exist_ok=True)

#         pipeline = KS4Pipeline(
#             settings=settings,
#             probe=probe,
#             results_dir=str(results_dir),
#             device=device,
#             deterministic_mode=False,
#             optimize_memory=True
#         )

#         with torch.no_grad():
#             out = pipeline()

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from torchbci.kilosort4.io import load_probe
from torchbci.algorithms.kilosort_ported import KS4Pipeline

# ---- config ----
n_runs = 1  # Number of runs per dataset

# File paths and sampling rates
file_configs = [
    # {
    #     "path": "../data/c14/c14_npx_raw.bin",
    #     "label": "C14",
    #     "fs": 50023.89868747
    # },
    # {
    #     "path": "../data/c27/c27_npx_raw.bin",
    #     "label": "C27",
    #     "fs": 50023.88788029
    # },
    {
        "path": "path/to/raw_1pct.bin",
        "label": "C46",
        "fs": 50023.87552924
    }
]
stages = ["preproc", "drift", "st0", "clu0", "st", "clu", "merge", "postproc"]

# ---- storage for all recordings (ported KS4) ----
all_results = {}  # will mirror all_recordings_results from original KS4

# Run the pipeline multiple times for each dataset
for config in file_configs:
    print(f"\n{'='*60}")
    print(f"Processing Recording: {config['label']}")
    print(f"Path: {config['path']}")
    print(f"Running {n_runs} iterations...")
    print(f"{'='*60}")
    
    # Per-recording storage (per run)
    runtime_total = []
    runtime_stage = {s: [] for s in stages}
    cpu_util = {s: [] for s in stages}
    gpu_mem_pct = {s: [] for s in stages}
    cpu_mem_used = {s: [] for s in stages}
    gpu_mem_used = {s: [] for s in stages}

    settings = {
        "n_chan_bin": 384,
        "fs": config['fs'],
        "filename": config['path'],
    }

    for run_idx in range(n_runs):
        print(f"\nRun {run_idx + 1}/{n_runs} for {config['label']}")
        
        probe = load_probe("../data/chanMap.mat")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        results_dir = Path("./results_cpu_delete") / f"{config['label']}_run_{run_idx}"
        results_dir.mkdir(parents=True, exist_ok=True)

        pipeline = KS4Pipeline(
            settings=settings,
            probe=probe,
            results_dir=str(results_dir),
            device=device,
            deterministic_mode=2,
            optimize_memory=True,
            seed=42
        )

        with torch.no_grad():
            out = pipeline()

        ops = out["ops"]
        runtime = ops["runtime"]
        runtime_total.append(runtime)

        # Per-stage metrics (same pattern as original KS4)
        for s in stages:
            # These keys mirror your original script:
            # ops[f"runtime_{s}"], ops[f"usage_{s}"]["cpu"], ops[f"usage_{s}"]["gpu"]
            usage_cpu = ops[f"usage_{s}"]["cpu"]
            usage_gpu = ops[f"usage_{s}"]["gpu"]

            runtime_stage[s].append(ops[f"runtime_{s}"])
            cpu_util[s].append(usage_cpu["util"])
            gpu_mem_pct[s].append(usage_gpu["mem_pct"])
            cpu_mem_used[s].append(usage_cpu["mem_used"])
            gpu_mem_used[s].append(usage_gpu["mem_used"])
        
        print(f"  Runtime: {runtime:.2f}s")
    
    # Convert to numpy arrays and calculate statistics (just like original KS4)
    runtime_total = np.array(runtime_total)
    runtime_stage_all = {s: np.array(runtime_stage[s]) for s in stages}

    label = config['label']
    all_results[label] = {
        'runtime_total': runtime_total,
        'runtime_total_mean': np.mean(runtime_total),
        'runtime_total_std': np.std(runtime_total),
        'runtime_total_min': np.min(runtime_total),
        'runtime_total_max': np.max(runtime_total),
        
        # Per-stage data
        'runtime_stage': runtime_stage_all,
        'runtime_stage_mean': {s: np.mean(runtime_stage[s]) for s in stages},
        'runtime_stage_std': {s: np.std(runtime_stage[s]) for s in stages},
        
        'cpu_util': {s: np.array(cpu_util[s]) for s in stages},
        'cpu_util_mean': {s: np.mean(cpu_util[s]) for s in stages},
        
        'gpu_mem_pct': {s: np.array(gpu_mem_pct[s]) for s in stages},
        'gpu_mem_pct_mean': {s: np.mean(gpu_mem_pct[s]) for s in stages},
        
        'cpu_mem_used': {s: np.array(cpu_mem_used[s]) for s in stages},
        'cpu_mem_used_mean': {s: np.mean(cpu_mem_used[s]) for s in stages},
        
        'gpu_mem_used': {s: np.array(gpu_mem_used[s]) for s in stages},
        'gpu_mem_used_mean': {s: np.mean(gpu_mem_used[s]) for s in stages},
        
        # Metadata
        'n_runs': n_runs,
        'config': config,
    }

    print(f"\n{label} Summary (ported KS4):")
    print(f"  Mean Runtime: {all_results[label]['runtime_total_mean']:.2f}s")
    print(f"  Std Runtime:  {all_results[label]['runtime_total_std']:.2f}s")

# ---- Save results for the ported KS4 (mirrors original_ks4_results.pkl) ----
print("\n" + "="*60)
print("SAVING PORTED KS4 RESULTS")
print("="*60)

# output_dir = Path("./kilosort_results")
# output_dir.mkdir(parents=True, exist_ok=True)

# ported_pickle_file = output_dir / "ported_ks4_results.pkl"
# with open(ported_pickle_file, 'wb') as f:
#     pickle.dump(all_results, f)
# print(f"✓ Saved pickle file: {ported_pickle_file}")