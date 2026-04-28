import numpy as np
import os
from scipy.optimize import linear_sum_assignment

def load_run(path):
    st = np.load(f"{path}/spike_times.npy").astype(np.int64)
    sc = np.load(f"{path}/spike_clusters.npy").astype(np.int64)
    units = np.unique(sc)
    return st, sc, units

def match_with_tolerance(a_times, b_times, tol):
    # both sorted
    i = j = 0
    matches = 0
    while i < len(a_times) and j < len(b_times):
        da = a_times[i]
        db = b_times[j]
        if abs(da - db) <= tol:
            matches += 1
            i += 1
            j += 1
        elif da < db:
            i += 1
        else:
            j += 1
    return matches

# Similarity metric: F1 score from matches, na spikes in unit A, nb spikes in unit B
def f1_from_matches(matches, na, nb):
    if na == 0 or nb == 0:
        return 0.0
    p = matches / nb
    r = matches / na
    return 0.0 if (p + r) == 0 else 2*p*r/(p+r)

def unit_f1_matrix(stA, scA, unitsA, stB, scB, unitsB, tol):
    # pre-split spike times per unit
    timesA = {u: np.sort(stA[scA == u]) for u in unitsA}
    timesB = {u: np.sort(stB[scB == u]) for u in unitsB}

    # Similarity score matrix between units, O(Na * Nb)
    M = np.zeros((len(unitsA), len(unitsB)), dtype=np.float32)
    for i,u in enumerate(unitsA):
        a = timesA[u]
        for j,v in enumerate(unitsB):
            b = timesB[v]
            m = match_with_tolerance(a, b, tol)
            M[i,j] = f1_from_matches(m, len(a), len(b))
    return M

def reproducibility_between_runs(pathA, pathB, tol_samples, f1_thresh=0.95):
    stA, scA, unitsA = load_run(pathA)
    stB, scB, unitsB = load_run(pathB)

    F = unit_f1_matrix(stA, scA, unitsA, stB, scB, unitsB, tol_samples)
    # Hungarian wants a cost matrix; maximize F1 -> minimize (1 - F1)
    # It returns map of the min(len(unitsA), len(unitsB)) units
    row, col = linear_sum_assignment(1.0 - F)
    # print(row, col)
    matched_f1 = F[row, col]
    # print(matched_f1)
    # Computes the mean F1 over all matched units, penalize when we have outliers
    R_F1   = float(np.mean(matched_f1))   if matched_f1.size else 0.0

    matched_good = (matched_f1 > f1_thresh).sum()
    R_units = matched_good / (0.5*(len(unitsA) + len(unitsB)))

    # global spike-time determinism
    allA = np.sort(stA)
    allB = np.sort(stB)
    m_all = match_with_tolerance(allA, allB, tol_samples)
    R_spikes = f1_from_matches(m_all, len(allA), len(allB))

    reproducibility_score = 0.4*R_units + 0.4*R_F1 + 0.2*R_spikes
    return dict(reproducibility_score=reproducibility_score, R_units=R_units, R_F1=R_F1, R_spikes=R_spikes)

c14_paths = [
    "path/to/C14_run_seed_0_no_det_opt",
    "path/to/C14_run_seed_1_no_det_opt",
    "path/to/C14_run_seed_42_no_det_opt",
    "path/to/C14_run_seed_123_no_det_opt",
    "path/to/C14_run_seed_1234_no_det_opt"
]

c27_paths = [
    "path/to/C27_run_seed_0_no_det_opt",
    "path/to/C27_run_seed_1_no_det_opt",
    "path/to/C27_run_seed_42_no_det_opt",
    "path/to/C27_run_seed_123_no_det_opt",
    "path/to/C27_run_seed_1234_no_det_opt"
]
c46_paths = [
    "path/to/C46_run_seed_0_no_det_opt",
    "path/to/C46_run_seed_1_no_det_opt",
    "path/to/C46_run_seed_42_no_det_opt",
    "path/to/C46_run_seed_123_no_det_opt",
    "path/to/C46_run_seed_1234_no_det_opt"
]

out_path_27 = './results/C27_reproducibility_stability_exp_ks3.txt'
out_path_46 = './results/C46_reproducibility_stability_exp_ks3.txt'
out_path_14 = './results/C14_reproducibility_stability_exp_ks3.txt'
repro_scores = []
lines = []

for i in range(len(c14_paths)):
    for j in range(i + 1, len(c14_paths)):
        pathA = c14_paths[i]
        pathB = c14_paths[j]
        print(f"Comparing {os.path.basename(pathA)} and {os.path.basename(pathB)}...")  
        r = reproducibility_between_runs(pathA, pathB, tol_samples=1, f1_thresh=0.95)

        repro_scores.append(r['reproducibility_score'])

        header = f"Reproducibility between {os.path.basename(pathA)} and {os.path.basename(pathB)}:"
        lines.append(header)
        lines.append(f"  reproducibility score: {r['reproducibility_score']:.3f}")
        lines.append(f"  R_units: {r['R_units']:.3f}")
        lines.append(f"  R_F1:    {r['R_F1']:.3f}")
        lines.append(f"  R_spikes:{r['R_spikes']:.3f}")
        lines.append("")

mean_repro = float(np.mean(repro_scores)) if repro_scores else 0.0
lines.append(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")

with open(out_path_14, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote results to {out_path_14}")
print(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")


repro_scores = []
lines = []

for i in range(len(c27_paths)):
    for j in range(i + 1, len(c27_paths)):
        pathA = c27_paths[i]
        pathB = c27_paths[j]
        print(f"Comparing {os.path.basename(pathA)} and {os.path.basename(pathB)}...")  
        r = reproducibility_between_runs(pathA, pathB, tol_samples=1, f1_thresh=0.95)

        repro_scores.append(r['reproducibility_score'])

        header = f"Reproducibility between {os.path.basename(pathA)} and {os.path.basename(pathB)}:"
        lines.append(header)
        lines.append(f"  reproducibility score: {r['reproducibility_score']:.3f}")
        lines.append(f"  R_units: {r['R_units']:.3f}")
        lines.append(f"  R_F1:    {r['R_F1']:.3f}")
        lines.append(f"  R_spikes:{r['R_spikes']:.3f}")
        lines.append("")

mean_repro = float(np.mean(repro_scores)) if repro_scores else 0.0
lines.append(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")

with open(out_path_27, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote results to {out_path_27}")
print(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")

repro_scores = []
lines = []

for i in range(len(c46_paths)):
    for j in range(i + 1, len(c46_paths)):
        pathA = c46_paths[i]
        pathB = c46_paths[j]
        print(f"Comparing {os.path.basename(pathA)} and {os.path.basename(pathB)}...")  
        r = reproducibility_between_runs(pathA, pathB, tol_samples=1, f1_thresh=0.95)

        repro_scores.append(r['reproducibility_score'])

        header = f"Reproducibility between {os.path.basename(pathA)} and {os.path.basename(pathB)}:"
        lines.append(header)
        lines.append(f"  reproducibility score: {r['reproducibility_score']:.3f}")
        lines.append(f"  R_units: {r['R_units']:.3f}")
        lines.append(f"  R_F1:    {r['R_F1']:.3f}")
        lines.append(f"  R_spikes:{r['R_spikes']:.3f}")
        lines.append("")

mean_repro = float(np.mean(repro_scores)) if repro_scores else 0.0
lines.append(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")

with open(out_path_46, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote results to {out_path_46}")
print(f"Mean reproducibility score over {len(repro_scores)} pairs: {mean_repro:.3f}")
