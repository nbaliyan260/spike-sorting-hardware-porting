from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PursuitParams:
    """
    Closer Python approximation of the KS3 MATLAB recursive pursuit logic.

    Notes
    -----
    This is still not a literal line-by-line port of the MATLAB/GPU code, but it
    now mirrors the MATLAB control flow and heuristics much more closely than the
    earlier simplified version:
      - non-random projection search over low-order basis combinations
      - histogram-valley split scoring (find_split analogue)
      - EM-like refinement of a 1D two-component split
      - waveform-similarity veto with optional temporal rolls (wroll analogue)
      - optional CCG veto
      - retry after deflating a failed projection
      - recursive ``break_a_cluster`` style cluster extraction
    """

    min_cluster_size: int = 50  # nlow
    min_bimodality: float = 0.25  # rmin on min(rr)
    max_clusters: int = 1000
    n_em_iter: int = 50
    retry: int = 1

    # find_split / projection search params
    npow: int = 6
    nbase: int = 2
    hist_nbins: int = 1001
    hist_smooth: int = 10
    quantiles: Tuple[float, float] = (0.001, 0.999)

    # optional waveform roll matrices, shape (6, 6, n_rolls)
    # applied blockwise to wav2 reshaped as (6, n_features/6)
    wroll: Optional[np.ndarray] = None

    # Optional split veto using CCG dip (off by default)
    use_ccg: bool = False
    ccg_refrac_s: float = 0.0015
    ccg_window_s: float = 0.008
    ccg_dip_ratio_thresh: float = 0.2


def _smooth1d(x: np.ndarray, width: int) -> np.ndarray:
    width = int(max(1, width))
    if width <= 1:
        return x.astype(np.float32, copy=False)
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(x.astype(np.float32, copy=False), kernel, mode="same")


def _find_split(x: np.ndarray, params: PursuitParams):
    """Approximate MATLAB find_split.m."""
    x = np.asarray(x, dtype=np.float32)
    if x.size < 4 or not np.all(np.isfinite(x)):
        return np.array([0.0, 0.0], dtype=np.float32), 0.0, 0.5, float(np.mean(x) if x.size else 0.0), 0.0, 0.0, 1.0

    q0, q1 = params.quantiles
    xq = np.quantile(x, [q0, q1])
    if not np.isfinite(xq).all() or xq[0] >= xq[1]:
        mu = float(np.mean(x))
        var = float(np.var(x) + 1e-6)
        return np.array([0.0, 0.0], dtype=np.float32), 0.0, 0.5, mu, mu, mu, var

    bins = np.linspace(float(xq[0]), float(xq[1]), int(params.hist_nbins) + 1, dtype=np.float32)
    xhist, _ = np.histogram(x, bins=bins)
    xhist = _smooth1d(xhist.astype(np.float32), params.hist_smooth)
    nt = len(xhist)
    if nt < 3:
        mu = float(np.mean(x))
        var = float(np.var(x) + 1e-6)
        return np.array([0.0, 0.0], dtype=np.float32), 0.0, 0.5, mu, mu, mu, var

    ival = np.ones(nt, dtype=bool)
    ival[[0, -1]] = False
    ival[:-1] &= xhist[:-1] < xhist[1:]
    ival[1:] &= xhist[1:] < xhist[:-1]
    imin = np.flatnonzero(ival)
    xmean = float(np.mean(x))

    if imin.size == 0:
        isplit = int(np.argmin(np.abs(bins[:-1] - xmean)))
        scmax = 0.0
        m0 = xmean
    else:
        sc = np.zeros(imin.size, dtype=np.float32)
        for j, ii in enumerate(imin.tolist()):
            d1 = np.max(xhist[: ii + 1]) - xhist[ii]
            d2 = np.max(xhist[ii + 1 :]) - xhist[ii] if ii + 1 < nt else 0.0
            sc[j] = np.sqrt(max(0.0, float(d1 * d2)))
        imax = int(np.argmax(sc))
        scmax = float(sc[imax])
        isplit = int(imin[imax])
        m0 = float(bins[isplit])

    i1 = int(np.argmax(xhist[: isplit + 1]))
    i2 = isplit + 1 + int(np.argmax(xhist[isplit + 1 :])) if isplit + 1 < nt else isplit
    denom = np.maximum(xhist[[i1, i2]], 1e-6)
    r = 1.0 - xhist[isplit] / denom

    thr = float(bins[min(isplit, len(bins) - 2)])
    ix = x < thr
    p = float(np.mean(ix))
    if ix.all() or (~ix).all():
        mu = float(np.mean(x))
        var = float(max(np.var(x), 1e-6))
        return r.astype(np.float32), scmax, p, m0, mu, mu, var

    mu1 = float(np.mean(x[ix]))
    mu2 = float(np.mean(x[~ix]))
    sig = float(p * np.var(x[ix]) + (1.0 - p) * np.var(x[~ix]))
    sig = max(sig, float(np.var(x) / 10.0), 1e-6)
    return r.astype(np.float32), scmax, p, m0, mu1, mu2, sig


def _make_rproj(npow: int, nbase: int) -> np.ndarray:
    ncols = nbase ** max(0, npow - 1)
    u = np.zeros((npow, ncols), dtype=np.float32)
    for j in range(ncols):
        k = j
        u[0, j] = 1.0
        for i in range(1, npow):
            u[i, j] = float(k % nbase)
            k //= nbase
    return u


def _nonrandom_projection(clp: np.ndarray, params: PursuitParams) -> np.ndarray:
    npow = min(int(params.npow), clp.shape[1])
    if npow <= 0:
        return np.zeros((clp.shape[1],), dtype=np.float32)
    if npow == 1:
        out = np.zeros((clp.shape[1],), dtype=np.float32)
        out[0] = 1.0
        return out

    u = _make_rproj(npow, int(params.nbase)) - 0.5
    w = u.copy()
    scales = np.arange(1, npow + 1, dtype=np.float32)[:, None]
    w /= scales
    w /= np.linalg.norm(w, axis=0, keepdims=True) + 1e-8

    Xd = clp[:, :npow]
    scores = np.empty(w.shape[1], dtype=np.float32)
    for j in range(w.shape[1]):
        x = Xd @ w[:, j]
        _, scmax, *_ = _find_split(x, params)
        scores[j] = scmax

    imax = int(np.argmax(scores))
    out = np.zeros((clp.shape[1],), dtype=np.float32)
    out[:npow] = w[:, imax]
    return out


def _best_rolled_wav2(wav1: np.ndarray, wav2: np.ndarray, wroll: Optional[np.ndarray]) -> np.ndarray:
    if wroll is None:
        return wav2
    if wav2.size % 6 != 0:
        return wav2
    n_rolls = wroll.shape[2]
    base = wav2.reshape(6, wav2.size // 6)
    best = wav2
    best_err = float(np.mean((wav1 - wav2) ** 2))
    for j in range(n_rolls):
        cand = (wroll[:, :, j] @ base).reshape(-1)
        err = float(np.mean((wav1 - cand) ** 2))
        if err < best_err:
            best = cand
            best_err = err
    return best


def _ccg_dip_ratio(t0: np.ndarray, t1: np.ndarray, params: PursuitParams) -> float:
    if t0.size == 0 or t1.size == 0:
        return 1.0
    t0 = np.sort(np.asarray(t0, dtype=np.float64))
    t1 = np.sort(np.asarray(t1, dtype=np.float64))
    refr = float(params.ccg_refrac_s)
    w = float(params.ccg_window_s)
    j_lo = 0
    j_hi = 0
    count_refr = 0
    count_base = 0
    for ti in t0:
        lo = ti - w
        hi = ti + w
        while j_lo < t1.size and t1[j_lo] < lo:
            j_lo += 1
        if j_hi < j_lo:
            j_hi = j_lo
        while j_hi < t1.size and t1[j_hi] <= hi:
            j_hi += 1
        if j_lo == j_hi:
            continue
        adt = np.abs(t1[j_lo:j_hi] - ti)
        count_refr += int(np.sum(adt < refr))
        count_base += int(np.sum((adt >= refr) & (adt < 2.0 * refr)))
    return (count_refr + 1.0) / (count_base + 1.0)


def _bimodal_pursuit(
    Xd: np.ndarray,
    params: PursuitParams,
    times_s: Optional[np.ndarray],
    retry: int,
):
    """Closer analogue of MATLAB bimodal_pursuit.m."""
    Xd = np.asarray(Xd, dtype=np.float32)
    nsamp, nfeat = Xd.shape
    if nsamp < 2 * params.min_cluster_size:
        return None, None, False

    clp = Xd.copy()
    mu_clp = np.mean(clp, axis=0)
    clp -= mu_clp

    CC = (clp.T @ clp) / max(1, nsamp)
    try:
        evals, u = np.linalg.eigh(CC)
        order = np.argsort(evals)[::-1]
        u = u[:, order].astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        u = np.eye(nfeat, dtype=np.float32)
    clp = clp @ u

    preg = 10.0 / max(1, nsamp)
    nW = preg * 1.0 + (1.0 - preg) * np.sqrt(np.mean(clp ** 2, axis=0))
    clp = clp / (nW + 1e-8)

    w = _nonrandom_projection(clp, params)
    x = clp @ w
    r, _, p, _, mu1, mu2, sig = _find_split(x, params)

    rs = np.zeros((nsamp, 2), dtype=np.float32)
    for k in range(int(params.n_em_iter)):
        logp0 = -0.5 * np.log(1e-10 + sig) - ((x - mu1) ** 2) / (2.0 * sig) + np.log(1e-10 + p)
        logp1 = -0.5 * np.log(1e-10 + sig) - ((x - mu2) ** 2) / (2.0 * sig) + np.log(1e-10 + 1.0 - p)
        lmax = np.maximum(logp0, logp1)
        rs[:, 0] = np.exp(logp0 - lmax)
        rs[:, 1] = np.exp(logp1 - lmax)
        rs /= np.sum(rs, axis=1, keepdims=True) + 1e-10

        if k < params.n_em_iter // 2:
            r, _, p, _, mu1, mu2, sig = _find_split(x, params)
        else:
            p = float(np.mean(rs[:, 0]))
            mu1 = float((rs[:, 0] @ x) / (np.sum(rs[:, 0]) + 1e-10))
            mu2 = float((rs[:, 1] @ x) / (np.sum(rs[:, 1]) + 1e-10))
            sig = float((rs[:, 0] @ ((x - mu1) ** 2) + rs[:, 1] @ ((x - mu2) ** 2)) / max(1, nsamp))
            sig = max(sig, 1e-6)

        StMu = ((mu1 * rs[:, 0] + mu2 * rs[:, 1])[:, None] * clp).mean(axis=0)
        w = StMu.astype(np.float32, copy=False)
        nww = float(np.linalg.norm(w) + 1e-8)
        w /= nww
        x = clp @ w

    rr, _, _, _, _, _, _ = _find_split(x, params)
    w_back = ((w * nW) @ u.T).astype(np.float32, copy=False)
    wav1 = mu_clp + mu1 * w_back
    wav2 = mu_clp + mu2 * w_back

    m1 = float(np.linalg.norm(wav1) + 1e-8)
    m2 = float(np.linalg.norm(wav2) + 1e-8)
    if m2 > m1:
        rs = rs[:, [1, 0]]
        rr = rr[::-1]
        mu1, mu2 = mu2, mu1
        wav1, wav2 = wav2, wav1
        m1, m2 = m2, m1

    iclust = rs[:, 0] > 0.5
    n1 = int(np.sum(iclust))
    n2 = int(nsamp - n1)
    nmin = min(n1, n2)
    flag = bool(np.min(rr) >= params.min_bimodality and nmin >= params.min_cluster_size)

    if flag:
        wav2_best = _best_rolled_wav2(wav1, wav2, params.wroll)
        rc = float(np.dot(wav1, wav2_best) / ((np.linalg.norm(wav1) + 1e-8) * (np.linalg.norm(wav2_best) + 1e-8)))
        dmu = float(2.0 * abs(m1 - m2) / (m1 + m2 + 1e-8))
        if rc > 0.9 and dmu < 0.2:
            flag = False

    if flag and params.use_ccg and times_s is not None:
        ratio = _ccg_dip_ratio(np.asarray(times_s)[iclust], np.asarray(times_s)[~iclust], params)
        if ratio < params.ccg_dip_ratio_thresh:
            flag = False

    if (not flag) and retry > 0:
        w_unit = w / (np.linalg.norm(w) + 1e-8)
        clp_retry = Xd - np.outer(Xd @ w_unit, w_unit)
        return _bimodal_pursuit(clp_retry, params=params, times_s=times_s, retry=retry - 1)

    return x, iclust, flag


def _break_a_cluster(X: np.ndarray, params: PursuitParams, times_s: Optional[np.ndarray]):
    ix = np.arange(X.shape[0], dtype=np.int64)
    xold = None
    xnew = None
    for _ in range(10):
        if ix.size < 2 * params.min_cluster_size:
            xnew = None
            break
        xnew, iclust, flag = _bimodal_pursuit(X[ix], params=params, times_s=(times_s[ix] if times_s is not None else None), retry=params.retry)
        if not flag:
            break
        ix = ix[iclust]
        xold = xnew
    return ix, xold, xnew


def run_pursuit(X: np.ndarray, params: PursuitParams, times_s: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Closer analogue of KS3 run_pursuit.m.

    Repeatedly extracts one cluster at a time from the remaining spikes using
    break_a_cluster, then labels that extracted subset with a new cluster ID.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_spikes, n_features)")
    n = X.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    labels = np.zeros(n, dtype=np.int32)
    for j in range(1, int(params.max_clusters) + 1):
        ind = np.flatnonzero(labels == 0)
        if ind.size == 0:
            break
        ix, _, _ = _break_a_cluster(X[ind], params=params, times_s=(times_s[ind] if times_s is not None else None))
        labels[ind[ix]] = j
        if ix.size == ind.size:
            break

    # Any leftovers stay in the last/unassigned cluster exactly as MATLAB's
    # outer loop behavior tends to do once break_a_cluster returns all indices.
    # Remap to consecutive 0..K-1 for Python convenience.
    uniq = np.unique(labels)
    remap = {old: new for new, old in enumerate(uniq.tolist())}
    return np.array([remap[int(v)] for v in labels], dtype=np.int32)
