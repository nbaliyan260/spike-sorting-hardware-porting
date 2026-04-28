from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from .recursive_pursuit import PursuitParams, run_pursuit


def _as_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _as_torch(a, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype if dtype is not None else a.dtype)
    return torch.as_tensor(a, device=device, dtype=dtype)


def _get_pursuit_params_from_ops(ops: Dict[str, Any]) -> PursuitParams:
    s = ops.get("settings", ops)
    return PursuitParams(
        min_cluster_size=int(s.get("ks3_min_cluster_size", 200)),
        min_bimodality=float(s.get("ks3_min_bimodality", s.get("ks3_score_threshold", 0.25))),
        max_clusters=int(s.get("ks3_max_clusters", s.get("ks3_max_splits", 512))),
        n_em_iter=int(s.get("ks3_n_em_iter", 50)),
        retry=int(s.get("ks3_retry", 1)),
        npow=int(s.get("ks3_npow", 6)),
        nbase=int(s.get("ks3_nbase", 2)),
        hist_nbins=int(s.get("ks3_hist_nbins", 1001)),
        hist_smooth=int(s.get("ks3_hist_smooth", 10)),
        quantiles=tuple(s.get("ks3_quantiles", (0.001, 0.999))),
        wroll=s.get("ks3_wroll", None),
        use_ccg=bool(s.get("ks3_use_ccg", False)),
        ccg_refrac_s=float(s.get("ks3_ccg_refrac_s", 0.0015)),
        ccg_window_s=float(s.get("ks3_ccg_window_s", 0.008)),
        ccg_dip_ratio_thresh=float(s.get("ks3_ccg_dip_ratio_thresh", 0.2)),
    )


def _centers_grid(xcup: np.ndarray, ycup: np.ndarray, dmin: float, dminx: float) -> Tuple[np.ndarray, np.ndarray]:
    # KS3 uses centers spaced by 2*dmin (and 2*dminx) so neighborhoods don't overlap.
    y0 = float(np.min(ycup)) + dmin - 1.0
    y1 = float(np.max(ycup)) + dmin + 1.0
    x0 = float(np.min(xcup)) + dminx - 1.0
    x1 = float(np.max(xcup)) + dminx + 1.0

    ycent = np.arange(y0, y1 + 1e-6, 2.0 * dmin, dtype=np.float32)
    xcent = np.arange(x0, x1 + 1e-6, 2.0 * dminx, dtype=np.float32)
    return xcent, ycent


def _assemble_dd(
    tF_group: np.ndarray,          # (n_spikes, nC, n_pcs)
    pid_group: np.ndarray,         # (n_spikes,) template ids
    itemp: np.ndarray,             # (n_templates_in_group,)
    iC: np.ndarray,                # (nC, n_templates_total)
    n_pcs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build KS3-style dd array: (n_spikes, n_pcs, n_union_chans)
    plus the union channel index array (ich).
    """
    # Union of channels across templates in this spatial center
    ich = np.unique(iC[:, itemp].reshape(-1))
    ich = ich[ich >= 0]  # just in case
    ich = ich.astype(np.int64)

    nsp = tF_group.shape[0]
    dd = np.zeros((nsp, n_pcs, ich.size), dtype=np.float32)

    # map each template's local channels into union channel positions
    # (ich is sorted, so searchsorted is valid)
    for t in itemp:
        spk_idx = np.flatnonzero(pid_group == t)
        if spk_idx.size == 0:
            continue

        chans = iC[:, t].astype(np.int64, copy=False)
        valid = chans >= 0
        if not np.any(valid):
            continue

        chans = chans[valid]
        pos = np.searchsorted(ich, chans)

        # Use np.ix_ to avoid mixed advanced-index broadcasting between
        # spike and channel index arrays.
        dd[np.ix_(spk_idx, np.arange(n_pcs), pos)] = np.transpose(
            tF_group[spk_idx][:, valid, :], (0, 2, 1)
        )

    return dd, ich


def _assemble_dd_torch(
    tF_group: torch.Tensor,        # (n_spikes, nC, n_pcs)
    pid_group: torch.Tensor,       # (n_spikes,)
    itemp: np.ndarray,             # (n_templates_in_group,)
    iC: torch.Tensor,              # (nC, n_templates_total)
    n_pcs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Torch version of _assemble_dd used by optimize_memory mode.
    """
    if not tF_group.is_floating_point():
        raise TypeError(f"tF_group must be floating point, got {tF_group.dtype}.")

    itemp_t = torch.as_tensor(itemp, device=iC.device, dtype=torch.long)
    ich = torch.unique(iC[:, itemp_t].reshape(-1), sorted=True)
    ich = ich[ich >= 0].to(torch.long)

    nsp = int(tF_group.shape[0])
    dd = torch.zeros(
        (nsp, n_pcs, int(ich.numel())),
        dtype=tF_group.dtype,
        device=tF_group.device,
    )

    for t in itemp.tolist():
        spk_idx = torch.nonzero(pid_group == int(t), as_tuple=False).flatten()
        if spk_idx.numel() == 0:
            continue

        chans = iC[:, int(t)].to(torch.long)
        valid = chans >= 0
        if not torch.any(valid):
            continue

        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        chans = chans[valid]
        pos = torch.searchsorted(ich, chans)

        for j in range(int(pos.numel())):
            dd[spk_idx, :, int(pos[j].item())] = tF_group[
                spk_idx, int(valid_idx[j].item()), :
            ].to(dtype=dd.dtype)

    return dd, ich


# TODO: Most probably this is not correct and needs working on it
def _cluster_one_pass_optimized(
    ops: Dict[str, Any],
    st: torch.Tensor,
    tF: torch.Tensor,
    mode: str,
) -> Tuple[np.ndarray, torch.Tensor]:
    if not torch.is_tensor(st) or not torch.is_tensor(tF):
        raise TypeError(
            "KS3 optimize_memory=True requires 'st' and 'tF' to be torch.Tensor "
            "inputs already placed on the target device."
        )

    if st.device != tF.device:
        raise ValueError(
            f"KS3 optimize_memory=True requires st and tF on same device; got "
            f"st={st.device}, tF={tF.device}."
        )

    s = ops["settings"]
    fs = float(s["fs"])
    n_pcs = int(s["n_pcs"])
    Nchan = int(ops.get("Nchan", len(ops["chanMap"])))
    device = tF.device

    # Which "template id" column do we cluster around?
    if mode == "spikes":
        pid = st[:, 5].to(torch.long)
        times_s = st[:, 0].to(torch.float64)  # already seconds
        iC = _as_torch(ops["iC"], device=device, dtype=torch.long)

        xcup = _as_numpy(ops.get("xcup", None))
        ycup = _as_numpy(ops.get("ycup", None))
        if xcup is None or ycup is None:
            xup = _as_numpy(ops.get("xup", ops["xc"]))
            yup = _as_numpy(ops.get("yup", ops["yc"]))
            X, Y = np.meshgrid(xup, yup)
            xcup = X.reshape(-1)
            ycup = Y.reshape(-1)
    else:
        pid = st[:, 1].to(torch.long)
        times_s = st[:, 0].to(torch.float64) / fs

        if "iU" not in ops or "iCC" not in ops:
            raise ValueError(
                "KS3 final clustering requires ops['iU'] and ops['iCC'], which should "
                "be populated by the template extraction step (template_matching.extract)."
            )
        iU = _as_torch(ops["iU"], device=device, dtype=torch.long)
        iCC = _as_torch(ops["iCC"], device=device, dtype=torch.long)
        iC = iCC.index_select(1, iU)

        xc = _as_torch(ops["xc"], device=device, dtype=torch.float32)
        yc = _as_torch(ops["yc"], device=device, dtype=torch.float32)
        xcup = xc.index_select(0, iU).detach().cpu().numpy()
        ycup = yc.index_select(0, iU).detach().cpu().numpy()

    dmin = float(ops.get("dmin", s.get("dmin", 0)) or 0)
    if dmin <= 0:
        yuniq = np.unique(_as_numpy(ops["yc"]))
        dmin = float(np.median(np.diff(np.sort(yuniq))))
    dminx = float(ops.get("dminx", s.get("dminx", 32.0)))

    xcent, ycent = _centers_grid(xcup, ycup, dmin=dmin, dminx=dminx)

    clu = torch.full((pid.shape[0],), -1, dtype=torch.int32, device=device)
    Wall_list = []

    params = _get_pursuit_params_from_ops(ops)
    next_unit = 0

    for y0 in ycent:
        for x0 in xcent:
            itemp = np.flatnonzero((np.abs(ycup - y0) < dmin) & (np.abs(xcup - x0) < dminx))
            if itemp.size == 0:
                continue

            itemp_t = torch.as_tensor(itemp, device=device, dtype=torch.long)
            in_group = torch.isin(pid, itemp_t)
            if not torch.any(in_group):
                continue

            idx = torch.nonzero(in_group, as_tuple=False).flatten()
            pid_g = pid.index_select(0, idx)
            tF_g = tF.index_select(0, idx)
            t_g = times_s.index_select(0, idx)

            dd_t, ich_t = _assemble_dd_torch(tF_g, pid_g, itemp, iC, n_pcs=n_pcs)
            dd = dd_t.detach().cpu().numpy()
            X = dd.reshape(dd.shape[0], -1)

            lab = run_pursuit(
                X, params=params, times_s=(t_g.detach().cpu().numpy() if params.use_ccg else None)
            )

            for lk in range(int(lab.max()) + 1):
                mk = np.flatnonzero(lab == lk)
                if mk.size == 0:
                    continue
                mk_t = torch.as_tensor(mk, device=device, dtype=torch.long)
                clu[idx.index_select(0, mk_t)] = next_unit
                next_unit += 1

            Wg = _dd_to_templates_full(
                dd, lab, ich_t.detach().cpu().numpy(), Nchan=Nchan, n_pcs=n_pcs
            )
            if Wg.shape[0] > 0:
                Wall_list.append(Wg)

    un = torch.nonzero(clu < 0, as_tuple=False).flatten()
    if un.numel() > 0:
        for i_t in un:
            i = int(i_t.item())
            clu[i] = next_unit
            next_unit += 1

            tmp = np.zeros((Nchan, n_pcs), dtype=np.float32)
            pid_i = int(pid[i].item())
            if 0 <= pid_i < iC.shape[1]:
                chans = iC[:, pid_i].to(torch.long)
                valid = chans >= 0
                if torch.any(valid):
                    chans_np = chans[valid].detach().cpu().numpy().astype(np.int64, copy=False)
                    vals_np = tF[i, valid, :].detach().cpu().numpy().astype(np.float32, copy=False)
                    tmp[chans_np] = vals_np
            Wall_list.append(tmp[None, :, :])

    Wall = np.concatenate(Wall_list, axis=0) if len(Wall_list) else np.zeros((0, Nchan, n_pcs), dtype=np.float32)
    Wall_torch = torch.from_numpy(Wall).to(device=device, dtype=torch.float32)

    clu_np = clu.detach().cpu().numpy().astype(np.int32, copy=False)
    uniq = np.unique(clu_np)
    remap = {old: new for new, old in enumerate(uniq.tolist())}
    clu_np = np.array([remap[int(x)] for x in clu_np], dtype=np.int32)

    return clu_np, Wall_torch


def _dd_to_templates_full(
    dd: np.ndarray,        # (nsp, n_pcs, n_union)
    labels: np.ndarray,    # (nsp,)
    ich: np.ndarray,       # (n_union,)
    Nchan: int,
    n_pcs: int
) -> np.ndarray:
    """
    Convert dd + labels into KS4-style Wall templates:
      (n_clusters, Nchan, n_pcs)
    """
    templates = []
    for k in range(int(labels.max()) + 1):
        mk = (labels == k)
        if not np.any(mk):
            continue
        centroid = dd[mk].mean(axis=0)          # (n_pcs, n_union)
        tmp = np.zeros((Nchan, n_pcs), dtype=np.float32)
        tmp[ich] = centroid.T.astype(np.float32, copy=False)
        templates.append(tmp)

    if len(templates) == 0:
        return np.zeros((0, Nchan, n_pcs), dtype=np.float32)

    return np.stack(templates, axis=0)          # (n_clusters, Nchan, n_pcs)


def _cluster_one_pass(
    ops: Dict[str, Any],
    st: np.ndarray | torch.Tensor,
    tF: torch.Tensor,
    mode: str,
    optimize_memory: bool = False,
) -> Tuple[np.ndarray, torch.Tensor]:
    print("KS3 clustering")
    """
    mode = 'spikes'   -> KS3 template_learning-like clustering (on universal spikes)
    mode = 'template' -> KS3 final_clustering-like clustering (on learned-template spikes)
    """
    if mode not in ("spikes", "template"):
        raise ValueError("mode must be 'spikes' or 'template'")
    if optimize_memory:
        return _cluster_one_pass_optimized(ops, st, tF, mode)

    s = ops["settings"]
    fs = float(s["fs"])
    n_pcs = int(s["n_pcs"])
    Nchan = int(ops.get("Nchan", len(ops["chanMap"])))

    tF_np = _as_numpy(tF).astype(np.float32, copy=False)  # (n_spikes, nC, n_pcs)

    # Which "template id" column do we cluster around?
    if mode == "spikes":
        # spikedetect.run produces st with 6 columns, and template index is in col 5
        pid = st[:, 5].astype(np.int32, copy=False)
        times_s = st[:, 0].astype(np.float64, copy=False)  # already seconds
        iC = _as_numpy(ops["iC"]).astype(np.int64, copy=False)
        # Template centers:
        xcup = _as_numpy(ops.get("xcup", None))
        ycup = _as_numpy(ops.get("ycup", None))
        if xcup is None or ycup is None:
            # fallback: construct from xup/yup if needed
            xup = _as_numpy(ops.get("xup", ops["xc"]))
            yup = _as_numpy(ops.get("yup", ops["yc"]))
            X, Y = np.meshgrid(xup, yup)
            xcup = X.reshape(-1)
            ycup = Y.reshape(-1)
    else:
        # learned-template spikes have st with 3 columns, template index is col 1
        pid = st[:, 1].astype(np.int32, copy=False)
        times_s = st[:, 0].astype(np.float64, copy=False) / fs  # samples -> seconds

        # These are created during template extraction/deconvolution in KS4
        if "iU" not in ops or "iCC" not in ops:
            raise ValueError(
                "KS3 final clustering requires ops['iU'] and ops['iCC'], which should "
                "be populated by the template extraction step (template_matching.extract)."
            )
        iU = _as_numpy(ops["iU"]).astype(np.int64, copy=False)      # (n_templates,)
        iCC = _as_numpy(ops["iCC"]).astype(np.int64, copy=False)    # (nC, Nchan)
        iC = iCC[:, iU]                                             # (nC, n_templates)

        xcup = _as_numpy(ops["xc"])[iU]
        ycup = _as_numpy(ops["yc"])[iU]

    # spacing
    dmin = float(ops.get("dmin", s.get("dmin", 0)) or 0)
    if dmin <= 0:
        # guess from probe y spacing
        yuniq = np.unique(_as_numpy(ops["yc"]))
        dmin = float(np.median(np.diff(np.sort(yuniq))))
    dminx = float(ops.get("dminx", s.get("dminx", 32.0)))

    # spatial centers
    xcent, ycent = _centers_grid(xcup, ycup, dmin=dmin, dminx=dminx)

    # output
    clu = -np.ones(pid.shape[0], dtype=np.int32)
    Wall_list = []

    params = _get_pursuit_params_from_ops(ops)

    next_unit = 0

    # Partition templates into disjoint spatial bins by (x0,y0)
    for y0 in ycent:
        for x0 in xcent:
            itemp = np.flatnonzero((np.abs(ycup - y0) < dmin) & (np.abs(xcup - x0) < dminx))
            if itemp.size == 0:
                continue

            in_group = np.isin(pid, itemp)
            if not np.any(in_group):
                continue

            idx = np.flatnonzero(in_group)
            pid_g = pid[idx]
            tF_g = tF_np[idx]  # (nsp, nC, n_pcs)
            t_g = times_s[idx]

            dd, ich = _assemble_dd(tF_g, pid_g, itemp, iC, n_pcs=n_pcs)
            X = dd.reshape(dd.shape[0], -1)

            # pursuit clustering
            lab = run_pursuit(X, params=params, times_s=(t_g if params.use_ccg else None))

            # turn group-local labels into global unit ids
            for lk in range(int(lab.max()) + 1):
                mk = (lab == lk)
                if not np.any(mk):
                    continue
                clu[idx[mk]] = next_unit
                next_unit += 1

            # templates for this group (in full channel space)
            Wg = _dd_to_templates_full(dd, lab, ich, Nchan=Nchan, n_pcs=n_pcs)
            if Wg.shape[0] > 0:
                Wall_list.append(Wg)

    # Any spikes not assigned (should be rare): assign them to their own unit.
    un = np.flatnonzero(clu < 0)
    if un.size > 0:
        for i in un:
            clu[i] = next_unit
            next_unit += 1
            # Make a minimal template from that spike's local channels
            tmp = np.zeros((Nchan, n_pcs), dtype=np.float32)
            # If we can infer channels for this spike's template id, fill them.
            if pid[i] >= 0 and pid[i] < iC.shape[1]:
                chans = iC[:, pid[i]].astype(np.int64, copy=False)
                chans = chans[chans >= 0]
                tmp[chans] = tF_np[i, :len(chans), :]
            Wall_list.append(tmp[None, :, :])

    Wall = np.concatenate(Wall_list, axis=0) if len(Wall_list) else np.zeros((0, Nchan, n_pcs), dtype=np.float32)
    Wall_torch = torch.from_numpy(Wall).float()

    # final sanity: relabel to 0..K-1
    uniq = np.unique(clu)
    remap = {old: new for new, old in enumerate(uniq.tolist())}
    clu = np.array([remap[int(x)] for x in clu], dtype=np.int32)

    return clu, Wall_torch


def ks3_first_clustering(
    ops: Dict[str, Any],
    st0: np.ndarray | torch.Tensor,
    tF: torch.Tensor,
    device: Optional[torch.device] = None,
    progress_bar=None,
    optimize_memory: bool = False,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Drop-in replacement for:
      clu, Wall = clustering_qr.run(..., mode='spikes')

    Returns:
      clu0 (n_spikes,), Wall0 (n_clusters, Nchan, n_pcs)
    """
    clu0, Wall0 = _cluster_one_pass(ops, st0, tF, mode="spikes", optimize_memory=optimize_memory)
    return clu0, Wall0


def ks3_final_clustering(
    ops: Dict[str, Any],
    st: np.ndarray | torch.Tensor,
    tF: torch.Tensor,
    device: Optional[torch.device] = None,
    progress_bar=None,
    optimize_memory: bool = False,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Drop-in replacement for:
      clu, Wall = clustering_qr.run(..., mode='template')

    Returns:
      clu (n_spikes,), Wall (n_clusters, Nchan, n_pcs)
    """
    clu, Wall = _cluster_one_pass(ops, st, tF, mode="template", optimize_memory=optimize_memory)
    return clu, Wall
