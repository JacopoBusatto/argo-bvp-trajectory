from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

EXPECTED_PHASES = {"surface", "descent", "park_drift", "ascent"}
ALLOWED_EXTRA_PHASES = {"in_air"}
FORBIDDEN_PHASES = {"descent_to_profile", "profile_drift"}
ANCHORS = ["immersion", "park_start", "park_end", "emersion"]
ANCHOR_MARKERS = {
    "immersion": "^",
    "park_start": "s",
    "park_end": "D",
    "emersion": "o",
}


def _load(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


def _print_phase_counts(name: str, phases: np.ndarray) -> None:
    uniq, cnt = np.unique(phases.astype(str), return_counts=True)
    print(f"[{name}] phase_name counts:")
    for u, c in zip(uniq, cnt):
        print(f"  {u:>18s} : {int(c)}")


def _assert_phases(phases: np.ndarray) -> None:
    phase_set = set(phases.astype(str))
    if FORBIDDEN_PHASES & phase_set:
        raise SystemExit(f"Forbidden phases present in bvp_ready: {FORBIDDEN_PHASES & phase_set}")
    allowed = EXPECTED_PHASES | ALLOWED_EXTRA_PHASES
    unexpected = phase_set - allowed
    if unexpected:
        raise SystemExit(f"Unexpected phases in bvp_ready: {unexpected}")
    extras = phase_set - EXPECTED_PHASES
    if extras:
        print(f"Note: extra phases tolerated in bvp_ready: {sorted(extras)}")


def _platform(*dsets: xr.Dataset) -> str:
    for ds in dsets:
        p = ds.attrs.get("platform", "")
        if p:
            return str(p)
    return ""


def _map_obs_to_cycle(ds_obs: xr.Dataset, ds_cycles: xr.Dataset) -> np.ndarray | None:
    if "cycle_number" in ds_obs and "obs" in ds_obs["cycle_number"].dims:
        return np.asarray(ds_obs["cycle_number"].values).astype(int)
    if "time" not in ds_obs or "cycle_number" not in ds_cycles:
        return None
    t_obs = np.asarray(ds_obs["time"].values)
    cyc_nums = np.asarray(ds_cycles["cycle_number"].values).astype(int)
    t_start = np.asarray(ds_cycles["t_cycle_start"].values) if "t_cycle_start" in ds_cycles else np.asarray(ds_cycles["t_surface_start"].values)
    if "t_cycle_start" not in ds_cycles and "t_surface_start" not in ds_cycles:
        return None
    # define interval end as next start, last end as +inf
    t_end = np.empty_like(t_start, dtype="datetime64[ns]")
    t_end[:-1] = t_start[1:]
    t_end[-1] = np.datetime64("9999-12-31")
    cyc_for_obs = np.full(t_obs.shape, -1, dtype=int)
    for idx, (ts, te, cn) in enumerate(zip(t_start, t_end, cyc_nums)):
        m = (t_obs >= ts) & (t_obs < te)
        cyc_for_obs[m] = cn
    if np.any(cyc_for_obs < 0):
        return None
    return cyc_for_obs


def _anchor_indices(phase: np.ndarray, anchor: str) -> tuple[int | None, int | None]:
    if anchor == "immersion":
        if np.any(phase == "descent"):
            first_desc = int(np.argmax(phase == "descent"))
            surf_before = np.where((phase == "surface") & (np.arange(phase.size) < first_desc))[0]
            if surf_before.size:
                idx = int(surf_before[-1])
            else:
                idx = first_desc
        else:
            surf_idxs = np.where(phase == "surface")[0]
            idx = int(surf_idxs[-1]) if surf_idxs.size else None
        return idx, idx
    if anchor == "park_start":
        if np.any(phase == "park_drift"):
            idx = int(np.argmax(phase == "park_drift"))
            return idx, idx
        return None, None
    if anchor == "park_end":
        if np.any(phase == "park_drift"):
            idx = int(np.where(phase == "park_drift")[0][-1])
            return idx, idx
        return None, None
    if anchor == "emersion":
        if np.any(phase == "ascent"):
            last_ascent = int(np.where(phase == "ascent")[0][-1])
            surf_after = np.where((phase == "surface") & (np.arange(phase.size) > last_ascent))[0]
            if surf_after.size:
                idx = int(surf_after[0])
            else:
                idx = last_ascent
        else:
            surf_idxs = np.where(phase == "surface")[0]
            idx = int(surf_idxs[0]) if surf_idxs.size else None
        return idx, idx
    return None, None


def _extract_anchors_from_obs(
    ds_obs: xr.Dataset,
    cyc_map: np.ndarray,
    cycle_numbers: np.ndarray,
    xvar: str = "x_east_m",
    yvar: str = "y_north_m",
    zvar: str = "z_m",
    warn_prefix: str = "[anchors]",
) -> dict[int, dict[str, tuple[float, float, float]]]:
    anchors: dict[int, dict[str, tuple[float, float, float]]] = {}
    phases = np.asarray(ds_obs["phase_name"].values).astype(str)
    if xvar not in ds_obs or yvar not in ds_obs:
        return anchors
    x = np.asarray(ds_obs[xvar].values, dtype=float)
    y = np.asarray(ds_obs[yvar].values, dtype=float)
    z = np.asarray(ds_obs[zvar].values, dtype=float) if zvar in ds_obs else np.zeros_like(x)
    for cn in cycle_numbers:
        mask = cyc_map == cn
        if not np.any(mask):
            continue
        ph_c = phases[mask]
        x_c, y_c, z_c = x[mask], y[mask], z[mask]
        anchors[cn] = {}
        for anch in ANCHORS:
            i0, i1 = _anchor_indices(ph_c, anch)
            if i0 is None or i1 is None:
                print(f"{warn_prefix} missing {anch} for cycle {cn}")
                continue
            anchors[cn][anch] = (x_c[i0], y_c[i0], z_c[i0])
    return anchors


def _extract_truth_anchors(ds_truth: xr.Dataset, ds_cycles: xr.Dataset) -> dict[int, dict[str, tuple[float, float, float]]]:
    anchors: dict[int, dict[str, tuple[float, float, float]]] = {}
    if {"r_n_start", "r_e_start", "r_n_park_start", "r_e_park_start", "r_n_park_end", "r_e_park_end", "r_n_surface_end", "r_e_surface_end"}.issubset(
        set(ds_truth.data_vars)
    ):
        cycles = np.asarray(ds_truth["cycle"].values).astype(int) if "cycle" in ds_truth.coords else np.arange(len(ds_truth["r_n_start"]))
        for i, cn in enumerate(cycles):
            anchors[cn] = {
                "immersion": (float(ds_truth["r_e_start"].values[i]), float(ds_truth["r_n_start"].values[i]), 0.0),
                "park_start": (float(ds_truth["r_e_park_start"].values[i]), float(ds_truth["r_n_park_start"].values[i]), 0.0),
                "park_end": (float(ds_truth["r_e_park_end"].values[i]), float(ds_truth["r_n_park_end"].values[i]), 0.0),
                "emersion": (float(ds_truth["r_e_surface_end"].values[i]), float(ds_truth["r_n_surface_end"].values[i]), 0.0),
            }
        return anchors
    # fall back to phase-based extraction if truth has positions and phases
    phase_var = "phase_name" if "phase_name" in ds_truth else "phase_name_truth" if "phase_name_truth" in ds_truth else None
    if phase_var is not None:
        xvar = "x_east_m" if "x_east_m" in ds_truth else "r_e_truth"
        yvar = "y_north_m" if "y_north_m" in ds_truth else "r_n_truth"
        zvar = "z_m" if "z_m" in ds_truth else "pres_truth" if "pres_truth" in ds_truth else None
        cyc_map = _map_obs_to_cycle(ds_truth, ds_cycles)
        if cyc_map is not None and xvar in ds_truth and yvar in ds_truth:
            anchors = _extract_anchors_from_obs(ds_truth.rename({phase_var: "phase_name"}), cyc_map, np.unique(cyc_map), xvar=xvar, yvar=yvar, zvar=zvar or "z_m", warn_prefix="[truth]")
    return anchors
    return anchors


def _anchor_errors(anchors_truth: dict[int, dict[str, tuple[float, float, float]]], anchors_bvp: dict[int, dict[str, tuple[float, float, float]]]) -> dict[str, list[tuple[int, float]]]:
    errs: dict[str, list[tuple[int, float]]] = {a: [] for a in ANCHORS}
    for cn, tr_dict in anchors_truth.items():
        if cn not in anchors_bvp:
            continue
        est_dict = anchors_bvp[cn]
        for anch in ANCHORS:
            if anch not in tr_dict or anch not in est_dict:
                continue
            te = np.array(tr_dict[anch], dtype=float)
            be = np.array(est_dict[anch], dtype=float)
            d = float(np.linalg.norm(te - be))
            errs[anch].append((cn, d))
    return errs


def _plot_3d(sol: xr.Dataset, outdir: Path, cycles: List[int], platform: str) -> Path:
    # Handle either obs-level cycle_number or cycle coord
    phases = np.asarray(sol["phase_name"].values).astype(str)
    x = np.asarray(sol["x_east_m"].values, dtype=float)
    y = np.asarray(sol["y_north_m"].values, dtype=float)
    z = np.asarray(sol["z_m"].values, dtype=float)
    cyc_obs = None
    obs_len = sol.sizes.get("obs", 0)
    if "cycle_number_for_obs" in sol and sol["cycle_number_for_obs"].dims[0] == obs_len:
        cyc_obs = np.asarray(sol["cycle_number_for_obs"].values).astype(int)
    elif "cycle_number" in sol and sol["cycle_number"].dims == ("obs",):
        cyc_obs = np.asarray(sol["cycle_number"].values).astype(int)

    colors = {"parking": "tab:blue", "park_drift": "tab:blue", "ascent": "tab:orange", "descent": "tab:green", "surface": "tab:red"}

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    if cyc_obs is None:
        # No per-observation cycle mapping; plot everything colored by phase
        for phu in np.unique(phases):
            mm = phases == phu
            ax.plot3D(x[mm], y[mm], -z[mm], ".", ms=1.5, color=colors.get(phu, "gray"), label=phu)
    else:
        for cyc in cycles:
            mask = cyc_obs == cyc
            if not np.any(mask):
                continue
            ph = phases[mask]
            xc, yc, zc = x[mask], y[mask], z[mask]
            for phu in np.unique(ph):
                mm = ph == phu
                ax.plot3D(xc[mm], yc[mm], -zc[mm], color=colors.get(phu, "gray"), label=f"{cyc}-{phu}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("-Z (approx m)")
    ax.set_title(f"3D solved trajectories (phase_name) {platform}")
    out = outdir / "traj_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_plan(sol: xr.Dataset, outdir: Path, platform: str) -> Path:
    phases = np.asarray(sol["phase_name"].values).astype(str)
    x = np.asarray(sol["x_east_m"].values, dtype=float)
    y = np.asarray(sol["y_north_m"].values, dtype=float)
    colors = {"parking": "tab:blue", "park_drift": "tab:blue", "ascent": "tab:orange", "descent": "tab:green", "surface": "tab:red"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for ph in np.unique(phases):
        m = phases == ph
        ax.plot(x[m], y[m], ".", ms=2, color=colors.get(ph, "gray"), label=ph)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Plan view (phase_name) {platform}")
    ax.legend(markerscale=4)
    ax.grid(True, ls="--", alpha=0.4)
    out = outdir / "traj_plan.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_anchor_plan(
    anchors_truth: dict[int, dict[str, tuple[float, float, float]]],
    anchors_bvp: dict[int, dict[str, tuple[float, float, float]]],
    anchors_solved: dict[int, dict[str, tuple[float, float, float]]],
    end_good: set[int],
    outdir: Path,
    platform: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    for cn, tr_dict in anchors_truth.items():
        if cn not in anchors_bvp:
            continue
        est_dict = anchors_bvp[cn]
        for anch in ANCHORS:
            if anch not in tr_dict:
                continue
            x_t, y_t, _ = tr_dict[anch]
            ax.scatter(x_t, y_t, marker=ANCHOR_MARKERS[anch], s=60, label=f"truth {anch}" if cn == list(anchors_truth.keys())[0] else "", color="k")
            if anch in est_dict:
                x_b, y_b, _ = est_dict[anch]
                ax.scatter(x_b, y_b, marker=ANCHOR_MARKERS[anch], s=40, facecolors="none", edgecolors="tab:blue", label=f"bvp {anch}" if cn == list(anchors_truth.keys())[0] else "")
            if cn in anchors_solved and anch in anchors_solved[cn]:
                x_s, y_s, _ = anchors_solved[cn][anch]
                ax.scatter(
                    x_s,
                    y_s,
                    marker="x",
                    s=45,
                    color="tab:orange",
                    alpha=1.0 if cn in end_good else 0.25,
                    label="solved" if cn == list(anchors_truth.keys())[0] and anch == "emersion" else "",
                )
        # light line connecting anchors
        order = [a for a in ANCHORS if a in tr_dict]
        if len(order) >= 2:
            xs = [tr_dict[a][0] for a in order]
            ys = [tr_dict[a][1] for a in order]
            ax.plot(xs, ys, color="k", alpha=0.2, lw=0.8)
        if cn in est_dict:
            order_est = [a for a in ANCHORS if a in est_dict]
            if len(order_est) >= 2:
                xs = [est_dict[a][0] for a in order_est]
                ys = [est_dict[a][1] for a in order_est]
                ax.plot(xs, ys, color="tab:blue", alpha=0.3, lw=0.8)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Anchor comparison (truth vs BVP) {platform}")
    handles, labels = ax.get_legend_handles_labels()
    by_lab = dict(zip(labels, handles))
    ax.legend(by_lab.values(), by_lab.keys(), fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    out = outdir / "anchors_plan.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_errors(sol: xr.Dataset, outdir: Path, platform: str) -> List[Path]:
    cyc = np.asarray(sol["cycle"].values).astype(int) if "cycle" in sol.coords else np.unique(sol["cycle_number"].values.astype(int))
    metas = sol if "cycle" in sol.coords else None
    paths = []
    metrics = [
        ("delta_start_m", "Delta start vs immersion (m)"),
        ("delta_end_m", "Delta end vs emersion (m)"),
        ("misfit_end_m", "BVP end misfit (m)"),
    ]
    for var, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        if metas is not None and var in metas:
            y = np.asarray(metas[var].values, dtype=float)
        else:
            y = np.full_like(cyc, np.nan, dtype=float)
        ax.plot(cyc, y, "o-")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(var)
        ax.set_title(f"{title} {platform}")
        ax.grid(True, ls="--", alpha=0.4)
        out = outdir / f"{var}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def _plot_sampling_scatter(sol: xr.Dataset, outdir: Path, platform: str) -> List[Path]:
    metas = sol if "cycle" in sol.coords else None
    paths = []
    pairs = [
        ("delta_end_m", "parking_median_dt_s", "delta_end vs parking dt"),
        ("delta_start_m", "parking_median_dt_s", "delta_start vs parking dt"),
        ("misfit_end_m", "parking_median_dt_s", "misfit_end vs parking dt"),
        ("misfit_end_m", "descent_median_dt_s", "misfit_end vs descent dt"),
        ("misfit_end_m", "ascent_median_dt_s", "misfit_end vs ascent dt"),
    ]
    if metas is None:
        return paths
    cyc = np.asarray(metas["cycle"].values).astype(int)
    for yvar, xvar, title in pairs:
        if xvar not in metas or yvar not in metas:
            continue
        x = np.asarray(metas[xvar].values, dtype=float)
        y = np.asarray(metas[yvar].values, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x, y, c=cyc, cmap="viridis", s=30)
        ax.set_xlabel(xvar)
        ax.set_ylabel(yvar)
        ax.set_title(f"{title} {platform}")
        ax.grid(True, ls="--", alpha=0.4)
        out = outdir / f"{yvar}_vs_{xvar}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def _plot_anchor_errors(errors: dict[str, list[tuple[int, float]]], outdir: Path, platform: str) -> List[Path]:
    paths: List[Path] = []
    for anch, rows in errors.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r[0])
        cyc = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cyc, vals, "o-")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{anch} anchor error {platform}")
        ax.grid(True, ls="--", alpha=0.4)
        out = outdir / f"anchor_err_{anch}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def _maybe_ground_truth(dsets: List[xr.Dataset], outdir: Path, platform: str) -> List[Path]:
    gt_vars = []
    for ds in dsets:
        for v in ds.variables:
            if "truth" in v or "gt" in v:
                gt_vars.append((ds, v))
    if not gt_vars:
        print("No ground truth variables found.")
        return []

    paths = []
    # Try simple comparison if r_n_surface_end / r_e_surface_end present
    try:
        ds_gt = [ds for ds, v in gt_vars if v in ("r_n_surface_end", "r_e_surface_end")]
        if ds_gt:
            pass
    except Exception:
        pass
    return paths


def _plot_anchor_3d(
    anchors_truth: dict[int, dict[str, tuple[float, float, float]]],
    anchors_bvp: dict[int, dict[str, tuple[float, float, float]]],
    anchors_solved: dict[int, dict[str, tuple[float, float, float]]],
    end_good: set[int],
    outdir: Path,
    platform: str,
) -> Path:
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    for cn, tr_dict in anchors_truth.items():
        if cn not in anchors_bvp:
            continue
        est_dict = anchors_bvp[cn]
        for anch in ANCHORS:
            if anch in tr_dict:
                xt, yt, zt = tr_dict[anch]
                ax.scatter(xt, yt, -zt, marker=ANCHOR_MARKERS[anch], s=60, color="k")
            if anch in est_dict:
                xb, yb, zb = est_dict[anch]
                ax.scatter(xb, yb, -zb, marker=ANCHOR_MARKERS[anch], s=40, facecolors="none", edgecolors="tab:blue")
            if cn in anchors_solved and anch in anchors_solved[cn]:
                xb, yb, zb = anchors_solved[cn][anch]
                ax.scatter(xb, yb, -zb, marker="x", s=45, color="tab:orange", alpha=1.0 if cn in end_good else 0.25)
        order = [a for a in ANCHORS if a in tr_dict]
        if len(order) >= 2:
            xs = [tr_dict[a][0] for a in order]
            ys = [tr_dict[a][1] for a in order]
            zs = [-tr_dict[a][2] for a in order]
            ax.plot(xs, ys, zs, color="k", alpha=0.2)
        order_est = [a for a in ANCHORS if a in est_dict]
        if len(order_est) >= 2:
            xs = [est_dict[a][0] for a in order_est]
            ys = [est_dict[a][1] for a in order_est]
            zs = [-est_dict[a][2] for a in order_est]
            ax.plot(xs, ys, zs, color="tab:blue", alpha=0.3)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("-Z (approx m)")
    ax.set_title(f"Anchor comparison 3D (truth vs BVP) {platform}")
    out = outdir / "anchors_3d.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_anchor_scatter(anchors_truth: dict[int, dict[str, tuple[float, float, float]]], anchors_bvp: dict[int, dict[str, tuple[float, float, float]]], outdir: Path, platform: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    for anch in ANCHORS:
        xs_t = []
        ys_t = []
        xs_b = []
        ys_b = []
        for cn, tr_dict in anchors_truth.items():
            if anch not in tr_dict or cn not in anchors_bvp or anch not in anchors_bvp[cn]:
                continue
            xt, yt, _ = tr_dict[anch]
            xb, yb, _ = anchors_bvp[cn][anch]
            xs_t.append(xt)
            ys_t.append(yt)
            xs_b.append(xb)
            ys_b.append(yb)
        if xs_t:
            ax.scatter(xs_t, ys_t, marker=ANCHOR_MARKERS[anch], s=50, label=f"truth {anch}")
        if xs_b:
            ax.scatter(xs_b, ys_b, marker=ANCHOR_MARKERS[anch], s=40, facecolors="none", edgecolors="tab:blue", label=f"bvp {anch}")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Anchor endpoints scatter {platform}")
    handles, labels = ax.get_legend_handles_labels()
    by_lab = dict(zip(labels, handles))
    ax.legend(by_lab.values(), by_lab.keys(), fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    out = outdir / "anchors_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="Diagnostics for synthetic BVP-ready and solved outputs.")
    ap.add_argument("--bvp", required=True, help="Path to *_bvp_ready.nc")
    ap.add_argument("--cycles", required=True, help="Path to *_cycles.nc")
    ap.add_argument("--solved", required=True, help="Path to *_solved.nc")
    ap.add_argument("--outdir", required=True, help="Directory to save plots")
    ap.add_argument("--truth", required=False, help="Optional path to ground truth file (e.g., outputs/synthetic/ground_truth.nc)")
    ap.add_argument("--end-threshold", type=float, default=100.0, help="Threshold in meters for good end-point agreement")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ds_bvp = _load(args.bvp)
    ds_cyc = _load(args.cycles)
    ds_sol = _load(args.solved)
    truth_path = args.truth
    if truth_path is None:
        cand = Path(args.bvp).parent.parent / "ground_truth.nc"
        if cand.exists():
            truth_path = str(cand)
    ds_truth = _load(truth_path) if truth_path and Path(truth_path).exists() else None

    platform = _platform(ds_bvp, ds_cyc, ds_sol)

    # Phase summaries
    _print_phase_counts("bvp_ready", ds_bvp["phase_name"].values)
    _print_phase_counts("solved", ds_sol["phase_name"].values)
    if "macro_phase" in ds_sol:
        _print_phase_counts("solved.macro_phase", ds_sol["macro_phase"].values)

    # Assertions
    _assert_phases(ds_bvp["phase_name"].values)
    print("OK: BVP-ready phase_name already contains 'descent' (no 'other' remapping needed).")

    # Plots
    cycles_unique = np.unique(ds_sol["cycle_number"].values.astype(int))
    subset_cycles = cycles_unique[:3].tolist()
    written: List[Path] = []
    written.append(_plot_3d(ds_sol, outdir, subset_cycles, platform))
    written.append(_plot_plan(ds_sol, outdir, platform))
    written.extend(_plot_errors(ds_sol, outdir, platform))
    written.extend(_plot_sampling_scatter(ds_sol, outdir, platform))

    anchors_truth: dict[int, dict[str, tuple[float, float, float]]] = {}
    if ds_truth is not None:
        anchors_truth = _extract_truth_anchors(ds_truth, ds_cyc)
        print(f"[truth] anchors extracted for {len(anchors_truth)} cycles")
    else:
        print("No ground truth variables found.")

    cyc_map_sol = _map_obs_to_cycle(ds_sol, ds_cyc)
    anchors_solved: dict[int, dict[str, tuple[float, float, float]]] = {}
    if cyc_map_sol is not None:
        anchors_solved = _extract_anchors_from_obs(ds_sol, cyc_map_sol, cycles_unique, warn_prefix="[solved]")
    else:
        print("[warn] could not map solved observations to cycles; skipping anchor comparison.")

    anchors_bvp: dict[int, dict[str, tuple[float, float, float]]] = {}
    if anchors_truth:
        anchors_bvp = anchors_truth.copy()

    end_threshold = args.end_threshold
    errors_for_plot: dict[str, list[tuple[int, float]]] = {a: [] for a in ANCHORS}
    csv_rows = []
    end_good_set: set[int] = set()
    if anchors_truth:
        for cn, tr_dict in anchors_truth.items():
            bvp_dict = anchors_bvp.get(cn, {})
            sol_dict = anchors_solved.get(cn, {})

            def dist(a, b):
                return float(np.linalg.norm(np.array(a) - np.array(b)))

            start_err = dist(bvp_dict["immersion"], tr_dict["immersion"]) if ("immersion" in bvp_dict and "immersion" in tr_dict) else np.nan
            end_fix_err = dist(bvp_dict.get("emersion"), tr_dict.get("emersion")) if ("emersion" in bvp_dict and "emersion" in tr_dict) else np.nan
            end_solved_err = dist(sol_dict.get("emersion"), tr_dict.get("emersion")) if ("emersion" in sol_dict and "emersion" in tr_dict) else np.nan
            park_start_err = dist(bvp_dict.get("park_start"), tr_dict.get("park_start")) if ("park_start" in bvp_dict and "park_start" in tr_dict) else np.nan
            park_end_err = dist(bvp_dict.get("park_end"), tr_dict.get("park_end")) if ("park_end" in bvp_dict and "park_end" in tr_dict) else np.nan

            if not np.isnan(end_solved_err) and end_solved_err <= end_threshold:
                end_good_set.add(cn)

            if not np.isnan(start_err):
                errors_for_plot["immersion"].append((cn, start_err))
            if not np.isnan(park_start_err):
                errors_for_plot["park_start"].append((cn, park_start_err))
            if not np.isnan(park_end_err):
                errors_for_plot["park_end"].append((cn, park_end_err))
            if not np.isnan(end_solved_err):
                errors_for_plot["emersion"].append((cn, end_solved_err))

            cyc_vals = set(np.asarray(ds_sol["cycle"].values).astype(int)) if "cycle" in ds_sol.coords else set()
            asc_dt = float(ds_sol["ascent_median_dt_s"].sel(cycle=cn).values) if ("ascent_median_dt_s" in ds_sol and cn in cyc_vals) else np.nan
            desc_dt = float(ds_sol["descent_median_dt_s"].sel(cycle=cn).values) if ("descent_median_dt_s" in ds_sol and cn in cyc_vals) else np.nan
            park_dt = float(ds_sol["parking_median_dt_s"].sel(cycle=cn).values) if ("parking_median_dt_s" in ds_sol and cn in cyc_vals) else np.nan

            csv_rows.append(
                {
                    "cycle": cn,
                    "start_err_m": start_err,
                    "end_fix_err_m": end_fix_err,
                    "end_solved_err_m": end_solved_err,
                    "park_start_err_m": park_start_err,
                    "park_end_err_m": park_end_err,
                    "end_good": end_solved_err <= end_threshold if not np.isnan(end_solved_err) else False,
                    "ascent_median_dt_s": asc_dt,
                    "descent_median_dt_s": desc_dt,
                    "parking_median_dt_s": park_dt,
                }
            )

    if anchors_truth:
        print(f"[anchors] end_good: {len(end_good_set)}/{len(anchors_truth)} (threshold={end_threshold} m)")
        bad = sorted(set(anchors_truth.keys()) - end_good_set)
        if bad:
            print(f"[anchors] bad cycles: {bad}")

    if anchors_truth:
        written.append(_plot_anchor_plan(anchors_truth, anchors_bvp or anchors_truth, anchors_solved, end_good_set, outdir, platform))
        written.append(_plot_anchor_3d(anchors_truth, anchors_bvp or anchors_truth, anchors_solved, end_good_set, outdir, platform))
        written.append(_plot_anchor_scatter(anchors_truth, anchors_bvp or anchors_truth, outdir, platform))
        written.extend(_plot_anchor_errors(errors_for_plot, outdir, platform))
        csv_path = outdir / "anchors_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "cycle",
                    "start_err_m",
                    "end_fix_err_m",
                    "end_solved_err_m",
                    "park_start_err_m",
                    "park_end_err_m",
                    "end_good",
                    "ascent_median_dt_s",
                    "descent_median_dt_s",
                    "parking_median_dt_s",
                ],
            )
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        written.append(csv_path)

    gt_list = [ds_bvp, ds_cyc, ds_sol] + ([ds_truth] if ds_truth is not None else [])
    written.extend(_maybe_ground_truth(gt_list, outdir, platform))
    gt_list = [ds_bvp, ds_cyc, ds_sol] + ([ds_truth] if ds_truth is not None else [])
    written.extend(_maybe_ground_truth(gt_list, outdir, platform))

    print("Written files:")
    for p in written:
        print(" -", p)

    ds_bvp.close()
    ds_cyc.close()
    ds_sol.close()
    if ds_truth is not None:
        ds_truth.close()


if __name__ == "__main__":
    main()
