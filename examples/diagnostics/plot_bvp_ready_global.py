from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _require_vars(ds: xr.Dataset, required: Iterable[str], label: str) -> None:
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"{label} missing required variables: {missing}")


def _pick_acc_vars(ds_cont: xr.Dataset) -> Tuple[str, str]:
    if ("acc_lin_ned_n" in ds_cont) and ("acc_lin_ned_e" in ds_cont):
        return "acc_lin_ned_n", "acc_lin_ned_e"
    if ("acc_ned_n" in ds_cont) and ("acc_ned_e" in ds_cont):
        return "acc_ned_n", "acc_ned_e"
    raise KeyError("Could not find acceleration components (need acc_lin_ned_n/e or acc_ned_n/e in ds_cont).")


def _parking_segments(ds_segments: xr.Dataset, cycle_number: int) -> List[Tuple[int, int]]:
    m = (np.asarray(ds_segments["cycle_number"].values).astype(int) == int(cycle_number)) & (
        np.asarray(ds_segments["is_parking_phase"].values).astype(bool)
    )
    if not np.any(m):
        return []
    idx0 = np.asarray(ds_segments["idx0"].values)[m].astype(int)
    idx1 = np.asarray(ds_segments["idx1"].values)[m].astype(int)
    order = np.argsort(idx0)
    return [(int(idx0[i]), int(idx1[i])) for i in order]


def _plot_map(ds_cycles: xr.Dataset, outdir: Path | None) -> None:
    _require_vars(ds_cycles, ["lon_surface_end", "lat_surface_end", "valid_for_bvp"], "ds_cycles")

    lon = np.asarray(ds_cycles["lon_surface_end"].values, dtype=float)
    lat = np.asarray(ds_cycles["lat_surface_end"].values, dtype=float)
    valid = np.asarray(ds_cycles["valid_for_bvp"].values, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))
    if np.isfinite(lon).any() and np.isfinite(lat).any():
        ax.plot(lon, lat, "-", color="0.7", lw=1.5, label="Surface track")
    ax.scatter(lon[~valid], lat[~valid], c="tab:red", s=20, label="invalid")
    ax.scatter(lon[valid], lat[valid], c="tab:green", s=20, label="valid")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    title = "Surface fixes"
    platform = ds_cycles.attrs.get("platform", "")
    if platform:
        title += f" — {platform}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.4)

    if outdir is None:
        plt.show()
    else:
        out = outdir / "map_surface_fixes.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close(fig)


def _plot_accel_for_cycle(
    cyc: int,
    segs: List[Tuple[int, int]],
    ds_cont: xr.Dataset,
    acc_n_var: str,
    acc_e_var: str,
    outdir: Path | None,
) -> None:
    time_all = np.asarray(ds_cont["time"].values).astype("datetime64[ns]")
    an = np.asarray(ds_cont[acc_n_var].values, dtype=float)
    ae = np.asarray(ds_cont[acc_e_var].values, dtype=float)

    t_list: List[np.ndarray] = []
    an_list: List[np.ndarray] = []
    ae_list: List[np.ndarray] = []
    for a, b in segs:
        if b <= a:
            continue
        t_list.append(time_all[a:b])
        an_list.append(an[a:b])
        ae_list.append(ae[a:b])

    if not t_list:
        print(f"[skip] cycle {cyc}: parking slices are empty")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for t, vn, ve in zip(t_list, an_list, ae_list):
        ax.plot(t, vn, label="acc_n")
        ax.plot(t, ve, label="acc_e", alpha=0.7)
    ax.set_title(f"Cycle {cyc} — parking acceleration ({acc_n_var}/{acc_e_var})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    if outdir is None:
        plt.show()
    else:
        out = outdir / f"accel_cycle_{cyc}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close(fig)


def _plot_pressure_for_cycle(
    cyc: int,
    segs: List[Tuple[int, int]],
    ds_cont: xr.Dataset,
    outdir: Path | None,
) -> None:
    time_all = np.asarray(ds_cont["time"].values).astype("datetime64[ns]")
    pres = np.asarray(ds_cont["pres"].values, dtype=float)

    t_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []
    for a, b in segs:
        if b <= a:
            continue
        t_list.append(time_all[a:b])
        p_list.append(pres[a:b])

    if not t_list:
        print(f"[skip] cycle {cyc}: parking slices are empty")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for t, p in zip(t_list, p_list):
        ax.plot(t, p, color="tab:blue")
    ax.set_title(f"Cycle {cyc} — parking pressure")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (dbar)")
    ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4)

    if outdir is None:
        plt.show()
    else:
        out = outdir / f"pressure_cycle_{cyc}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Global diagnostics for BVP-ready outputs (parking phase).")
    p.add_argument("--cycles", required=True, help="Path to *_cycles.nc")
    p.add_argument("--segments", required=True, help="Path to *_segments.nc")
    p.add_argument("--cont", required=True, help="Path to *_preprocessed_imu.nc")
    p.add_argument("--bvp", required=True, help="Path to *_bvp_ready.nc (used to sanity-check availability)")
    p.add_argument("--outdir", default=None, help="If provided, save PNGs here; otherwise show plots interactively.")
    return p


def main():
    args = _build_parser().parse_args()
    outdir = Path(args.outdir) if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    ds_cycles = xr.open_dataset(args.cycles)
    ds_segments = xr.open_dataset(args.segments)
    ds_cont = xr.open_dataset(args.cont)
    ds_bvp = xr.open_dataset(args.bvp)

    # Basic presence checks
    _require_vars(ds_cycles, ["cycle_number", "valid_for_bvp", "lon_surface_end", "lat_surface_end"], "ds_cycles")
    _require_vars(ds_segments, ["cycle_number", "idx0", "idx1", "is_parking_phase"], "ds_segments")
    _require_vars(ds_cont, ["time", "pres"], "ds_cont")
    _require_vars(ds_bvp, ["cycle_number", "row_start", "row_size"], "ds_bvp")

    _plot_map(ds_cycles, outdir)

    # Pick up to first 6 valid cycles
    valid_mask = np.asarray(ds_cycles["valid_for_bvp"].values, dtype=bool)
    cyc_numbers = np.asarray(ds_cycles["cycle_number"].values).astype(int)
    valid_cycles = cyc_numbers[valid_mask]
    valid_cycles = np.sort(valid_cycles)
    if valid_cycles.size == 0:
        print("No valid cycles found (valid_for_bvp=False everywhere). Nothing to plot.")
        return
    valid_cycles = valid_cycles[:6]

    acc_n_var, acc_e_var = _pick_acc_vars(ds_cont)

    for cyc in valid_cycles:
        segs = _parking_segments(ds_segments, cyc)
        if not segs:
            print(f"[skip] cycle {cyc}: no parking segment found")
            continue
        _plot_accel_for_cycle(cyc, segs, ds_cont, acc_n_var, acc_e_var, outdir)
        _plot_pressure_for_cycle(cyc, segs, ds_cont, outdir)

    ds_cycles.close()
    ds_segments.close()
    ds_cont.close()
    ds_bvp.close()


if __name__ == "__main__":
    main()
