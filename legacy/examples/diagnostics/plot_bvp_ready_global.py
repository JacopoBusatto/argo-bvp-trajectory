from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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
        title += f" - {platform}"
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


def _print_parking_summary(ds_cycles: xr.Dataset) -> None:
    _require_vars(ds_cycles, ["parking_n_obs", "valid_for_bvp"], "ds_cycles")
    counts = np.asarray(ds_cycles["parking_n_obs"].values, dtype=float)
    counts = counts[np.isfinite(counts)]
    n_total = counts.size
    n_valid = int(np.sum(np.asarray(ds_cycles["valid_for_bvp"].values, dtype=bool)))
    if n_total == 0:
        print("No cycles found in ds_cycles.")
        return
    q = np.percentile(counts, [0, 10, 50, 90, 100]) if counts.size else [np.nan] * 5
    print(
        f"Cycles summary: total={n_total}, valid_for_bvp={n_valid}, "
        f"parking_n_obs min/p10/med/p90/max = "
        f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}/{q[4]:.0f}"
    )


def _print_phase_counts(ds_cycles: xr.Dataset, max_cycles: int = 6) -> None:
    bases = []
    for v in ds_cycles.variables:
        if v.endswith("_n_obs"):
            bases.append(v[:-6])  # strip _n_obs
    bases = sorted(set(bases))
    if not bases:
        print("No per-phase counts found in ds_cycles.")
        return
    n_cycles = int(ds_cycles.sizes.get("cycle", 0))
    n_show = min(max_cycles, n_cycles)
    print(f"Phase counts (first {n_show} cycles):")
    header = "cycle " + " ".join([f"{b:>18s}" for b in bases])
    print(header)
    for i in range(n_show):
        row = ds_cycles.isel(cycle=i)
        parts = []
        for b in bases:
            n_name = f"{b}_n_obs"
            a_name = f"{b}_attendible"
            n_val = int(row[n_name].values) if n_name in row else -1
            att = bool(row[a_name].values) if a_name in row else False
            parts.append(f"{n_val:6d}{'*' if att else ' '}")
        print(f"{int(row['cycle_number'].values):5d} " + " ".join(parts))


def _plot_accel_for_cycle(cyc: int, ds_bvp: xr.Dataset, outdir: Path | None) -> None:
    if "row_start" not in ds_bvp or "row_size" not in ds_bvp:
        print(f"[skip] cycle {cyc}: row_start/row_size missing in ds_bvp")
        return
    row = ds_bvp.sel(cycle=cyc)
    i0 = int(row["row_start"].values)
    n = int(row["row_size"].values)
    if n <= 0:
        print(f"[skip] cycle {cyc}: no obs in bvp_ready slice")
        return
    obs_idx = np.arange(i0, i0 + n, dtype=int)
    t = np.asarray(ds_bvp["time"].values)[obs_idx]
    an = np.asarray(ds_bvp["acc_n"].values, dtype=float)[obs_idx]
    ae = np.asarray(ds_bvp["acc_e"].values, dtype=float)[obs_idx]
    phases = np.asarray(ds_bvp["phase_name"].values).astype(str)[obs_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    for ph in np.unique(phases):
        m = phases == ph
        ax.plot(t[m], an[m], label=f"{ph} acc_n")
        ax.plot(t[m], ae[m], label=f"{ph} acc_e", alpha=0.7)
    ax.set_title(f"Cycle {cyc} - acceleration (attendible phases)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    if outdir is None:
        plt.show()
    else:
        out = outdir / f"accel_cycle_{cyc}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[saved] {out}")
        plt.close(fig)


def _plot_pressure_for_cycle(cyc: int, ds_bvp: xr.Dataset, outdir: Path | None) -> None:
    if "row_start" not in ds_bvp or "row_size" not in ds_bvp:
        print(f"[skip] cycle {cyc}: row_start/row_size missing in ds_bvp")
        return
    row = ds_bvp.sel(cycle=cyc)
    i0 = int(row["row_start"].values)
    n = int(row["row_size"].values)
    if n <= 0:
        print(f"[skip] cycle {cyc}: no obs in bvp_ready slice")
        return
    obs_idx = np.arange(i0, i0 + n, dtype=int)
    t = np.asarray(ds_bvp["time"].values)[obs_idx]
    p = np.asarray(ds_bvp["z_from_pres"].values, dtype=float)[obs_idx]
    phases = np.asarray(ds_bvp["phase_name"].values).astype(str)[obs_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    for ph in np.unique(phases):
        m = phases == ph
        ax.plot(t[m], p[m], label=ph)
    ax.set_title(f"Cycle {cyc} - pressure (attendible phases)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pressure (dbar)")
    ax.invert_yaxis()
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

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
    p = argparse.ArgumentParser(description="Global diagnostics for BVP-ready outputs (attendible phases).")
    p.add_argument("--cycles", required=True, help="Path to *_cycles.nc")
    p.add_argument("--segments", required=False, help="(unused) Path to *_segments.nc")
    p.add_argument("--cont", required=False, help="(unused) Path to *_preprocessed_imu.nc")
    p.add_argument("--bvp", required=True, help="Path to *_bvp_ready.nc")
    p.add_argument("--outdir", default=None, help="If provided, save PNGs here; otherwise show plots interactively.")
    return p


def main():
    args = _build_parser().parse_args()
    outdir = Path(args.outdir) if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    ds_cycles = xr.open_dataset(args.cycles)
    ds_bvp = xr.open_dataset(args.bvp)

    # Basic presence checks
    _require_vars(
        ds_cycles,
        ["cycle_number", "valid_for_bvp", "lon_surface_end", "lat_surface_end", "parking_n_obs"],
        "ds_cycles",
    )
    _require_vars(ds_bvp, ["cycle_number", "row_start", "row_size", "phase_name", "time", "z_from_pres"], "ds_bvp")

    _print_parking_summary(ds_cycles)
    _plot_map(ds_cycles, outdir)
    _print_phase_counts(ds_cycles)

    # Pick up to first 6 valid cycles
    valid_cycles = np.asarray(ds_bvp["cycle_number"].values).astype(int)
    if valid_cycles.size == 0:
        print("No cycles present in bvp_ready (obs=0). Nothing to plot.")
        ds_cycles.close()
        ds_bvp.close()
        return
    valid_cycles = np.unique(valid_cycles)
    valid_cycles = np.sort(valid_cycles)[:6]

    for cyc in valid_cycles:
        if cyc not in ds_bvp["cycle_number"].values:
            print(f"[skip] cycle {cyc}: not found in bvp_ready")
            continue
        _plot_accel_for_cycle(cyc, ds_bvp, outdir)
        _plot_pressure_for_cycle(cyc, ds_bvp, outdir)

    ds_cycles.close()
    ds_bvp.close()


if __name__ == "__main__":
    main()
