from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick 3D plot of solved trajectories.")
    p.add_argument("--solved", required=True, help="Path to *_solved.nc")
    p.add_argument("--cycles", nargs="*", type=int, default=None, help="Cycle numbers to plot (default: first 3).")
    return p


def main():
    args = _build_parser().parse_args()
    ds = xr.open_dataset(args.solved)

    cycles = np.asarray(ds["cycle_number"].values).astype(int)
    uniq = np.unique(cycles)
    if args.cycles:
        to_plot = [c for c in args.cycles if c in uniq]
    else:
        to_plot = uniq[:3].tolist()

    if not to_plot:
        print("No cycles selected to plot.")
        return

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    phases = np.asarray(ds["macro_phase"].values).astype(str)
    x = np.asarray(ds["x_east_m"].values, dtype=float)
    y = np.asarray(ds["y_north_m"].values, dtype=float)
    z = np.asarray(ds["z_m"].values, dtype=float)
    cyc_obs = np.asarray(ds["cycle_number"].values).astype(int)

    colors = {"parking": "tab:blue", "ascent": "tab:orange", "descent": "tab:green", "surface": "tab:red", "other": "gray"}

    for cyc in to_plot:
        mask = cyc_obs == cyc
        if not np.any(mask):
            continue
        ph = phases[mask]
        xc = x[mask]
        yc = y[mask]
        zc = z[mask]
        for mphase in np.unique(ph):
            mm = ph == mphase
            ax.plot3D(xc[mm], yc[mm], -zc[mm], color=colors.get(mphase, "gray"), label=f"{cyc}-{mphase}")
        # mark parking start/end
        idx_parking = np.where(ph == "parking")[0]
        if idx_parking.size > 0:
            ax.scatter(xc[idx_parking[0]], yc[idx_parking[0]], -zc[idx_parking[0]], color="k", marker="o", s=40)
            ax.scatter(xc[idx_parking[-1]], yc[idx_parking[-1]], -zc[idx_parking[-1]], color="k", marker="x", s=40)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_zlabel("-Z (approx m)")
    ax.set_title("Solved trajectories (attendible phases)")
    plt.tight_layout()
    plt.show()

    ds.close()


if __name__ == "__main__":
    main()
