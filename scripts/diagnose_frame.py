"""Diagnose whether TRUTH accelerations align with ENU x/y coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _second_derivative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    v = np.gradient(x, t, edge_order=2)
    return np.gradient(v, t, edge_order=2)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def diagnose(path: Path) -> int:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        ax = np.asarray(ds["ax"].values, dtype=float)
        ay = np.asarray(ds["ay"].values, dtype=float)
        t = np.asarray(ds["t"].values, dtype=float)

    d2x = _second_derivative(t, x)
    d2y = _second_derivative(t, y)

    print(f"x min/max: {np.nanmin(x):.3f} {np.nanmax(x):.3f}")
    print(f"y min/max: {np.nanmin(y):.3f} {np.nanmax(y):.3f}")
    print(f"ax min/max: {np.nanmin(ax):.6f} {np.nanmax(ax):.6f}")
    print(f"ay min/max: {np.nanmin(ay):.6f} {np.nanmax(ay):.6f}")
    print(f"corr(ax, d2x): {_corr(ax, d2x):.3f}")
    print(f"corr(ay, d2y): {_corr(ay, d2y):.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("truth_path", type=Path, help="Path to TRUTH netCDF file")
    args = parser.parse_args()
    return diagnose(args.truth_path)


if __name__ == "__main__":
    raise SystemExit(main())
