from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path

from argobvp.preprocess.config import load_config


def pick_first_existing(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _as_dt64_ns(x) -> np.ndarray:
    """Convert xarray time-like to numpy datetime64[ns] array, keeping NaT."""
    a = np.asarray(x)
    # already datetime64?
    if np.issubdtype(a.dtype, np.datetime64):
        return a.astype("datetime64[ns]")
    # fallback: try to coerce
    return a.astype("datetime64[ns]")


def main():
    cfg = load_config("configs/4903848.yml")

    traj_path = Path(cfg.paths.traj)
    cycles_path = Path(f"outputs/preprocess/{cfg.platform}_cycles.nc")

    # decode_times=True is default, but we keep explicit
    ds_traj = xr.open_dataset(traj_path, decode_times=True)
    ds_cycles = xr.open_dataset(cycles_path)

    # Choose vars
    v_time = pick_first_existing(ds_traj, ["JULD", "JULD_LOCATION", "JULD_GPS", "JULD_FIRST_LOCATION", "JULD_LAST_LOCATION"])
    v_lat  = pick_first_existing(ds_traj, ["LATITUDE", "LAT", "latitude"])
    v_lon  = pick_first_existing(ds_traj, ["LONGITUDE", "LON", "longitude"])

    print("\nChosen variables:")
    print("  time var:", v_time)
    print("  lat var :", v_lat)
    print("  lon var :", v_lon)

    if v_time is None or v_lat is None or v_lon is None:
        raise RuntimeError("Could not find essential time/lat/lon variables in traj file.")

    # Decode time robustly:
    # If xarray decoded it, this is datetime64 already; otherwise this will show numbers.
    t_fix = _as_dt64_ns(ds_traj[v_time].values)
    lat = ds_traj[v_lat].values.astype(float)
    lon = ds_traj[v_lon].values.astype(float)

    # Valid fixes = finite lat/lon AND time not NaT
    valid_time = ~np.isnat(t_fix)
    valid = np.isfinite(lat) & np.isfinite(lon) & valid_time

    tv = t_fix[valid]
    lv = lat[valid]
    lov = lon[valid]

    print("\n--- TRAJ fix sampling ---")
    print("N valid fixes:", tv.size)
    if tv.size > 1:
        dt = np.diff(tv).astype("timedelta64[s]").astype(int)
        print("Median dt between fixes (s):", int(np.median(dt)))
        print("Min dt between fixes (s):   ", int(np.min(dt)))
        print("Max dt between fixes (s):   ", int(np.max(dt)))
        print("Time range:", str(tv[0]), "->", str(tv[-1]))
    elif tv.size == 1:
        print("Only 1 valid fix time:", str(tv[0]))
    else:
        print("No valid fix times found. (JULD likely not decoded or all missing)")

    # ---- Check alignment for a given cycle ----
    cycle_to_check = 1
    row = ds_cycles.sel(cycle=cycle_to_check)

    t_surface_start = row["t_surface_start"].values.astype("datetime64[ns]")
    t_surface_end = row["t_surface_end"].values.astype("datetime64[ns]")

    print("\n--- Cycle check ---")
    print("Cycle:", cycle_to_check)
    print("t_surface_start:", str(t_surface_start))
    print("t_surface_end  :", str(t_surface_end))

    if tv.size == 0:
        print("\nCannot compare: no valid traj fix times.")
    else:
        dsec = np.abs((tv - t_surface_end) / np.timedelta64(1, "s"))
        k = int(np.argmin(dsec))

        print("\nNearest TRAJ fix to t_surface_end:")
        print("  t_fix :", str(tv[k]))
        print("  dt(s) :", float(dsec[k]))
        print("  lat/lon:", float(lv[k]), float(lov[k]))

        inside = (tv >= t_surface_start) & (tv <= t_surface_end)
        print("\nFixes inside surface window:", int(np.count_nonzero(inside)))
        if np.any(inside):
            idx = np.where(inside)[0]
            print("  first:", str(tv[idx[0]]), " last:", str(tv[idx[-1]]))

    ds_traj.close()
    ds_cycles.close()


if __name__ == "__main__":
    main()