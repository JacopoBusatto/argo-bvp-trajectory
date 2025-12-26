from __future__ import annotations

import numpy as np
import xarray as xr

from argobvp.preprocess.config import load_config


def _fmt_dt64(t) -> str:
    t = np.asarray(t).astype("datetime64[ns]")
    if np.isnat(t):
        return "NaT"
    return str(t).replace("T", " ")


def main():
    cfg = load_config("configs/4903848.yml")
    cycles_nc = f"outputs/preprocess/{cfg.platform}_cycles.nc"
    ds = xr.open_dataset(cycles_nc)

    required = [
        "t_surface_end",
        "lat_surface_end",
        "lon_surface_end",
        "pos_source",
        "t_pos_used",
        "pos_age_s",
        "dt_before_s",
        "dt_after_s",
        "gap_s",
        "dt_nearest_s",
        "t_fix_nearest",
    ]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise RuntimeError(f"Missing variables in cycles.nc: {missing}")

    methods = ds["pos_source"].values.astype(str)
    uniq, cnt = np.unique(methods, return_counts=True)

    print("\n=== Surface position constraint summary ===")
    for u, c in zip(uniq, cnt):
        print(f"{u:>8s} : {int(c)}")

    # stats su pos_age (quanto è distante temporalmente la posizione usata)
    age = ds["pos_age_s"].values.astype(float)
    age = age[np.isfinite(age)]
    if age.size:
        q = np.percentile(np.abs(age), [10, 50, 90, 99])
        print("\n|pos_age_s| quantiles (s): p10={:.0f}, p50={:.0f}, p90={:.0f}, p99={:.0f}".format(*q))

    N = 12
    ncyc = int(ds.sizes["cycle"])
    N = min(N, ncyc)

    print(f"\n=== First {N} cycles ===")
    print(
        "cyc  t_surface_end            source    pos_age(s)   "
        "dtB(s)    dtA(s)    gap(s)   dtNear(s)    lat_surf    lon_surf    t_pos_used"
    )

    for cyc in range(1, N + 1):
        row = ds.sel(cycle=cyc)

        t_surf = row["t_surface_end"].values
        src = str(row["pos_source"].values)

        age_s = float(row["pos_age_s"].values) if np.isfinite(row["pos_age_s"].values) else np.nan
        dtB = float(row["dt_before_s"].values) if np.isfinite(row["dt_before_s"].values) else np.nan
        dtA = float(row["dt_after_s"].values) if np.isfinite(row["dt_after_s"].values) else np.nan
        gap = float(row["gap_s"].values) if np.isfinite(row["gap_s"].values) else np.nan
        dtN = float(row["dt_nearest_s"].values) if np.isfinite(row["dt_nearest_s"].values) else np.nan

        lat = float(row["lat_surface_end"].values) if np.isfinite(row["lat_surface_end"].values) else np.nan
        lon = float(row["lon_surface_end"].values) if np.isfinite(row["lon_surface_end"].values) else np.nan

        t_used = row["t_pos_used"].values

        def f(x, w=8, fmt="7.0f"):
            if not np.isfinite(x):
                return " " * w
            return f"{x:{fmt}}"

        def fc(x):
            if not np.isfinite(x):
                return " " * 10
            return f"{x:10.5f}"

        print(
            f"{cyc:3d}  "
            f"{_fmt_dt64(t_surf):>22s}  "
            f"{src:>7s}  "
            f"{f(age_s, w=9, fmt='8.0f'):>9s}  "
            f"{f(dtB):>8s}  {f(dtA):>8s}  {f(gap):>8s}  {f(dtN):>8s}  "
            f"{fc(lat)}  {fc(lon)}  "
            f"{_fmt_dt64(t_used):>22s}"
        )

    ds.close()


if __name__ == "__main__":
    main()