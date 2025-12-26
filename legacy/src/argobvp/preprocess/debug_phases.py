from __future__ import annotations

from pathlib import Path
import numpy as np
import xarray as xr

from argobvp.preprocess.config import load_config
from argobvp.preprocess.io_coriolis import open_aux
from argobvp.preprocess.products import build_preprocessed_dataset
from argobvp.preprocess.cycles import build_cycle_products


def _fmt_dt(dt64) -> str:
    s = str(np.datetime_as_string(dt64, unit="s"))
    return s.replace("T", " ")


def _safe_minmax(a: np.ndarray):
    if a.size == 0:
        return np.nan, np.nan
    return float(np.nanmin(a)), float(np.nanmax(a))


def main():
    # ---- scegli UNA delle due modalità ----

    # (A) leggi direttamente il netcdf preprocessato
    preprocessed_path = Path("outputs/preprocess/4903848_preprocessed_imu.nc")

    # (B) oppure rifai preprocess dal config (commenta A e decommenta B)
    # cfg = load_config("configs/4903848.yml")
    # ds_aux = open_aux(cfg.paths.aux)
    # ds = build_preprocessed_dataset(ds_aux, cfg)
    # ds_cycles, ds_segments = build_cycle_products(ds, cfg)
    # ds_aux.close()

    if preprocessed_path.exists():
        ds = xr.open_dataset(preprocessed_path)
        # serve cfg per pres_surface_max nei keypoints; lo carichiamo dal yaml
        cfg = load_config("configs/4903848.yml")
        ds_cycles, ds_segments = build_cycle_products(ds, cfg)
    else:
        raise FileNotFoundError(f"Cannot find: {preprocessed_path}")

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    cycles = np.unique(ds["cycle_number"].values.astype(int))
    print(f"\nN cycles in continuous dataset: {len(cycles)}")
    print(f"First cycles: {cycles[:10]}\n")

    # indicizzazione rapida
    t = ds["time"].values.astype("datetime64[ns]")
    pres = ds["pres"].values.astype(float)
    cyc = ds["cycle_number"].values.astype(int)
    mc = ds["measurement_code"].values.astype(int)

    # ------------------------------------------------------------
    # For each cycle, print phase sequence from segments
    # ------------------------------------------------------------
    seg_cyc = ds_segments["cycle_number"].values.astype(int)
    seg_name = ds_segments["segment_name"].values.astype(object)
    seg_i0 = ds_segments["idx0"].values.astype(int)
    seg_i1 = ds_segments["idx1"].values.astype(int)

    # helper: keypoints per ciclo
    cyc_table = {int(c): i for i, c in enumerate(ds_cycles["cycle_number"].values.astype(int))}

    for c in cycles:
        # segments in this cycle
        mseg = np.where(seg_cyc == c)[0]
        if mseg.size == 0:
            continue

        # order segments by idx0
        order = np.argsort(seg_i0[mseg])
        mseg = mseg[order]

        # print header
        idx_cycle = np.where(cyc == c)[0]
        t0 = t[idx_cycle[0]]
        t1 = t[idx_cycle[-1]]
        print("=" * 88)
        print(f"Cycle {c} | { _fmt_dt(t0) }  ->  { _fmt_dt(t1) }  |  n_obs={idx_cycle.size}")

        # keypoints
        if c in cyc_table:
            i = cyc_table[c]
            def g(name):
                return ds_cycles[name].values[i]

            print("Keypoints:")
            print(f"  t_cycle_start            : {_fmt_dt(g('t_cycle_start'))}")
            print(f"  t_park_start             : {str(g('t_park_start'))}")
            print(f"  t_descent_to_profile_start: {str(g('t_descent_to_profile_start'))}")
            print(f"  t_profile_deepest        : {_fmt_dt(g('t_profile_deepest'))}")
            print(f"  t_ascent_start           : {str(g('t_ascent_start'))}")
            print(f"  t_surface_start          : {str(g('t_surface_start'))}")
            print(f"  t_surface_end            : {str(g('t_surface_end'))}")
            print(f"  pres_park_rep (dbar)      : {float(ds_cycles['pres_park_rep'].values[i]):.2f}")
            print(f"  pres_profile_deepest (dbar): {float(ds_cycles['pres_profile_deepest'].values[i]):.2f}")

        # phase sequence summary
        seq = [str(seg_name[k]) for k in mseg]
        print("\nPhase sequence:")
        print("  " + "  ->  ".join(seq))

        # detailed segments
        print("\nSegments:")
        for k in mseg:
            i0 = seg_i0[k]
            i1 = seg_i1[k]
            name = str(seg_name[k])

            tt0 = t[i0]
            tt1 = t[i1 - 1]
            dt_s = (tt1 - tt0) / np.timedelta64(1, "s")
            pmin, pmax = _safe_minmax(pres[i0:i1])

            # also show measurement_code range quickly
            mmin, mmax = _safe_minmax(mc[i0:i1])

            print(
                f"  [{name:14s}] "
                f"{_fmt_dt(tt0)} -> {_fmt_dt(tt1)} | "
                f"dt={dt_s:8.0f}s | "
                f"pres=[{pmin:7.1f},{pmax:7.1f}] dbar | "
                f"mcode=[{int(mmin):4d},{int(mmax):4d}] | "
                f"idx=[{i0},{i1})"
            )

        print()

    ds.close()


if __name__ == "__main__":
    main()