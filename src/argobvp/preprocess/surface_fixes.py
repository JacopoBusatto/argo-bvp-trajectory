from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SurfaceFixConfig:
    """
    Config per ricostruire lat/lon alla t_surface_end usando fix del TRAJ.
    """
    time_var_candidates: tuple[str, ...] = (
        # tempo associato ai record di posizione (spesso è proprio JULD su N=83042)
        "JULD",
        # altri candidati possibili (ma spesso su N_CYCLE e NON allineati a LAT/LON)
        "JULD_LAST_LOCATION",
        "JULD_FIRST_LOCATION",
        "JULD_LOCATION",
    )
    lat_var_candidates: tuple[str, ...] = ("LATITUDE", "LAT", "latitude")
    lon_var_candidates: tuple[str, ...] = ("LONGITUDE", "LON", "longitude")

    # Controlli qualità
    accepted_position_qc: tuple[str, ...] = ("1", "2", "5", "8", "A", "B")

    # Criteri per interp / nearest
    max_gap_seconds: float = 3.0 * 24.0 * 3600.0          # max distanza tra fix before/after per considerare bracket
    max_abs_dt_nearest_seconds: float = 6.0 * 3600.0      # accetta nearest se |dt| <= 6h
    max_dt_before_for_interp_seconds: float = 6.0 * 3600.0  # interp solo se before non è troppo lontano
    max_dt_after_for_interp_seconds: float = 6.0 * 3600.0   # interp solo se after non è troppo lontano


def _pick_first_existing(ds: xr.Dataset, candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _as_dt64_ns(values) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")
    raise TypeError(
        f"Time variable is not decoded to datetime64 (dtype={arr.dtype}). "
        "Open traj with xr.open_dataset(..., decode_times=True)."
    )


def _datetime64_to_seconds(t: np.ndarray) -> np.ndarray:
    """
    Convert datetime64[ns] to float seconds from epoch, with NaT -> NaN.
    """
    t = np.asarray(t).astype("datetime64[ns]")
    out = np.full(t.shape, np.nan, dtype=float)
    mask = ~np.isnat(t)
    out[mask] = t[mask].astype("datetime64[ns]").astype("int64") / 1e9
    return out


def add_surface_position_from_traj(
    ds_cycles: xr.Dataset,
    ds_traj: xr.Dataset,
    *,
    cfg: SurfaceFixConfig | None = None,
) -> xr.Dataset:
    """
    Aggiunge a ds_cycles:
      - lat_surface_end, lon_surface_end (stimati a t_surface_end)
      - pos_source: 'interp' | 'nearest' | 'missing'
      - t_pos_used (timestamp del fix usato come vincolo posizione)
      - pos_age_s = t_pos_used - t_surface_end
      - diagnostica: dt_before_s, dt_after_s, gap_s, dt_nearest_s, t_fix_before/after/nearest

    Regole:
      1) Bracket (before/after) valido se:
         - esistono entrambi
         - gap <= max_gap_seconds
      2) Interp SOLO se:
         dt_before <= max_dt_before_for_interp_seconds AND dt_after <= max_dt_after_for_interp_seconds
      3) Se non interp, fallback:
         nearest se |dt_nearest| <= max_abs_dt_nearest_seconds
      4) Altrimenti missing.
    """
    if cfg is None:
        cfg = SurfaceFixConfig()

    # --- lat/lon vars
    v_lat = _pick_first_existing(ds_traj, cfg.lat_var_candidates)
    v_lon = _pick_first_existing(ds_traj, cfg.lon_var_candidates)
    if v_lat is None or v_lon is None:
        raise RuntimeError(f"Cannot find traj lat/lon vars. Got lat={v_lat}, lon={v_lon}")

    lat = ds_traj[v_lat].values.astype(float)
    lon = ds_traj[v_lon].values.astype(float)

    # --- choose a time var aligned with lat/lon shape
    v_time = None
    for cand in cfg.time_var_candidates:
        if cand in ds_traj.variables:
            t_try = ds_traj[cand].values
            if np.asarray(t_try).shape == np.asarray(lat).shape:
                v_time = cand
                break

    if v_time is None:
        # Prova una lista più ampia, giusto in caso
        for cand in ("JULD", "JULD_LOCATION", "JULD_LAST_LOCATION", "JULD_FIRST_LOCATION"):
            if cand in ds_traj.variables:
                t_try = ds_traj[cand].values
                if np.asarray(t_try).shape == np.asarray(lat).shape:
                    v_time = cand
                    break

    if v_time is None:
        raise RuntimeError(
            "Could not find a time variable aligned with LATITUDE/LONGITUDE in traj file. "
            "Need join-by-cycle fallback."
        )

    t_fix = _as_dt64_ns(ds_traj[v_time].values)

    # Optional QC filter
    if "POSITION_QC" in ds_traj.variables:
        pos_qc = np.asarray(ds_traj["POSITION_QC"].values).astype(str)
        qc_ok = np.isin(pos_qc, np.array(cfg.accepted_position_qc, dtype=str))
    else:
        qc_ok = np.ones_like(lat, dtype=bool)

    valid = (~np.isnat(t_fix)) & np.isfinite(lat) & np.isfinite(lon) & qc_ok

    tv = t_fix[valid]
    lv = lat[valid]
    lov = lon[valid]

    # Sort by time
    order = np.argsort(tv.astype("datetime64[ns]"))
    tv = tv[order].astype("datetime64[ns]")
    lv = lv[order]
    lov = lov[order]

    tv_s = _datetime64_to_seconds(tv)

    # Cycle target times
    if "t_surface_end" not in ds_cycles.variables:
        raise RuntimeError("ds_cycles must contain 't_surface_end'.")

    t_target = np.asarray(ds_cycles["t_surface_end"].values).astype("datetime64[ns]")
    t_target_s = _datetime64_to_seconds(t_target)

    n = t_target.size

    lat_out = np.full((n,), np.nan, dtype=float)
    lon_out = np.full((n,), np.nan, dtype=float)

    pos_source = np.full((n,), "missing", dtype=object)

    dt_before_s = np.full((n,), np.nan, dtype=float)
    dt_after_s = np.full((n,), np.nan, dtype=float)
    gap_s = np.full((n,), np.nan, dtype=float)
    dt_nearest_s = np.full((n,), np.nan, dtype=float)

    t_fix_before = np.full((n,), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_fix_after = np.full((n,), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_fix_nearest = np.full((n,), np.datetime64("NaT"), dtype="datetime64[ns]")

    # New: chosen position time + age
    t_pos_used = np.full((n,), np.datetime64("NaT"), dtype="datetime64[ns]")
    pos_age_s = np.full((n,), np.nan, dtype=float)

    if tv.size == 0:
        return ds_cycles.assign(
            lat_surface_end=("cycle", lat_out),
            lon_surface_end=("cycle", lon_out),
            pos_source=("cycle", pos_source.astype(str)),
            t_pos_used=("cycle", t_pos_used),
            pos_age_s=("cycle", pos_age_s),
            dt_before_s=("cycle", dt_before_s),
            dt_after_s=("cycle", dt_after_s),
            gap_s=("cycle", gap_s),
            dt_nearest_s=("cycle", dt_nearest_s),
            t_fix_before=("cycle", t_fix_before),
            t_fix_after=("cycle", t_fix_after),
            t_fix_nearest=("cycle", t_fix_nearest),
        )

    for i in range(n):
        tt_s = t_target_s[i]
        if not np.isfinite(tt_s):
            continue

        # nearest
        k_near = int(np.nanargmin(np.abs(tv_s - tt_s)))
        t_fix_nearest[i] = tv[k_near]
        dt_nearest_s[i] = float(tv_s[k_near] - tt_s)

        # bracket indices
        j = int(np.searchsorted(tv_s, tt_s, side="left"))
        j0 = j - 1
        j1 = j

        can_bracket = (j0 >= 0) and (j1 < tv_s.size) and np.isfinite(tv_s[j0]) and np.isfinite(tv_s[j1])
        if can_bracket:
            before_s = tv_s[j0]
            after_s = tv_s[j1]
            g = after_s - before_s

            if (g >= 0) and (g <= cfg.max_gap_seconds):
                dtb = tt_s - before_s
                dta = after_s - tt_s

                # store bracket diagnostics
                t_fix_before[i] = tv[j0]
                t_fix_after[i] = tv[j1]
                dt_before_s[i] = float(dtb)
                dt_after_s[i] = float(dta)
                gap_s[i] = float(g)

                # interp only if both sides close enough
                if (dtb <= cfg.max_dt_before_for_interp_seconds) and (dta <= cfg.max_dt_after_for_interp_seconds):
                    w = (tt_s - before_s) / g if g > 0 else 0.0
                    lat_out[i] = (1 - w) * lv[j0] + w * lv[j1]
                    lon_out[i] = (1 - w) * lov[j0] + w * lov[j1]
                    pos_source[i] = "interp"
                    # choose a representative "pos time" for the constraint: exact target time
                    t_pos_used[i] = np.asarray(t_target[i]).astype("datetime64[ns]")
                    pos_age_s[i] = 0.0
                    continue

        # fallback: nearest
        if np.isfinite(dt_nearest_s[i]) and abs(dt_nearest_s[i]) <= cfg.max_abs_dt_nearest_seconds:
            lat_out[i] = lv[k_near]
            lon_out[i] = lov[k_near]
            pos_source[i] = "nearest"
            t_pos_used[i] = tv[k_near]
            pos_age_s[i] = float(tv_s[k_near] - tt_s)
        # else missing

    return ds_cycles.assign(
        lat_surface_end=("cycle", lat_out),
        lon_surface_end=("cycle", lon_out),
        pos_source=("cycle", pos_source.astype(str)),
        t_pos_used=("cycle", t_pos_used),
        pos_age_s=("cycle", pos_age_s),
        dt_before_s=("cycle", dt_before_s),
        dt_after_s=("cycle", dt_after_s),
        gap_s=("cycle", gap_s),
        dt_nearest_s=("cycle", dt_nearest_s),
        t_fix_before=("cycle", t_fix_before),
        t_fix_after=("cycle", t_fix_after),
        t_fix_nearest=("cycle", t_fix_nearest),
    )