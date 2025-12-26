from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..integrators import integrate_2nd_order
from .geo import lonlat_to_local_m, local_m_to_lonlat
from .phases import macro_phase, phase_stats


@dataclass(frozen=True)
class SolveConfig:
    acc_n_var: str = "acc_n"
    acc_e_var: str = "acc_e"
    drop_non_positive_dt: bool = True
    assume_surface_anchor: bool = True  # use surface_end as both start/end anchor if nothing else


def _require(ds: xr.Dataset, vars: List[str], label: str):
    missing = [v for v in vars if v not in ds.variables]
    if missing:
        raise KeyError(f"{label} missing required variables: {missing}")


def _cycle_obs_slice(ds_bvp: xr.Dataset, cyc: int) -> xr.Dataset:
    if "row_start" in ds_bvp.variables and "row_size" in ds_bvp.variables:
        row = ds_bvp.sel(cycle=cyc)
        i0 = int(row["row_start"].values)
        n = int(row["row_size"].values)
        return ds_bvp.isel(obs=slice(i0, i0 + n))
    # fallback by mask
    mask = np.asarray(ds_bvp["cycle_number_for_obs"].values) == int(cyc)
    return ds_bvp.sel(obs=mask)


def _sanitize_time(time: np.ndarray, drop_non_positive: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    t_ns = np.asarray(time).astype("datetime64[ns]")
    t0 = t_ns[0]
    t_s = (t_ns - t0).astype("timedelta64[ns]").astype(float) / 1e9
    if drop_non_positive:
        dt = np.diff(t_s)
        keep = np.concatenate([[True], dt > 0])
        if not np.all(keep):
            t_s = t_s[keep]
            idx = np.where(keep)[0]
            return t_s, idx
    return t_s, np.arange(t_s.size)


def _integrate_cycle(
    obs: xr.Dataset,
    *,
    acc_n_var: str,
    acc_e_var: str,
    anchor_lon: Optional[float],
    anchor_lat: Optional[float],
    anchor_target_lon: Optional[float],
    anchor_target_lat: Optional[float],
    has_descent: bool,
    has_parking: bool,
    has_ascent: bool,
    cfg: SolveConfig,
) -> Dict[str, np.ndarray]:
    t_s, keep_idx = _sanitize_time(obs["time"].values, drop_non_positive=cfg.drop_non_positive_dt)
    acc_n = np.asarray(obs[acc_n_var].values, dtype=float)[keep_idx]
    acc_e = np.asarray(obs[acc_e_var].values, dtype=float)[keep_idx]
    pres = np.asarray(obs["z_from_pres"].values, dtype=float)[keep_idx]
    phases = np.asarray(obs["phase_name"].values)[keep_idx].astype(str)
    macro = np.array([macro_phase(p) for p in phases], dtype=object)

    dt = np.diff(t_s, prepend=t_s[0])

    # Reference anchor: force origin at start of descent (immersion) for consistency across cycles
    r0 = np.zeros(2, dtype=float)
    lon0, lat0 = (anchor_lon, anchor_lat) if (anchor_lon is not None and anchor_lat is not None) else (0.0, 0.0)

    used_bvp = False
    misfit_end = np.nan
    r_target = None

    if anchor_target_lon is not None and anchor_target_lat is not None:
        r_t_e, r_t_n = lonlat_to_local_m(anchor_target_lon, anchor_target_lat, lon0, lat0)
        r_target = np.array([r_t_e, r_t_n], dtype=float)

        # trapezoid integral of (T - t) * a(t)
        T = t_s[-1] - t_s[0]
        if T <= 0:
            T = 1e-6
        a_stack = np.stack([acc_e, acc_n], axis=1)
        integral = np.zeros(2, dtype=float)
        for i in range(len(t_s) - 1):
            dt_i = t_s[i + 1] - t_s[i]
            w0 = T - t_s[i]
            w1 = T - t_s[i + 1]
            integral += 0.5 * (a_stack[i] * w0 + a_stack[i + 1] * w1) * dt_i
        v0 = (r_target - r0 - integral) / T
        used_bvp = True
    else:
        v0 = np.zeros(2, dtype=float)

    # Integrate forward with simple kinematics (Euler on v, position update with constant accel per step)
    v = np.zeros((t_s.size, 2), dtype=float)
    r = np.zeros((t_s.size, 2), dtype=float)
    v[0] = v0
    r[0] = r0
    for i in range(t_s.size - 1):
        dt_i = dt[i + 1]
        a_i = np.array([acc_e[i], acc_n[i]], dtype=float)
        v[i + 1] = v[i] + a_i * dt_i
        r[i + 1] = r[i] + v[i] * dt_i + 0.5 * a_i * dt_i * dt_i

    if used_bvp and r_target is not None:
        misfit_end = float(np.linalg.norm(r[-1] - r_target))

    # Z as positive downward (pressure ~ meters)
    z_m = np.asarray(pres, dtype=float)

    # Optional back-conversion to lat/lon
    lon = np.zeros_like(z_m, dtype=float)
    lat = np.zeros_like(z_m, dtype=float)
    for i in range(r.shape[0]):
        lon[i], lat[i] = local_m_to_lonlat(r[i, 0], r[i, 1], lon0, lat0)

    stats = phase_stats(macro, t_s)

    # Metrics
    delta_start = np.nan
    delta_end = np.nan
    if has_descent:
        idx_parking = np.where(macro == "parking")[0]
        if idx_parking.size > 0:
            rn, re = r[idx_parking[0], 1], r[idx_parking[0], 0]
            delta_start = float(np.sqrt(rn * rn + re * re))
    if has_ascent and r_target is not None:
        delta_end = misfit_end

    integration_mode = []
    if has_descent:
        integration_mode.append("descent")
    integration_mode.append("park")
    if has_ascent:
        integration_mode.append("ascent")
    integration_mode = "+".join(integration_mode)

    t_ns = np.asarray(obs["time"].values)[keep_idx]
    out = dict(
        time=t_ns,
        phase_name=phases,
        macro_phase=macro,
        x_east_m=r[:, 0],
        y_north_m=r[:, 1],
        z_m=z_m,
        lat=lat,
        lon=lon,
        cycle_number=int(obs["cycle_number_for_obs"].values[0]),
        cycle_number_for_obs=np.full_like(t_ns, int(obs["cycle_number_for_obs"].values[0]), dtype=int),
        integration_mode=integration_mode,
        has_descent=bool(has_descent),
        has_parking=bool(has_parking),
        has_ascent=bool(has_ascent),
        used_bvp=bool(used_bvp),
        misfit_end_m=misfit_end,
        delta_start_m=delta_start,
        delta_end_m=delta_end,
        T_total_s=float(t_s[-1] - t_s[0]) if t_s.size > 1 else 0.0,
        phase_stats=stats,
        lon0=lon0,
        lat0=lat0,
    )
    return out


def solve_bvp_ready(
    path_bvp_ready: Path,
    *,
    path_cycles: Optional[Path] = None,
    cfg: SolveConfig | None = None,
) -> xr.Dataset:
    cfg = cfg or SolveConfig()
    ds_bvp = xr.open_dataset(path_bvp_ready)
    ds_cycles = xr.open_dataset(path_cycles) if path_cycles else None

    _require(
        ds_bvp,
        ["time", "z_from_pres", "phase_name", "cycle_number", "cycle_number_for_obs", cfg.acc_n_var, cfg.acc_e_var],
        "bvp_ready",
    )

    cycles = np.asarray(ds_bvp["cycle_number"].values).astype(int)
    cycles_unique = np.unique(cycles)

    obs_records: List[Dict] = []
    cycle_records: List[Dict] = []

    for cyc in cycles_unique:
        obs = _cycle_obs_slice(ds_bvp, cyc)
        if obs.sizes.get("obs", 0) == 0:
            continue

        # Identify available phases
        phases = np.asarray(obs["phase_name"].values).astype(str)
        macros = np.array([macro_phase(p) for p in phases], dtype=object)
        has_parking = np.any(macros == "parking")
        has_descent = np.any(macros == "descent")
        has_ascent = np.any(macros == "ascent")

        if not has_parking or not (has_descent or has_ascent):
            continue

        # Anchors: use surface_end fix if present; surface_start not available -> approximate with same
        lat_surface_end = None
        lon_surface_end = None
        if "lat_surface_end" in ds_bvp and "lon_surface_end" in ds_bvp:
            row_cyc = ds_bvp.sel(cycle=cyc)
            lat_surface_end = float(row_cyc["lat_surface_end"].values)
            lon_surface_end = float(row_cyc["lon_surface_end"].values)
        if (lat_surface_end is None or lon_surface_end is None) and ds_cycles is not None:
            if "lat_surface_end" in ds_cycles:
                row_cyc = ds_cycles.sel(cycle=cyc)
                lat_surface_end = float(row_cyc["lat_surface_end"].values)
                lon_surface_end = float(row_cyc["lon_surface_end"].values)

        lat_anchor = lat_surface_end
        lon_anchor = lon_surface_end
        lat_target = lat_surface_end
        lon_target = lon_surface_end

        if lat_anchor is None or lon_anchor is None:
            continue

        res = _integrate_cycle(
            obs,
            acc_n_var=cfg.acc_n_var,
            acc_e_var=cfg.acc_e_var,
            anchor_lon=lon_anchor,
            anchor_lat=lat_anchor,
            anchor_target_lon=lon_target if lon_target is not None else None,
            anchor_target_lat=lat_target if lat_target is not None else None,
            has_descent=has_descent,
            has_parking=has_parking,
            has_ascent=has_ascent,
            cfg=cfg,
        )

        n_obs = res["time"].shape[0]
        obs_records.append(
            dict(
                time=res["time"],
                phase_name=res["phase_name"],
                macro_phase=res["macro_phase"],
                x_east_m=res["x_east_m"],
                y_north_m=res["y_north_m"],
                z_m=res["z_m"],
                lat=res["lat"],
                lon=res["lon"],
                cycle_number=np.full(n_obs, res["cycle_number"], dtype=int),
            )
        )

        cycle_records.append(
            dict(
                cycle_number=res["cycle_number"],
                integration_mode=res["integration_mode"],
                has_descent_data=res["has_descent"],
                has_parking_data=res["has_parking"],
                has_ascent_data=res["has_ascent"],
                used_bvp=res["used_bvp"],
                misfit_end_m=res["misfit_end_m"],
                delta_start_m=res["delta_start_m"],
                delta_end_m=res["delta_end_m"],
                T_total_s=res["T_total_s"],
                lon0=res["lon0"],
                lat0=res["lat0"],
                descent_n_obs=res["phase_stats"]["descent"]["n_obs"],
                descent_median_dt_s=res["phase_stats"]["descent"]["median_dt_s"],
                parking_n_obs=res["phase_stats"]["parking"]["n_obs"],
                parking_median_dt_s=res["phase_stats"]["parking"]["median_dt_s"],
                ascent_n_obs=res["phase_stats"]["ascent"]["n_obs"],
                ascent_median_dt_s=res["phase_stats"]["ascent"]["median_dt_s"],
                surface_n_obs=res["phase_stats"]["surface"]["n_obs"],
                surface_median_dt_s=res["phase_stats"]["surface"]["median_dt_s"],
                profile_n_obs=res["phase_stats"]["profile"]["n_obs"],
                profile_median_dt_s=res["phase_stats"]["profile"]["median_dt_s"],
                grounded_n_obs=res["phase_stats"]["grounded"]["n_obs"],
                grounded_median_dt_s=res["phase_stats"]["grounded"]["median_dt_s"],
                other_n_obs=res["phase_stats"]["other"]["n_obs"],
                other_median_dt_s=res["phase_stats"]["other"]["median_dt_s"],
            )
        )

    # Assemble datasets
    if not obs_records:
        raise RuntimeError("No cycles solved (empty BVP-ready input?).")

    obs_concat = {k: np.concatenate([rec[k] for rec in obs_records]) for k in obs_records[0].keys()}
    ds_obs = xr.Dataset(
        coords=dict(obs=("obs", np.arange(obs_concat["time"].size, dtype=int))),
        data_vars={k: ("obs", v) for k, v in obs_concat.items()},
    )

    cyc_concat = {k: np.array([rec[k] for rec in cycle_records]) for k in cycle_records[0].keys()}
    ds_cyc = xr.Dataset(
        coords=dict(cycle=("cycle", cyc_concat["cycle_number"].astype(int))),
        data_vars={k: ("cycle", v) for k, v in cyc_concat.items()},
    )

    data_vars = {}
    data_vars.update({k: v for k, v in ds_obs.data_vars.items()})
    data_vars.update({k: v for k, v in ds_cyc.data_vars.items()})
    ds_out = xr.Dataset(
        coords=dict(obs=ds_obs.obs, cycle=ds_cyc.cycle),
        data_vars=data_vars,
        attrs=dict(
            platform=str(ds_bvp.attrs.get("platform", "")),
            acc_source=ds_bvp.attrs.get("acc_source", ""),
            notes="First-pass solver: integrates horizontal kinematics over attendible phases; parking mandatory gate.",
        ),
    )

    ds_bvp.close()
    if ds_cycles is not None:
        ds_cycles.close()

    return ds_out
