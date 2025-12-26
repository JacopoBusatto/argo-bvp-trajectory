from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

"""
Analytic synthetic generator for Coriolis-like IMU datasets.

One single physical truth trajectory is defined analytically over a 24h cycle.
All synthetic cycles are parallel replicas of this truth; differences come only
from IMU sampling cadence (and optional noise, currently off).

Phases: surface_before -> descent -> park_drift -> ascent -> surface_after.
"""

BASE_TIME = np.datetime64("2020-01-01T00:00:00", "s")
PLATFORM = "SYNTHETIC01"
OUT_DIR = Path("outputs/synthetic")
RAW_DIR = OUT_DIR / "raw"
PREPROCESS_SUBDIR = "preprocess"
TRUTH_PLOTS_DIR = OUT_DIR / "truth_plots"

# Cycle timing (seconds)
SURFACE_BEFORE_S = int(0.5 * 3600)
DESCENT_S = int(3 * 3600)
PARK_S = int(17.5 * 3600)
ASCENT_S = int(3 * 3600)
SURFACE_AFTER_S = int(0.5 * 3600)
T_CYCLE = SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S + SURFACE_AFTER_S

DT_TRUTH = 1.0  # seconds

# Vertical profile
Z_MAX = 1000.0  # meters, positive downward

# Descent/ascent helical parameters
R_HELIX = 50.0
T_HELIX = 10 * 60.0
OMEGA_HELIX = 2.0 * math.pi / T_HELIX
VZ_DESC = Z_MAX / DESCENT_S
VZ_ASC = Z_MAX / ASCENT_S

# Parking arc parameters
R_MES = 5000.0
F_ARC = 0.25  # fraction of circle during parking
THETA_TOTAL = 2.0 * math.pi * F_ARC
OMEGA_PARK = THETA_TOTAL / PARK_S
THETA0_PARK = 0.0
R_SWIRL = 200.0
T_SWIRL = 45 * 60.0
OMEGA_SWIRL = 2.0 * math.pi / T_SWIRL
PHI_SWIRL = 0.3
Z_HELIX_AMP = 5.0
OMEGA_Z = 2.0 * math.pi / (30 * 60.0)

# IMU sampling per cycle (phase dependent)
# DESC_DT_LEVELS = [10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0]
# PARK_DT_LEVELS = [30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1200.0, 1800.0]
# ASCENT_DT_LEVELS = DESC_DT_LEVELS
DESC_DT_LEVELS = [1.]
PARK_DT_LEVELS = [1.]
ASCENT_DT_LEVELS = DESC_DT_LEVELS
SURFACE_DT = 60.0

G = 9.80665


def _phase_schedule() -> List[Tuple[str, float, float]]:
    return [
        ("surface", 0.0, SURFACE_BEFORE_S),
        ("descent", SURFACE_BEFORE_S, SURFACE_BEFORE_S + DESCENT_S),
        ("park_drift", SURFACE_BEFORE_S + DESCENT_S, SURFACE_BEFORE_S + DESCENT_S + PARK_S),
        ("ascent", SURFACE_BEFORE_S + DESCENT_S + PARK_S, SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S),
        ("surface", SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S, T_CYCLE),
    ]


def _descent_state(tau: float) -> Tuple[float, float, float, float, float, float]:
    x = R_HELIX * math.cos(OMEGA_HELIX * tau)
    y = R_HELIX * math.sin(OMEGA_HELIX * tau)
    z = VZ_DESC * tau
    vx = -R_HELIX * OMEGA_HELIX * math.sin(OMEGA_HELIX * tau)
    vy = R_HELIX * OMEGA_HELIX * math.cos(OMEGA_HELIX * tau)
    vz = VZ_DESC
    ax = -R_HELIX * (OMEGA_HELIX ** 2) * math.cos(OMEGA_HELIX * tau)
    ay = -R_HELIX * (OMEGA_HELIX ** 2) * math.sin(OMEGA_HELIX * tau)
    az = 0.0
    return x, y, z, vx, vy, vz, ax, ay, az


def _ascent_state(tau: float, x0: float, y0: float) -> Tuple[float, float, float, float, float, float]:
    x = x0 + R_HELIX * math.cos(OMEGA_HELIX * tau + math.pi / 3.0)
    y = y0 + R_HELIX * math.sin(OMEGA_HELIX * tau + math.pi / 3.0)
    z = Z_MAX - VZ_ASC * tau
    vx = -R_HELIX * OMEGA_HELIX * math.sin(OMEGA_HELIX * tau + math.pi / 3.0)
    vy = R_HELIX * OMEGA_HELIX * math.cos(OMEGA_HELIX * tau + math.pi / 3.0)
    vz = -VZ_ASC
    ax = -R_HELIX * (OMEGA_HELIX ** 2) * math.cos(OMEGA_HELIX * tau + math.pi / 3.0)
    ay = -R_HELIX * (OMEGA_HELIX ** 2) * math.sin(OMEGA_HELIX * tau + math.pi / 3.0)
    az = 0.0
    return x, y, z, vx, vy, vz, ax, ay, az


def _parking_state(tau: float, x_center: float, y_center: float) -> Tuple[float, float, float, float, float, float]:
    theta = THETA0_PARK + OMEGA_PARK * tau
    x_base = x_center + R_MES * math.cos(theta)
    y_base = y_center + R_MES * math.sin(theta)
    dx_base = -R_MES * OMEGA_PARK * math.sin(theta)
    dy_base = R_MES * OMEGA_PARK * math.cos(theta)
    ddx_base = -R_MES * (OMEGA_PARK ** 2) * math.cos(theta)
    ddy_base = -R_MES * (OMEGA_PARK ** 2) * math.sin(theta)

    dx_swirl = R_SWIRL * math.cos(OMEGA_SWIRL * tau + PHI_SWIRL)
    dy_swirl = R_SWIRL * math.sin(OMEGA_SWIRL * tau + PHI_SWIRL)
    ddx_swirl = -R_SWIRL * (OMEGA_SWIRL ** 2) * math.cos(OMEGA_SWIRL * tau + PHI_SWIRL)
    ddy_swirl = -R_SWIRL * (OMEGA_SWIRL ** 2) * math.sin(OMEGA_SWIRL * tau + PHI_SWIRL)
    d2x_swirl = ddx_swirl
    d2y_swirl = ddy_swirl

    x = x_base + dx_swirl
    y = y_base + dy_swirl
    vx = dx_base - R_SWIRL * OMEGA_SWIRL * math.sin(OMEGA_SWIRL * tau + PHI_SWIRL)
    vy = dy_base + R_SWIRL * OMEGA_SWIRL * math.cos(OMEGA_SWIRL * tau + PHI_SWIRL)
    ax = ddx_base + d2x_swirl
    ay = ddy_base + d2y_swirl
    z = Z_MAX + Z_HELIX_AMP * math.sin(OMEGA_Z * tau)
    vz = Z_HELIX_AMP * OMEGA_Z * math.cos(OMEGA_Z * tau)
    az = -Z_HELIX_AMP * (OMEGA_Z ** 2) * math.sin(OMEGA_Z * tau)
    return x, y, z, vx, vy, vz, ax, ay, az


def build_truth() -> Dict[str, np.ndarray]:
    t = np.arange(0.0, T_CYCLE + DT_TRUTH, DT_TRUTH, dtype=float)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    z = np.zeros_like(t)
    vx = np.zeros_like(t)
    vy = np.zeros_like(t)
    vz = np.zeros_like(t)
    ax = np.zeros_like(t)
    ay = np.zeros_like(t)
    az = np.zeros_like(t)
    phase_name = np.empty_like(t, dtype=object)

    # surface_before: stay at origin
    mask_surf_before = (t >= 0) & (t < SURFACE_BEFORE_S)
    phase_name[mask_surf_before] = "surface"

    # descent
    mask_desc = (t >= SURFACE_BEFORE_S) & (t < SURFACE_BEFORE_S + DESCENT_S)
    tau_desc = t[mask_desc] - SURFACE_BEFORE_S
    for i, tau in zip(np.where(mask_desc)[0], tau_desc):
        x[i], y[i], z[i], vx[i], vy[i], vz[i], ax[i], ay[i], az[i] = _descent_state(tau)
        z[i] = z[i]  # positive downward
        phase_name[i] = "descent"

    # parking
    mask_park = (t >= SURFACE_BEFORE_S + DESCENT_S) & (t < SURFACE_BEFORE_S + DESCENT_S + PARK_S)
    x_desc_end = x[np.where(mask_desc)[0][-1]] if np.any(mask_desc) else 0.0
    y_desc_end = y[np.where(mask_desc)[0][-1]] if np.any(mask_desc) else 0.0
    x_center = x_desc_end - (R_MES * math.cos(THETA0_PARK) + R_SWIRL * math.cos(PHI_SWIRL))
    y_center = y_desc_end - (R_MES * math.sin(THETA0_PARK) + R_SWIRL * math.sin(PHI_SWIRL))
    tau_park = t[mask_park] - (SURFACE_BEFORE_S + DESCENT_S)
    for i, tau in zip(np.where(mask_park)[0], tau_park):
        x[i], y[i], z[i], vx[i], vy[i], vz[i], ax[i], ay[i], az[i] = _parking_state(tau, x_center, y_center)
        phase_name[i] = "park_drift"

    # ascent (start from parking end point)
    idx_park_end = np.where(mask_park)[0][-1]
    x0_ascent = x[idx_park_end]
    y0_ascent = y[idx_park_end]
    mask_ascent = (t >= SURFACE_BEFORE_S + DESCENT_S + PARK_S) & (t < SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S)
    tau_asc = t[mask_ascent] - (SURFACE_BEFORE_S + DESCENT_S + PARK_S)
    for i, tau in zip(np.where(mask_ascent)[0], tau_asc):
        x[i], y[i], z[i], vx[i], vy[i], vz[i], ax[i], ay[i], az[i] = _ascent_state(tau, x0_ascent, y0_ascent)
        phase_name[i] = "ascent"

    # surface_after: hold last point
    idx_ascent_end = np.where(mask_ascent)[0][-1] if np.any(mask_ascent) else idx_park_end
    mask_surf_after = t >= SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S
    x[mask_surf_after] = x[idx_ascent_end] + 0.0
    y[mask_surf_after] = y[idx_ascent_end] + 0.0
    z[mask_surf_after] = 0.0
    vx[mask_surf_after] = 0.0
    vy[mask_surf_after] = 0.0
    vz[mask_surf_after] = 0.0
    ax[mask_surf_after] = 0.0
    ay[mask_surf_after] = 0.0
    az[mask_surf_after] = 0.0
    phase_name[mask_surf_after] = "surface"

    # fill any unset phase_name (last point)
    phase_name[phase_name == None] = "surface"  # type: ignore

    return dict(
        t_seconds=t,
        time=BASE_TIME + t.astype("timedelta64[s]"),
        x_true=x,
        y_true=y,
        z_true=z,
        vx_true=vx,
        vy_true=vy,
        vz_true=vz,
        ax_true=ax,
        ay_true=ay,
        az_true=az,
        phase_name=np.asarray(phase_name, dtype=str),
    )


def _truth_anchor_indices(truth: Dict[str, np.ndarray]) -> Tuple[int, int, int]:
    t = truth["t_seconds"]
    phase = truth["phase_name"]

    mask_surface_before = t < SURFACE_BEFORE_S
    if not np.any(mask_surface_before):
        raise RuntimeError("No surface_before samples found in truth.")
    idx_surface_before_end = int(np.where(mask_surface_before)[0][-1])

    mask_ascent = phase == "ascent"
    if not np.any(mask_ascent):
        raise RuntimeError("No ascent samples found in truth.")
    idx_ascent_end = int(np.where(mask_ascent)[0][-1])

    surf_after_start_t = SURFACE_BEFORE_S + DESCENT_S + PARK_S + ASCENT_S
    mask_surface_after = t >= surf_after_start_t
    if not np.any(mask_surface_after):
        raise RuntimeError("No surface_after samples found in truth.")
    idx_surface_after_start = int(np.where(mask_surface_after)[0][0])

    return idx_surface_before_end, idx_ascent_end, idx_surface_after_start


def _plot_truth(truth: Dict[str, np.ndarray]) -> None:
    TRUTH_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    phase = truth["phase_name"]
    x = truth["x_true"]
    y = truth["y_true"]
    z = truth["z_true"]
    t = truth["t_seconds"] / 3600.0
    ax_e = truth["ax_true"]
    ay_e = truth["ay_true"]
    colors = {"surface": "gray", "descent": "tab:green", "park_drift": "tab:blue", "ascent": "tab:orange"}

    fig, axp = plt.subplots(figsize=(7, 6))
    for ph in np.unique(phase):
        m = phase == ph
        axp.plot(x[m], y[m], ".", ms=1, color=colors.get(ph, "k"), label=ph)
    axp.set_xlabel("East (m)")
    axp.set_ylabel("North (m)")
    axp.set_title("Truth plan view")
    axp.legend()
    axp.grid(True, ls="--", alpha=0.4)
    fig.savefig(TRUTH_PLOTS_DIR / "truth_plan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 7))
    ax3d = fig.add_subplot(111, projection="3d")
    for ph in np.unique(phase):
        m = phase == ph
        ax3d.plot(x[m], y[m], -z[m], color=colors.get(ph, "k"), label=ph, lw=1.2)
    ax3d.set_xlabel("East (m)")
    ax3d.set_ylabel("North (m)")
    ax3d.set_zlabel("Depth (m)")
    ax3d.set_title("Truth trajectory (3D)")
    max_range = np.max([x.max() - x.min(), y.max() - y.min()])
    xm = 0.5 * (x.max() + x.min())
    ym = 0.5 * (y.max() + y.min())
    ax3d.set_xlim(xm - max_range / 2, xm + max_range / 2)
    ax3d.set_ylim(ym - max_range / 2, ym + max_range / 2)
    ax3d.legend()
    fig.savefig(TRUTH_PLOTS_DIR / "truth_traj_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(t, x, label="x (m)")
    axs[0].plot(t, y, label="y (m)")
    axs[0].legend()
    axs[0].grid(True, ls="--", alpha=0.4)
    axs[1].plot(t, z, label="z (m)")
    axs[1].legend()
    axs[1].grid(True, ls="--", alpha=0.4)
    axs[2].plot(t, ax_e, label="acc_e (m/s2)")
    axs[2].plot(t, ay_e, label="acc_n (m/s2)")
    axs[2].legend()
    axs[2].grid(True, ls="--", alpha=0.4)
    axs[2].set_xlabel("Time (hours)")
    fig.savefig(TRUTH_PLOTS_DIR / "truth_time_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for ph in ["descent", "park_drift", "ascent", "surface"]:
        m = phase == ph
        if np.any(m):
            rms = math.sqrt(np.mean(ax_e[m] ** 2 + ay_e[m] ** 2))
            amax = float(np.max(np.sqrt(ax_e[m] ** 2 + ay_e[m] ** 2)))
            print(f"[truth] {ph} acc rms={rms:.4f} m/s2, max={amax:.4f} m/s2")


def write_ground_truth(truth: Dict[str, np.ndarray]) -> Path:
    ds_gt = xr.Dataset(
        coords=dict(obs_truth=("obs_truth", truth["time"])),
        data_vars=dict(
            time=("obs_truth", truth["time"]),
            x_true=("obs_truth", truth["x_true"]),
            y_true=("obs_truth", truth["y_true"]),
            z_true=("obs_truth", truth["z_true"]),
            phase_name=("obs_truth", truth["phase_name"]),
        ),
        attrs=dict(platform=PLATFORM, notes="Analytic single-cycle truth trajectory"),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "ground_truth.nc"
    ds_gt.to_netcdf(path, mode="w")
    ds_gt.close()
    print(f"[step1] wrote truth to {path}")
    return path


def meters_to_latlon(delta_n: float, delta_e: float, lat0_deg: float, lon0_deg: float) -> Tuple[float, float]:
    deg_per_m_lat = 1.0 / 111320.0
    lat_deg = lat0_deg + delta_n * deg_per_m_lat
    deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat_deg)))
    lon_deg = lon0_deg + delta_e * deg_per_m_lon
    return lat_deg, lon_deg


def sample_cycle(truth: Dict[str, np.ndarray], dt_desc: float, dt_park: float, dt_ascent: float) -> Dict[str, np.ndarray]:
    t_truth = truth["t_seconds"]
    phase = truth["phase_name"]

    times = []
    phases_out = []
    for name, t0, t1 in _phase_schedule():
        dt = SURFACE_DT if name == "surface" else (dt_desc if name == "descent" else dt_park if name == "park_drift" else dt_ascent)
        ts = np.arange(t0, t1 + 1e-6, dt)
        # ensure endpoints included
        if ts[-1] < t1:
            ts = np.append(ts, t1)
        times.append(ts)
        phases_out.append(np.full(ts.shape, name, dtype=object))
    t_sample = np.concatenate(times)
    phase_sample = np.concatenate(phases_out)

    def interp(arr: np.ndarray) -> np.ndarray:
        return np.interp(t_sample, t_truth, arr)

    x = interp(truth["x_true"])
    y = interp(truth["y_true"])
    z = interp(truth["z_true"])
    ax = interp(truth["ax_true"])
    ay = interp(truth["ay_true"])
    az = interp(truth["az_true"])

    return dict(
        time=BASE_TIME + t_sample.astype("timedelta64[s]"),
        t_seconds=t_sample,
        phase_name=phase_sample.astype(str),
        x=x,
        y=y,
        z=z,
        acc_e=ax,
        acc_n=ay,
        acc_d=az,
    )


def build_raw_dataset(truth: Dict[str, np.ndarray], n_cycles: int = 1) -> Tuple[xr.Dataset, xr.Dataset]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_vars = {
        "time": [],
        "pres": [],
        "cycle_number": [],
        "measurement_code": [],
        "LINEAR_ACCELERATION_COUNT_X": [],
        "LINEAR_ACCELERATION_COUNT_Y": [],
        "LINEAR_ACCELERATION_COUNT_Z": [],
        "ANGULAR_RATE_COUNT_X": [],
        "ANGULAR_RATE_COUNT_Y": [],
        "ANGULAR_RATE_COUNT_Z": [],
        "MAGNETIC_FIELD_COUNT_X": [],
        "MAGNETIC_FIELD_COUNT_Y": [],
        "MAGNETIC_FIELD_COUNT_Z": [],
    }
    lat_fix = []
    lon_fix = []
    t_fix = []

    lat0 = 0.0
    lon0 = 0.0
    idx_surface_before_end, idx_ascent_end, idx_surface_after_start = _truth_anchor_indices(truth)
    x_immersion = float(truth["x_true"][idx_surface_before_end])
    y_immersion = float(truth["y_true"][idx_surface_before_end])
    x_emersion = float(truth["x_true"][idx_ascent_end])
    y_emersion = float(truth["y_true"][idx_ascent_end])
    lat_start, lon_start = meters_to_latlon(0.0, 0.0, lat0, lon0)
    lat_end, lon_end = meters_to_latlon(y_emersion, x_emersion, lat0, lon0)
    time_start_fix = truth["time"][idx_surface_before_end]
    time_end_fix = truth["time"][idx_surface_after_start]
    print(f"[truth] immersion ENU (surface_before end): x={x_immersion:.3f}, y={y_immersion:.3f}")
    print(f"[truth] emersion ENU (ascent end): x={x_emersion:.3f}, y={y_emersion:.3f}")
    print(f"[traj] surface fixes lat/lon: start=({lat_start:.6f},{lon_start:.6f}) end=({lat_end:.6f},{lon_end:.6f})")
    print("[truth] surface_after holds ascent-end position")

    for idx in range(n_cycles):
        time_shift = np.timedelta64(int(idx * T_CYCLE), "s")
        dt_desc = DESC_DT_LEVELS[min(idx, len(DESC_DT_LEVELS) - 1)]
        dt_park = PARK_DT_LEVELS[min(idx, len(PARK_DT_LEVELS) - 1)]
        dt_ascent = ASCENT_DT_LEVELS[min(idx, len(ASCENT_DT_LEVELS) - 1)]
        cyc = sample_cycle(truth, dt_desc=dt_desc, dt_park=dt_park, dt_ascent=dt_ascent)
        time_cycle = cyc["time"] + time_shift

        # measurement codes
        mc = np.full(cyc["phase_name"].shape, 290, dtype=int)
        mc[cyc["phase_name"] == "descent"] = 200
        mc[cyc["phase_name"] == "ascent"] = 503
        mc[cyc["phase_name"] == "surface"] = 689
        idx_cycle_start = np.where(cyc["t_seconds"] <= SURFACE_BEFORE_S)[0]
        if idx_cycle_start.size:
            mc[int(idx_cycle_start[-1])] = 89
        if mc.size:
            mc[-1] = 711  # last sample in air

        # accelerations -> counts (body=NED, zero gravity handling; use linear directly)
        acc_n_true = cyc["acc_n"]
        acc_e_true = cyc["acc_e"]
        acc_d_true = cyc["acc_d"] + G  # include gravity so that removal yields acc_d

        # counts hold NED directly; roll/pitch/yaw=0 => no rotation applied
        acc_x = acc_n_true / G  # treat body x as N (identity frame)
        acc_y = acc_e_true / G  # body y as E
        acc_z = acc_d_true / G  # body z as Down with gravity

        gyro = np.zeros_like(acc_x)
        mag = np.zeros_like(acc_x)

        all_vars["time"].append(time_cycle)
        all_vars["pres"].append(cyc["z"])
        all_vars["cycle_number"].append(np.full_like(cyc["time"], idx + 1, dtype=int))
        all_vars["measurement_code"].append(mc)
        all_vars["LINEAR_ACCELERATION_COUNT_X"].append(acc_x)
        all_vars["LINEAR_ACCELERATION_COUNT_Y"].append(acc_y)
        all_vars["LINEAR_ACCELERATION_COUNT_Z"].append(acc_z)
        all_vars["ANGULAR_RATE_COUNT_X"].append(gyro)
        all_vars["ANGULAR_RATE_COUNT_Y"].append(gyro)
        all_vars["ANGULAR_RATE_COUNT_Z"].append(gyro)
        all_vars["MAGNETIC_FIELD_COUNT_X"].append(mag + 1.0)
        all_vars["MAGNETIC_FIELD_COUNT_Y"].append(mag + 0.1)
        all_vars["MAGNETIC_FIELD_COUNT_Z"].append(mag - 0.2)

        # surface fixes: start/end based on immersion/emersion anchors
        t_fix.append(np.asarray([time_start_fix + time_shift, time_end_fix + time_shift], dtype="datetime64[ns]"))
        lat_fix.append([lat_start, lat_end])
        lon_fix.append([lon_start, lon_end])

    concat = {k: np.concatenate(v) for k, v in all_vars.items()}
    ds_aux = xr.Dataset(
        data_vars=dict(
            JULD=("obs", concat["time"]),
            PRES=("obs", concat["pres"]),
            CYCLE_NUMBER=("obs", concat["cycle_number"]),
            MEASUREMENT_CODE=("obs", concat["measurement_code"]),
            LINEAR_ACCELERATION_COUNT_X=("obs", concat["LINEAR_ACCELERATION_COUNT_X"]),
            LINEAR_ACCELERATION_COUNT_Y=("obs", concat["LINEAR_ACCELERATION_COUNT_Y"]),
            LINEAR_ACCELERATION_COUNT_Z=("obs", concat["LINEAR_ACCELERATION_COUNT_Z"]),
            ANGULAR_RATE_COUNT_X=("obs", concat["ANGULAR_RATE_COUNT_X"]),
            ANGULAR_RATE_COUNT_Y=("obs", concat["ANGULAR_RATE_COUNT_Y"]),
            ANGULAR_RATE_COUNT_Z=("obs", concat["ANGULAR_RATE_COUNT_Z"]),
            MAGNETIC_FIELD_COUNT_X=("obs", concat["MAGNETIC_FIELD_COUNT_X"]),
            MAGNETIC_FIELD_COUNT_Y=("obs", concat["MAGNETIC_FIELD_COUNT_Y"]),
            MAGNETIC_FIELD_COUNT_Z=("obs", concat["MAGNETIC_FIELD_COUNT_Z"]),
        ),
        attrs=dict(platform=PLATFORM, notes="Analytic synthetic Coriolis-like AUX; counts already scaled."),
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    aux_path = RAW_DIR / f"{PLATFORM}_AUX.nc"
    ds_aux.to_netcdf(aux_path)

    # TRAJ
    t_fix_arr = np.concatenate(t_fix)
    lat_arr = np.concatenate(lat_fix)
    lon_arr = np.concatenate(lon_fix)
    ds_traj = xr.Dataset(
        data_vars=dict(
            JULD=("obs", t_fix_arr),
            LATITUDE=("obs", lat_arr),
            LONGITUDE=("obs", lon_arr),
        ),
        attrs=dict(platform=PLATFORM, notes="Surface fixes at immersion/emersion (analytic)."),
    )
    traj_path = RAW_DIR / f"{PLATFORM}_TRAJ.nc"
    ds_traj.to_netcdf(traj_path)
    print(f"[step2] wrote AUX to {aux_path}, TRAJ to {traj_path}")
    ds_aux.close()
    ds_traj.close()
    return aux_path, traj_path


def write_config(aux_path: Path, traj_path: Path) -> Path:
    cfg = {
        "platform": PLATFORM,
        "paths": {"aux": str(aux_path), "traj": str(traj_path)},
        "imu": {
            "frame": "NED",
            "g": G,
            "accel": {
                "scale_g": 1.0,
                "denom": 1.0,
                "bias_counts": {"x": 0.0, "y": 0.0, "z": 0.0},
                "gain": {"x": 1.0, "y": 1.0, "z": 1.0},
                "axis_map": {"x": "X", "y": "Y", "z": "Z"},
                "sign": {"x": 1, "y": 1, "z": 1},
            },
            "gyro": {"scale": 1.0, "units": "rad/s", "bias_counts": {"x": 0.0, "y": 0.0, "z": 0.0}},
            "mag": {"hard_iron": {"x": 0.0, "y": 0.0, "z": 0.0}, "soft_iron_xy": {"xx": 1.0, "xy": 0.0, "yx": 0.0, "yy": 1.0}},
        },
        "pres_surface_max": 5.0,
        "min_parking_samples_for_bvp": 10,
        "min_phase_samples_for_bvp": 10,
    }
    cfg_path = OUT_DIR / "config_synthetic.yml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[config] wrote {cfg_path}")
    return cfg_path


def run_subprocess(cmd: List[str]) -> int:
    print("[cmd]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    if res.returncode != 0:
        print(f"[warn] command returned {res.returncode}")
    return res.returncode


def main():
    # Step 1: build truth
    truth = build_truth()
    write_ground_truth(truth)
    _plot_truth(truth)

    # Step 2: raw dataset (single truth replicated across cycles with varying dt)
    aux_path, traj_path = build_raw_dataset(truth, n_cycles=len(DESC_DT_LEVELS))
    cfg_path = write_config(aux_path, traj_path)

    # Step 3: preprocess + bvp_ready
    out_preprocess = OUT_DIR / PREPROCESS_SUBDIR
    out_preprocess.mkdir(parents=True, exist_ok=True)
    runner_cmd = [sys.executable, "-m", "argobvp.preprocess.runner", "--config", str(cfg_path), "--out", str(out_preprocess)]
    run_subprocess(runner_cmd)

    cycles_nc = out_preprocess / f"{PLATFORM}_cycles.nc"
    bvp_cmd = [
        sys.executable,
        "-m",
        "argobvp.preprocess.bvp_ready",
        "--cont",
        str(out_preprocess / f"{PLATFORM}_preprocessed_imu.nc"),
        "--cycles",
        str(cycles_nc),
        "--segments",
        str(out_preprocess / f"{PLATFORM}_segments.nc"),
        "--out",
        str(out_preprocess),
    ]
    run_subprocess(bvp_cmd)
    bvp_path = out_preprocess / f"{PLATFORM}_bvp_ready.nc"
    if bvp_path.exists():
        ds_tmp = xr.open_dataset(bvp_path)
        ds_bvp = ds_tmp.load()
        ds_tmp.close()
        ph = np.asarray(ds_bvp["phase_name"].values).astype(object)
        ph[ph == "other"] = "descent"
        ds_bvp["phase_name"] = ("obs", ph.astype(str))
        ds_bvp.to_netcdf(bvp_path)
        ds_bvp.close()

    # Step 4: solver and diagnostics
    solve_out = OUT_DIR / "solve" / f"{PLATFORM}_solved.nc"
    solve_out.parent.mkdir(parents=True, exist_ok=True)
    solve_cmd = [
        sys.executable,
        "-m",
        "argobvp.solve.runner",
        "--bvp-ready",
        str(out_preprocess / f"{PLATFORM}_bvp_ready.nc"),
        "--cycles",
        str(cycles_nc),
        "--out",
        str(solve_out),
    ]
    run_subprocess(solve_cmd)

    diag_out = OUT_DIR / "diagnostics"
    diag_out.mkdir(parents=True, exist_ok=True)
    diag_cmd = [
        sys.executable,
        "examples/diagnostics/plot_synthetic_bvp_and_solve_diagnostics.py",
        "--bvp",
        str(out_preprocess / f"{PLATFORM}_bvp_ready.nc"),
        "--cycles",
        str(cycles_nc),
        "--solved",
        str(solve_out),
        "--truth",
        str(OUT_DIR / "ground_truth.nc"),
        "--outdir",
        str(diag_out),
    ]
    run_subprocess(diag_cmd)
    print("[done] synthetic analytic benchmark generated.")

    # Debug diagnostics on bvp_ready accelerations and anchors
    debug_dir = OUT_DIR / "diagnostics_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    bvp_path = out_preprocess / f"{PLATFORM}_bvp_ready.nc"
    cycles_path = out_preprocess / f"{PLATFORM}_cycles.nc"
    solved_path = solve_out
    if bvp_path.exists():
        ds_bvp = xr.open_dataset(bvp_path)
        ds_cyc = xr.open_dataset(cycles_path)
        # Cycle 1 accelerations
        cyc_idx = 1
        mask = np.asarray(ds_bvp["cycle_number_for_obs"].values) == cyc_idx
        if np.any(mask):
            t = ds_bvp["time"].values[mask]
            acc_e = np.asarray(ds_bvp["acc_e"].values[mask], dtype=float)
            acc_n = np.asarray(ds_bvp["acc_n"].values[mask], dtype=float)
            print(f"[debug] cycle {cyc_idx} acc_e mean/std/min/max: {acc_e.mean():.4e}/{acc_e.std():.4e}/{acc_e.min():.4e}/{acc_e.max():.4e}")
            print(f"[debug] cycle {cyc_idx} acc_n mean/std/min/max: {acc_n.mean():.4e}/{acc_n.std():.4e}/{acc_n.min():.4e}/{acc_n.max():.4e}")
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(t, acc_e, label="acc_e")
            ax.plot(t, acc_n, label="acc_n")
            ax.set_xlabel("Time")
            ax.set_ylabel("Acceleration (m/s2)")
            ax.set_title(f"BVP-ready accelerations cycle {cyc_idx}")
            ax.legend()
            ax.grid(True, ls="--", alpha=0.4)
            fig.savefig(debug_dir / "bvp_ready_acc_timeseries_cycle1.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].hist(acc_e, bins=50, color="tab:blue", alpha=0.7)
            ax[0].set_title("acc_e histogram")
            ax[1].hist(acc_n, bins=50, color="tab:orange", alpha=0.7)
            ax[1].set_title("acc_n histogram")
            for a in ax:
                a.grid(True, ls="--", alpha=0.4)
            fig.savefig(debug_dir / "bvp_ready_acc_hist_cycle1.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            print("[debug] cycle 1 not found in bvp_ready")

        # Anchor table from cycles
        cols = ["cycle", "lat_surface_start", "lon_surface_start", "lat_surface_end", "lon_surface_end", "t0", "t1"]
        rows = []
        lat_start = ds_cyc["lat_surface_start"].values if "lat_surface_start" in ds_cyc else np.full(ds_cyc.sizes["cycle"], np.nan)
        lon_start = ds_cyc["lon_surface_start"].values if "lon_surface_start" in ds_cyc else np.full(ds_cyc.sizes["cycle"], np.nan)
        lat_end = ds_cyc["lat_surface_end"].values if "lat_surface_end" in ds_cyc else np.full(ds_cyc.sizes["cycle"], np.nan)
        lon_end = ds_cyc["lon_surface_end"].values if "lon_surface_end" in ds_cyc else np.full(ds_cyc.sizes["cycle"], np.nan)
        t0 = ds_cyc["t0"].values if "t0" in ds_cyc else ds_cyc["t_cycle_start"].values
        t1 = ds_cyc["t1"].values if "t1" in ds_cyc else ds_cyc["t_surface_end"].values
        for i, cyc in enumerate(ds_cyc["cycle_number"].values.astype(int)):
            rows.append(
                [
                    cyc,
                    float(lat_start[i]) if not np.isnan(lat_start[i]) else np.nan,
                    float(lon_start[i]) if not np.isnan(lon_start[i]) else np.nan,
                    float(lat_end[i]) if not np.isnan(lat_end[i]) else np.nan,
                    float(lon_end[i]) if not np.isnan(lon_end[i]) else np.nan,
                    str(t0[i]),
                    str(t1[i]),
                ]
            )
        csv_path = debug_dir / "bvp_anchors_table.csv"
        import csv as _csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"[debug] wrote anchors table to {csv_path}")
        ds_bvp.close()
        ds_cyc.close()

    # Solved keypoints
    if solved_path.exists():
        ds_sol = xr.open_dataset(solved_path)
        cols = [
            "cycle",
            "descent_start_x",
            "descent_start_y",
            "descent_start_time",
            "park_start_x",
            "park_start_y",
            "park_start_time",
            "ascent_end_x",
            "ascent_end_y",
            "ascent_end_time",
        ]
        rows = []
        cyc_nums = np.asarray(ds_sol["cycle"].values).astype(int) if "cycle" in ds_sol.coords else np.unique(ds_sol["cycle_number"].values)
        cyc_map = ds_sol["cycle_number_for_obs"].values if "cycle_number_for_obs" in ds_sol else None
        if cyc_map is not None:
            for cn in cyc_nums:
                m = np.asarray(cyc_map) == cn
                if not np.any(m):
                    continue
                ph = ds_sol["phase_name"].values[m].astype(str)
                x = ds_sol["x_east_m"].values[m]
                y = ds_sol["y_north_m"].values[m]
                t = ds_sol["time"].values[m]
                # descent start
                idx_desc = np.where(ph == "descent")[0]
                idx_park = np.where(ph == "park_drift")[0]
                idx_ascent = np.where(ph == "ascent")[0]
                dsx = x[idx_desc[0]] if idx_desc.size else x[0]
                dsy = y[idx_desc[0]] if idx_desc.size else y[0]
                dst = t[idx_desc[0]] if idx_desc.size else t[0]
                psx = x[idx_park[0]] if idx_park.size else np.nan
                psy = y[idx_park[0]] if idx_park.size else np.nan
                pst = t[idx_park[0]] if idx_park.size else np.datetime64("NaT")
                aex = x[idx_ascent[-1]] if idx_ascent.size else x[-1]
                aey = y[idx_ascent[-1]] if idx_ascent.size else y[-1]
                aet = t[idx_ascent[-1]] if idx_ascent.size else t[-1]
                rows.append([cn, dsx, dsy, str(dst), psx, psy, str(pst), aex, aey, str(aet)])
        csv2 = debug_dir / "solved_cycle_keypoints.csv"
        import csv as _csv

        with csv2.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.writer(f)
            writer.writerow(cols)
            writer.writerows(rows)
        print(f"[debug] wrote solved keypoints to {csv2}")
        ds_sol.close()


if __name__ == "__main__":
    main()
