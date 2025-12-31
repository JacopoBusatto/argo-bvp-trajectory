"""ENU reconstruction helpers for a single cycle using Fubini BVP."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .fubini import integrate_fubini_1d

EARTH_RADIUS_M = 6371000.0


def latlon_to_enu_m(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon (deg) to local ENU east/north meters."""
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    lat0_rad = np.deg2rad(float(lat0))
    dlat = np.deg2rad(lat_arr - float(lat0))
    dlon = np.deg2rad(lon_arr - float(lon0))
    north = EARTH_RADIUS_M * dlat
    east = EARTH_RADIUS_M * np.cos(lat0_rad) * dlon
    return east, north


def enu_to_latlon(
    east_m: np.ndarray | float,
    north_m: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert local ENU east/north meters to lat/lon (deg)."""
    east_arr = np.asarray(east_m, dtype=float)
    north_arr = np.asarray(north_m, dtype=float)
    lat0_rad = np.deg2rad(float(lat0))
    dlat = north_arr / EARTH_RADIUS_M
    dlon = east_arr / (EARTH_RADIUS_M * np.cos(lat0_rad))
    lat = float(lat0) + np.rad2deg(dlat)
    lon = float(lon0) + np.rad2deg(dlon)
    return lat, lon


def reconstruct_xy_enu_fubini(
    t_s: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    x0: float,
    y0: float,
    xT: float,
    yT: float,
    phase: np.ndarray | None = None,
    underwater_mask: np.ndarray | None = None,
    method: str = "trap",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct ENU x/y using Fubini on underwater samples only."""
    t = np.asarray(t_s, dtype=float)
    ax_arr = np.asarray(ax, dtype=float)
    ay_arr = np.asarray(ay, dtype=float)

    if t.ndim != 1 or ax_arr.ndim != 1 or ay_arr.ndim != 1:
        raise ValueError("t_s, ax, ay must be 1D arrays")
    if not (t.size == ax_arr.size == ay_arr.size):
        raise ValueError("t_s, ax, ay must have the same length")

    if underwater_mask is None:
        if phase is None:
            raise ValueError("Provide underwater_mask or phase")
        phase_arr = np.asarray(phase)
        if phase_arr.shape[0] != t.size:
            raise ValueError("phase must match t_s length")
        mask = phase_arr != 1
    else:
        mask = np.asarray(underwater_mask, dtype=bool)
        if mask.shape[0] != t.size:
            raise ValueError("underwater_mask must match t_s length")

    if not np.any(mask):
        raise ValueError("underwater_mask has no True samples")

    t_u = t[mask]
    ax_u = ax_arr[mask]
    ay_u = ay_arr[mask]

    x_u = integrate_fubini_1d(t_u, ax_u, x0, xT, method=method)
    y_u = integrate_fubini_1d(t_u, ay_u, y0, yT, method=method)

    x_full = np.full(t.shape, np.nan, dtype=float)
    y_full = np.full(t.shape, np.nan, dtype=float)
    x_full[mask] = x_u
    y_full[mask] = y_u
    idx = np.where(mask)[0]
    x_full[idx[0]] = x0
    y_full[idx[0]] = y0
    x_full[idx[-1]] = xT
    y_full[idx[-1]] = yT

    return x_full, y_full, mask


def reconstruct_cycle_xy_from_ds(ds_cycle: xr.Dataset, method: str = "trap") -> xr.Dataset:
    """Reconstruct ENU/lat-lon for a cycle, skipping surface samples."""
    lat0 = float(ds_cycle["lat0"].values)
    lon0 = float(ds_cycle["lon0"].values)
    anchor_lat = np.asarray(ds_cycle["anchor_lat"].values, dtype=float)
    anchor_lon = np.asarray(ds_cycle["anchor_lon"].values, dtype=float)

    x0 = 0.0
    y0 = 0.0
    xT, yT = latlon_to_enu_m(anchor_lat[1], anchor_lon[1], lat0, lon0)

    if "t" in ds_cycle.coords:
        t_s = np.asarray(ds_cycle.coords["t"].values, dtype=float)
    else:
        t_s = np.asarray(ds_cycle["t"].values, dtype=float)

    lin_acc = np.asarray(ds_cycle["lin_acc"].values, dtype=float)
    ax = lin_acc[:, 0]
    ay = lin_acc[:, 1]

    phase = np.asarray(ds_cycle["phase"].values)
    x_full, y_full, mask = reconstruct_xy_enu_fubini(
        t_s,
        ax,
        ay,
        x0,
        y0,
        float(xT),
        float(yT),
        phase=phase,
        method=method,
    )

    lat_full = np.full(t_s.shape, np.nan, dtype=float)
    lon_full = np.full(t_s.shape, np.nan, dtype=float)
    lat_u, lon_u = enu_to_latlon(x_full[mask], y_full[mask], lat0, lon0)
    lat_full[mask] = lat_u
    lon_full[mask] = lon_u

    ds_out = ds_cycle.copy(deep=False)
    ds_out["x_enu"] = ("obs", x_full, {"units": "m"})
    ds_out["y_enu"] = ("obs", y_full, {"units": "m"})
    ds_out["lat_rec"] = ("obs", lat_full, {"units": "degree_north"})
    ds_out["lon_rec"] = ("obs", lon_full, {"units": "degree_east"})
    ds_out["underwater_mask"] = ("obs", mask.astype(bool))
    return ds_out


__all__ = [
    "latlon_to_enu_m",
    "enu_to_latlon",
    "reconstruct_xy_enu_fubini",
    "reconstruct_cycle_xy_from_ds",
]
