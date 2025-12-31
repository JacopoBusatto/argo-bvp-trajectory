"""Local tangent-plane ENU conversions (km-scale approximation)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

EARTH_RADIUS_M = 6371000.0


def latlon_to_enu_m(
    lat: np.ndarray | float,
    lon: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
    e_m: np.ndarray | float,
    n_m: np.ndarray | float,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local ENU east/north meters to lat/lon (deg)."""
    e_arr = np.asarray(e_m, dtype=float)
    n_arr = np.asarray(n_m, dtype=float)

    lat0_rad = np.deg2rad(float(lat0))
    dlat = n_arr / EARTH_RADIUS_M
    dlon = e_arr / (EARTH_RADIUS_M * np.cos(lat0_rad))

    lat = float(lat0) + np.rad2deg(dlat)
    lon = float(lon0) + np.rad2deg(dlon)
    return lat, lon


__all__ = ["EARTH_RADIUS_M", "enu_to_latlon", "latlon_to_enu_m"]
