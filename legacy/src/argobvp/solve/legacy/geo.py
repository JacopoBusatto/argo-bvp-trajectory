from __future__ import annotations

import math
from typing import Tuple

R_EARTH = 6_371_000.0  # meters


def deg2rad(x: float) -> float:
    return x * math.pi / 180.0


def lonlat_to_local_m(lon: float, lat: float, lon0: float, lat0: float) -> Tuple[float, float]:
    """
    Convert lon/lat (deg) to local ENU meters using a simple tangent-plane approximation at (lon0, lat0).
    """
    dlat = deg2rad(lat - lat0)
    dlon = deg2rad(lon - lon0)
    north = dlat * R_EARTH
    east = dlon * R_EARTH * math.cos(deg2rad(lat0))
    return east, north


def local_m_to_lonlat(east: float, north: float, lon0: float, lat0: float) -> Tuple[float, float]:
    dlat = north / R_EARTH
    lat = lat0 + dlat * 180.0 / math.pi
    dlon = east / (R_EARTH * math.cos(deg2rad(lat0)))
    lon = lon0 + dlon * 180.0 / math.pi
    return lon, lat
