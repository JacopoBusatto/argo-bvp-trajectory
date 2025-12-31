"""Tests for local ENU roundtrip conversions."""

import numpy as np

from argo_bvp.geodesy.local_enu import enu_to_latlon, latlon_to_enu_m


def test_local_enu_roundtrip() -> None:
    lat0 = 40.0
    lon0 = 15.0

    east = np.array([0.0, 1000.0, 5000.0, -7500.0, 10000.0])
    north = np.array([0.0, -2000.0, 6000.0, 8000.0, -10000.0])

    lat, lon = enu_to_latlon(east, north, lat0, lon0)
    east2, north2 = latlon_to_enu_m(lat, lon, lat0, lon0)

    diff = np.sqrt((east2 - east) ** 2 + (north2 - north) ** 2)
    assert np.all(diff <= 0.5)
