"""Tests for surface window detection and anchor selection."""

import numpy as np
import xarray as xr

from argo_bvp.io.traj_reader import extract_traj_positions
from argo_bvp.preprocess.surface_windows import find_surface_windows, select_anchor_points


def test_anchor_selection_with_qc() -> None:
    base_juld = 25000.0
    juld = base_juld + np.array([0, 1, 2, 3, 4, 5], dtype=float) / 86400.0
    pres = np.array([0, 0, 50, 50, 0, 0], dtype=float)
    lat = np.array([43.0, 43.1, 43.2, 43.3, 43.4, 43.5], dtype=float)
    lon = np.array([9.0, 9.1, 9.2, 9.3, 9.4, 9.5], dtype=float)
    qc = np.array(["1", "4", "1", "1", "4", "2"], dtype="U1")

    ds = xr.Dataset(
        data_vars={
            "JULD": ("obs", juld),
            "PRES": ("obs", pres),
            "LATITUDE": ("obs", lat),
            "LONGITUDE": ("obs", lon),
            "POSITION_QC": ("obs", qc),
        }
    )

    traj = extract_traj_positions(ds)
    windows = find_surface_windows(traj["juld"], traj["pres"], p_surface=5.0)

    assert windows == [(0, 2), (4, 6)]

    start_anchor, end_anchor = select_anchor_points(
        traj["juld"],
        traj["pres"],
        traj["lat"],
        traj["lon"],
        traj["position_qc"],
        windows=windows,
        window_index=0,
        qc_ok={"1", "2", "5"},
    )

    assert start_anchor["index"] == 0
    assert end_anchor["index"] == 5
    assert start_anchor["position_qc"] == "1"
    assert end_anchor["position_qc"] == "2"
