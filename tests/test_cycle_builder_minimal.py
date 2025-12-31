"""Tests for minimal TRAJ+AUX preprocessing to cycle schema."""

import numpy as np
import xarray as xr

from argo_bvp.cycle_schema import validate_cycle_dataset
from argo_bvp.preprocess.cycle_builder import build_cycle_from_traj_aux


def test_build_cycle_from_traj_aux_minimal() -> None:
    base_juld = 25000.0
    traj_juld = base_juld + np.array([0, 1, 2, 3, 4, 5], dtype=float) / 86400.0
    traj_pres = np.array([0, 0, 50, 50, 0, 0], dtype=float)
    traj_lat = np.array([43.0, 43.1, 43.2, 43.3, 43.4, 43.5], dtype=float)
    traj_lon = np.array([9.0, 9.1, 9.2, 9.3, 9.4, 9.5], dtype=float)
    traj_qc = np.array(["1", "4", "1", "1", "4", "2"], dtype="U1")

    traj_ds = xr.Dataset(
        data_vars={
            "JULD": ("obs", traj_juld),
            "PRES": ("obs", traj_pres),
            "LATITUDE": ("obs", traj_lat),
            "LONGITUDE": ("obs", traj_lon),
            "POSITION_QC": ("obs", traj_qc),
        }
    )

    aux_juld = base_juld + np.array([0, 1, 2, 4, 5], dtype=float) / 86400.0
    aux_pres = np.array([0, 5, 20, 30, 0], dtype=float)
    lin_acc = np.ones((aux_juld.size, 3), dtype=float)
    ang_rate = np.full((aux_juld.size, 3), 2.0, dtype=float)
    mag = np.full((aux_juld.size, 3), 3.0, dtype=float)

    aux_ds = xr.Dataset(
        data_vars={
            "JULD": ("obs", aux_juld),
            "PRES": ("obs", aux_pres),
            "LIN_ACC_COUNT": (("obs", "vec"), lin_acc),
            "ANG_RATE_COUNT": (("obs", "vec"), ang_rate),
            "MAG_FIELD_COUNT": (("obs", "vec"), mag),
        },
        coords={"vec": np.array(["x", "y", "z"], dtype="U1")},
    )

    ds = build_cycle_from_traj_aux(
        traj_ds,
        aux_ds,
        window_index=0,
        config={"float_id": "TEST", "cycle_number": 7},
    )

    validate_cycle_dataset(ds, strict=True)

    assert ds.sizes["obs"] == aux_juld.size
    assert ds["anchor_juld"].values[0] == traj_juld[0]
    assert ds["anchor_juld"].values[1] == traj_juld[5]
    assert np.allclose(ds["lin_acc_count"].values, lin_acc)
    assert np.allclose(ds["ang_rate_count"].values, ang_rate)
    assert np.allclose(ds["mag_field_count"].values, mag)
