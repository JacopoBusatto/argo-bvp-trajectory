"""Roundtrip checks for synthetic NetCDF outputs."""

import xarray as xr

from argo_bvp.instruments import INSTRUMENTS
from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_synthetic_raw import generate_synthetic_raw


def test_netcdf_roundtrip_units_attrs(tmp_path) -> None:
    outputs = generate_synthetic_raw(tmp_path, DEFAULT_EXPERIMENT, INSTRUMENTS["synth_v1"])

    with xr.open_dataset(outputs["truth_nc"], engine="h5netcdf") as ds_truth:
        assert _get_units(ds_truth["t"]) == "s"
        assert _get_units(ds_truth["x"]) == "m"
        assert _get_units(ds_truth["ax"]) == "m s-2"
        assert "start_juld" in ds_truth.attrs
        assert "park_depth_m" in ds_truth.attrs

    with xr.open_dataset(outputs["traj_nc"], engine="h5netcdf") as ds_traj:
        assert _get_units(ds_traj["JULD"]) in {"days", "days since 1950-01-01 00:00:00"}
        assert _get_units(ds_traj["PRES"]) == "decibar"
        assert _get_units(ds_traj["LATITUDE"]) == "degree_north"

    with xr.open_dataset(outputs["aux_nc"], engine="h5netcdf") as ds_aux:
        assert _get_units(ds_aux["JULD"]) in {"days", "days since 1950-01-01 00:00:00"}
        assert "lsb_to_ms2" in ds_aux.attrs
        assert "gyro_lsb_to_rads" in ds_aux.attrs


def _get_units(var: xr.DataArray) -> str | None:
    units = var.attrs.get("units")
    if units:
        return units
    return var.encoding.get("units")
