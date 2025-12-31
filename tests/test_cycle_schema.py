"""Tests for the cycle file schema helpers and NetCDF I/O."""

import numpy as np
import pytest
import xarray as xr

from argo_bvp.cycle_schema import ANCHOR_LABELS, VEC_LABELS, make_empty_cycle_dataset, validate_cycle_dataset
from argo_bvp.io.cycle_io import read_cycle_netcdf, write_cycle_netcdf


def test_make_empty_cycle_dataset_schema() -> None:
    ds = make_empty_cycle_dataset(n_obs=10, float_id="TEST", cycle_number=1)

    assert ds.sizes["obs"] == 10
    assert ds.sizes["anchor"] == 2
    assert ds.sizes["vec"] == 3
    assert tuple(ds.coords["anchor"].values.tolist()) == ANCHOR_LABELS
    assert tuple(ds.coords["vec"].values.tolist()) == VEC_LABELS

    required_vars = [
        "anchor_juld",
        "anchor_lat",
        "anchor_lon",
        "anchor_position_qc",
        "juld",
        "pres",
        "z",
        "lin_acc_count",
        "lin_acc",
        "ang_rate_count",
        "ang_rate",
        "mag_field_count",
        "mag_field",
        "phase",
    ]
    for name in required_vars:
        assert name in ds.data_vars

    assert ds["anchor_juld"].dims == ("anchor",)
    assert ds["lin_acc"].dims == ("obs", "vec")
    assert ds["phase"].dims == ("obs",)

    required_global_attrs = [
        "schema_name",
        "schema_version",
        "cycle_number",
        "float_id",
        "time_origin_juld",
        "time_units",
        "geodesy",
        "lon_convention",
    ]
    for name in required_global_attrs:
        assert name in ds.attrs

    validate_cycle_dataset(ds, strict=True)


def test_cycle_netcdf_round_trip(tmp_path) -> None:
    ds = make_empty_cycle_dataset(n_obs=6, float_id="TEST", cycle_number=2)
    path = tmp_path / "cycle.nc"

    write_cycle_netcdf(ds, path)
    with xr.open_dataset(path) as ds_raw:
        print("anchor_juld attrs:", ds_raw["anchor_juld"].attrs)
        print("anchor_juld encoding:", ds_raw["anchor_juld"].encoding)
        print("juld attrs:", ds_raw["juld"].attrs)
        print("juld encoding:", ds_raw["juld"].encoding)
    ds_read = read_cycle_netcdf(path)

    assert ds_read.sizes == ds.sizes
    assert set(ds_read.data_vars) == set(ds.data_vars)
    for name in ds.data_vars:
        assert ds_read[name].dims == ds[name].dims


def test_validate_cycle_dataset_failures() -> None:
    ds = make_empty_cycle_dataset(n_obs=5, float_id="TEST", cycle_number=3)

    with pytest.raises(ValueError):
        validate_cycle_dataset(ds.drop_vars("lin_acc"))

    with pytest.raises(ValueError):
        validate_cycle_dataset(ds.isel(anchor=slice(0, 1)))

    with pytest.raises(ValueError):
        validate_cycle_dataset(ds.isel(vec=slice(0, 2)))

    bad_t = ds.copy()
    t = np.asarray(bad_t.coords["t"].values).copy()
    t[3] = t[2] - 1.0
    bad_t = bad_t.assign_coords(t=("obs", t))
    with pytest.raises(ValueError):
        validate_cycle_dataset(bad_t)
