"""AUX reader helpers for extracting IMU arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

_NETCDF_ENGINE = "h5netcdf"


def read_aux(path: str | Path) -> xr.Dataset:
    """Read an AUX NetCDF file as an in-memory xarray.Dataset."""
    path = Path(path)
    try:
        with xr.open_dataset(path, engine=_NETCDF_ENGINE, decode_times=False) as ds:
            return ds.load()
    except Exception as exc:
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                return ds.load()
        except Exception as exc2:
            raise RuntimeError(f"Failed to open AUX dataset: {path}") from exc2


def extract_imu_arrays(
    ds: xr.Dataset,
    var_map: Mapping[str, str | list[str] | tuple[str, str, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Extract IMU arrays from an AUX dataset."""
    mapping: dict[str, str | list[str] | tuple[str, str, str]] = {
        "juld": "JULD",
        "pres": "PRES",
        "lin_acc_count": "LIN_ACC_COUNT",
        "ang_rate_count": "ANG_RATE_COUNT",
        "mag_field_count": "MAG_FIELD_COUNT",
    }
    if var_map:
        mapping.update(var_map)

    juld = _as_1d_float(ds[mapping["juld"]].values, mapping["juld"])
    pres = _as_1d_float(ds[mapping["pres"]].values, mapping["pres"])

    lin_acc_count = _extract_vector_array(ds, mapping["lin_acc_count"], "lin_acc_count")
    ang_rate_count = _extract_vector_array(ds, mapping["ang_rate_count"], "ang_rate_count")
    mag_field_count = _extract_vector_array(ds, mapping["mag_field_count"], "mag_field_count")

    _require_same_length(juld, pres, lin_acc_count, ang_rate_count, mag_field_count)

    return {
        "juld": juld,
        "pres": pres,
        "lin_acc_count": lin_acc_count,
        "ang_rate_count": ang_rate_count,
        "mag_field_count": mag_field_count,
    }


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.asarray(arr, dtype=float)


def _extract_vector_array(
    ds: xr.Dataset,
    spec: str | list[str] | tuple[str, str, str],
    name: str,
) -> np.ndarray:
    if isinstance(spec, (list, tuple)):
        if len(spec) != 3:
            raise ValueError(f"{name} must have 3 component variables")
        parts = [_as_1d_float(ds[var].values, str(var)) for var in spec]
        return np.stack(parts, axis=1)

    arr = np.asarray(ds[spec].values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with 3 components")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"{name} must have a component dimension of size 3")


def _require_same_length(*arrays: np.ndarray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("AUX arrays must have the same length")


__all__ = ["read_aux", "extract_imu_arrays"]
