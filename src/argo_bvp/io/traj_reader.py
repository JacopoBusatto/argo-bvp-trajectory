"""TRAJ reader helpers for extracting GPS fixes and pressures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

_NETCDF_ENGINE = "h5netcdf"


def read_traj(path: str | Path) -> xr.Dataset:
    """Read a TRAJ NetCDF file as an in-memory xarray.Dataset."""
    path = Path(path)
    try:
        with xr.open_dataset(path, engine=_NETCDF_ENGINE, decode_times=False) as ds:
            return ds.load()
    except Exception as exc:
        try:
            with xr.open_dataset(path, decode_times=False) as ds:
                return ds.load()
        except Exception as exc2:
            raise RuntimeError(f"Failed to open TRAJ dataset: {path}") from exc2


def extract_traj_positions(
    ds: xr.Dataset,
    var_map: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract juld/pres/lat/lon/QC arrays from a TRAJ dataset."""
    mapping = {
        "juld": "JULD",
        "pres": "PRES",
        "lat": "LATITUDE",
        "lon": "LONGITUDE",
        "position_qc": "POSITION_QC",
    }
    if var_map:
        mapping.update(var_map)

    juld = _as_1d_float(ds[mapping["juld"]].values, mapping["juld"])
    pres = _as_1d_float(ds[mapping["pres"]].values, mapping["pres"])
    lat = _as_1d_float(ds[mapping["lat"]].values, mapping["lat"])
    lon = _as_1d_float(ds[mapping["lon"]].values, mapping["lon"])
    position_qc = _normalize_qc_array(ds[mapping["position_qc"]].values)

    _require_same_length(juld, pres, lat, lon, position_qc)

    return {
        "juld": juld,
        "pres": pres,
        "lat": lat,
        "lon": lon,
        "position_qc": position_qc,
    }


def _as_1d_float(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return np.asarray(arr, dtype=float)


def _normalize_qc_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim == 2 and arr.shape[1] > 1:
        joined = []
        for row in arr:
            joined.append(_normalize_qc_value(row))
        return np.asarray(joined, dtype="U8")
    if arr.ndim != 1:
        arr = arr.reshape(arr.shape[0], -1)
        return _normalize_qc_array(arr)
    out = [_normalize_qc_value(value) for value in arr.tolist()]
    return np.asarray(out, dtype="U8")


def _normalize_qc_value(value: object) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8").strip()
    if isinstance(value, np.ndarray):
        parts = []
        for item in value.tolist():
            parts.append(_normalize_qc_value(item))
        return "".join(parts).strip()
    return str(value).strip()


def _require_same_length(*arrays: np.ndarray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("TRAJ arrays must have the same length")


__all__ = ["read_traj", "extract_traj_positions"]
