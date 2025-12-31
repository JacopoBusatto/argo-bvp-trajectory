"""Cycle file NetCDF schema (v1) helpers and validation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import xarray as xr

ANCHOR_LABELS = ("start", "end")
VEC_LABELS = ("x", "y", "z")
PHASE_MEANING: Mapping[int, str] = {
    0: "unknown",
    1: "surface",
    2: "descent",
    3: "park",
    4: "ascent",
}

REQUIRED_GLOBAL_ATTRS = (
    "schema_name",
    "schema_version",
    "cycle_number",
    "float_id",
    "time_origin_juld",
    "time_units",
    "geodesy",
    "lon_convention",
)

REQUIRED_VARS = (
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
)

EXPECTED_UNITS = {
    "anchor_juld": "days since 1950-01-01 00:00:00 UTC",
    "anchor_lat": "degree_north",
    "anchor_lon": "degree_east",
    "juld": "days since 1950-01-01 00:00:00 UTC",
    "pres": "dbar",
    "z": "m",
    "lin_acc_count": "count",
    "lin_acc": "m s-2",
    "ang_rate_count": "count",
    "ang_rate": "rad s-1",
    "mag_field_count": "count",
    "mag_field": "uT",
}


def make_empty_cycle_dataset(
    n_obs: int,
    float_id: str | int,
    cycle_number: int,
) -> xr.Dataset:
    """Create an empty cycle dataset that conforms to schema v1."""
    if n_obs < 0:
        raise ValueError("n_obs must be >= 0")

    t = np.arange(n_obs, dtype="float64")
    anchor_juld = np.full((2,), np.nan, dtype="float64")
    anchor_lat = np.full((2,), np.nan, dtype="float64")
    anchor_lon = np.full((2,), np.nan, dtype="float64")
    anchor_position_qc = np.full((2,), 0, dtype="int8")

    juld = np.full((n_obs,), np.nan, dtype="float64")
    pres = np.full((n_obs,), np.nan, dtype="float64")
    z = np.full((n_obs,), np.nan, dtype="float64")

    lin_acc_count = np.full((n_obs, 3), np.nan, dtype="float64")
    lin_acc = np.full((n_obs, 3), np.nan, dtype="float64")
    ang_rate_count = np.full((n_obs, 3), np.nan, dtype="float64")
    ang_rate = np.full((n_obs, 3), np.nan, dtype="float64")
    mag_field_count = np.full((n_obs, 3), np.nan, dtype="float64")
    mag_field = np.full((n_obs, 3), np.nan, dtype="float64")

    phase = np.full((n_obs,), 0, dtype="int8")

    ds = xr.Dataset(
        data_vars={
            "anchor_juld": ("anchor", anchor_juld, {"units": EXPECTED_UNITS["anchor_juld"]}),
            "anchor_lat": ("anchor", anchor_lat, {"units": EXPECTED_UNITS["anchor_lat"]}),
            "anchor_lon": ("anchor", anchor_lon, {"units": EXPECTED_UNITS["anchor_lon"]}),
            "anchor_position_qc": ("anchor", anchor_position_qc),
            "juld": ("obs", juld, {"units": EXPECTED_UNITS["juld"]}),
            "pres": ("obs", pres, {"units": EXPECTED_UNITS["pres"]}),
            "z": ("obs", z, {"units": EXPECTED_UNITS["z"], "positive": "down"}),
            "lin_acc_count": (
                ("obs", "vec"),
                lin_acc_count,
                {"units": EXPECTED_UNITS["lin_acc_count"]},
            ),
            "lin_acc": (("obs", "vec"), lin_acc, {"units": EXPECTED_UNITS["lin_acc"]}),
            "ang_rate_count": (
                ("obs", "vec"),
                ang_rate_count,
                {"units": EXPECTED_UNITS["ang_rate_count"]},
            ),
            "ang_rate": (("obs", "vec"), ang_rate, {"units": EXPECTED_UNITS["ang_rate"]}),
            "mag_field_count": (
                ("obs", "vec"),
                mag_field_count,
                {"units": EXPECTED_UNITS["mag_field_count"]},
            ),
            "mag_field": (("obs", "vec"), mag_field, {"units": EXPECTED_UNITS["mag_field"]}),
            "phase": (
                "obs",
                phase,
                {"phase_meaning": "0=unknown,1=surface,2=descent,3=park,4=ascent"},
            ),
            "lat0": ((), np.nan, {"units": "degree_north"}),
            "lon0": ((), np.nan, {"units": "degree_east"}),
            "g": ((), np.nan, {"units": "m s-2"}),
        },
        coords={
            "t": ("obs", t),
            "anchor": np.array(ANCHOR_LABELS, dtype="U5"),
            "vec": np.array(VEC_LABELS, dtype="U1"),
        },
        attrs={
            "schema_name": "argo_bvp_cycle",
            "schema_version": "1.0",
            "cycle_number": int(cycle_number),
            "float_id": float_id,
            "time_origin_juld": float("nan"),
            "time_units": "s since cycle start (anchor start JULD)",
            "geodesy": "WGS84",
            "lon_convention": "[-180,180)",
        },
    )
    return ds


def validate_cycle_dataset(ds: xr.Dataset, strict: bool = True) -> None:
    """Validate a cycle dataset against schema v1."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    def _normalize_units(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8").strip()
        return str(value).strip()

    def _get_units(var: xr.DataArray) -> str | None:
        units = _normalize_units(var.attrs.get("units"))
        if units is None:
            units = _normalize_units(var.encoding.get("units"))
        return units

    def _normalize_juld_units(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized

    for dim in ("obs", "anchor", "vec"):
        if dim not in ds.dims:
            raise ValueError(f"Missing dimension: {dim}")

    if ds.sizes["anchor"] != 2:
        raise ValueError("anchor dimension must have size 2")
    if ds.sizes["vec"] != 3:
        raise ValueError("vec dimension must have size 3")

    for coord in ("t", "anchor", "vec"):
        if coord not in ds.coords:
            raise ValueError(f"Missing coordinate: {coord}")

    anchor_values = ds.coords["anchor"].values
    anchor_labels = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in anchor_values
    )
    if anchor_labels != ANCHOR_LABELS:
        raise ValueError(f"anchor coordinate labels must be {ANCHOR_LABELS}")

    vec_values = ds.coords["vec"].values
    vec_labels = tuple(
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in vec_values
    )
    if vec_labels != VEC_LABELS:
        raise ValueError(f"vec coordinate labels must be {VEC_LABELS}")

    missing_vars = [var for var in REQUIRED_VARS if var not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")

    expected_dims = {
        "anchor_juld": ("anchor",),
        "anchor_lat": ("anchor",),
        "anchor_lon": ("anchor",),
        "anchor_position_qc": ("anchor",),
        "juld": ("obs",),
        "pres": ("obs",),
        "z": ("obs",),
        "lin_acc_count": ("obs", "vec"),
        "lin_acc": ("obs", "vec"),
        "ang_rate_count": ("obs", "vec"),
        "ang_rate": ("obs", "vec"),
        "mag_field_count": ("obs", "vec"),
        "mag_field": ("obs", "vec"),
        "phase": ("obs",),
    }

    for var_name, dims in expected_dims.items():
        if tuple(ds[var_name].dims) != dims:
            raise ValueError(f"{var_name} dims must be {dims}")

    t = np.asarray(ds.coords["t"].values, dtype="float64")
    if t.ndim != 1:
        raise ValueError("t coordinate must be 1D")
    if t.size and not np.all(np.isfinite(t)):
        raise ValueError("t coordinate must be finite")
    if t.size > 1 and np.any(np.diff(t) < 0):
        raise ValueError("t coordinate must be monotonic non-decreasing")

    if strict:
        for attr in REQUIRED_GLOBAL_ATTRS:
            if attr not in ds.attrs:
                raise ValueError(f"Missing global attribute: {attr}")

        if ds.attrs.get("schema_name") != "argo_bvp_cycle":
            raise ValueError("schema_name must be 'argo_bvp_cycle'")
        if ds.attrs.get("schema_version") != "1.0":
            raise ValueError("schema_version must be '1.0'")
        if ds.attrs.get("time_units") != "s since cycle start (anchor start JULD)":
            raise ValueError("time_units must be 's since cycle start (anchor start JULD)'")
        if ds.attrs.get("geodesy") != "WGS84":
            raise ValueError("geodesy must be 'WGS84'")
        if ds.attrs.get("lon_convention") != "[-180,180)":
            raise ValueError("lon_convention must be '[-180,180)'")

        for var_name, expected in EXPECTED_UNITS.items():
            units = _get_units(ds[var_name])
            if var_name in {"anchor_juld", "juld"}:
                normalized = _normalize_juld_units(units)
                if normalized is None or not normalized.startswith("days since 1950-01-01"):
                    raise ValueError(f"{var_name} units invalid: {units!r}")
            elif units != expected:
                raise ValueError(f"{var_name} units must be '{expected}', got {units!r}")

        if "phase_meaning" not in ds["phase"].attrs:
            raise ValueError("phase variable missing phase_meaning attribute")


__all__ = [
    "ANCHOR_LABELS",
    "VEC_LABELS",
    "PHASE_MEANING",
    "make_empty_cycle_dataset",
    "validate_cycle_dataset",
]
