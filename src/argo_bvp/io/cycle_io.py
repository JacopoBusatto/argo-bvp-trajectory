"""NetCDF read/write helpers for the cycle file schema."""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from ..cycle_schema import validate_cycle_dataset


_NETCDF_ENGINE = "h5netcdf"
_JULD_UNITS = "days since 1950-01-01 00:00:00"


def write_cycle_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    """Validate and write a cycle dataset to NetCDF.

    Notes
    -----
    We force the 'h5netcdf' engine to avoid the compiled netCDF4 backend on Windows,
    which can emit ABI warnings depending on binary wheel compatibility.
    """
    validate_cycle_dataset(ds, strict=True)

    ds = ds.copy(deep=False)
    ds["anchor_juld"].encoding["units"] = _JULD_UNITS
    ds["juld"].encoding["units"] = _JULD_UNITS

    ds.to_netcdf(Path(path), engine=_NETCDF_ENGINE)


def read_cycle_netcdf(path: str | Path) -> xr.Dataset:
    """Read and validate a cycle dataset from NetCDF.

    Notes
    -----
    We force the 'h5netcdf' engine for consistency with write_cycle_netcdf.
    """
    path = Path(path)
    ds = xr.open_dataset(path, engine=_NETCDF_ENGINE).load()
    validate_cycle_dataset(ds, strict=True)
    return ds


__all__ = ["write_cycle_netcdf", "read_cycle_netcdf"]
