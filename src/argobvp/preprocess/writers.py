from __future__ import annotations

from pathlib import Path
import xarray as xr


def write_netcdf(ds: xr.Dataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def write_parquet(ds: xr.Dataset, path: str | Path) -> None:
    """
    Requires pandas + pyarrow.
    Writes a flat table (one row per obs).
    """
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = ds.to_dataframe().reset_index()
    df.to_parquet(path, index=False)
