"""Convenience entrypoint for preprocessing to cycle files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import xarray as xr

from .io.cycle_io import write_cycle_netcdf
from .preprocess.cycle_builder import build_cycle_from_traj_aux


def build_cycle_file(
    traj_path: str | Path | xr.Dataset,
    aux_path: str | Path | xr.Dataset,
    out_path: str | Path,
    window_index: int = 0,
    instrument: object | None = "synth_v1",
    config: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Build a cycle dataset and write it to disk."""
    ds = build_cycle_from_traj_aux(
        traj_path=traj_path,
        aux_path=aux_path,
        window_index=window_index,
        instrument=instrument,
        config=config,
    )
    write_cycle_netcdf(ds, out_path)
    return ds


def derive_base_from_traj_path(traj_path: Path) -> str:
    """Derive the output base name from a TRAJ filename."""
    stem = Path(traj_path).stem
    suffix = "_TRAJ"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


__all__ = ["build_cycle_file", "derive_base_from_traj_path"]
