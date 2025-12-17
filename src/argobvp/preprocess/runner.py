from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import xarray as xr

from .config import load_config
from .io_coriolis import open_aux, open_traj
from .products import build_preprocessed_dataset
from .cycles import build_cycle_products
from .writers import write_netcdf, write_parquet


@dataclass(frozen=True)
class PreprocessOutputs:
    ds_continuous: xr.Dataset
    ds_cycles: xr.Dataset
    ds_segments: xr.Dataset


def run_preprocess(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    write_parquet_products: bool = True,
    open_traj_file: bool = False,
) -> PreprocessOutputs:
    """
    End-to-end preprocessing runner.

    Parameters
    ----------
    config_path : path to YAML config.
    out_dir : output directory.
    write_parquet_products : if True, write cycles/segments parquet in addition to NetCDF.
    open_traj_file : if True, open traj file (currently for sanity checks only).

    Outputs
    -------
    - preprocessed_imu.nc (continuous, obs-dimension)
    - cycles.nc (one row per cycle)
    - segments.nc (one row per contiguous segment inside cycles)
    - cycles.parquet / segments.parquet (optional)
    """
    cfg = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Open AUX (mandatory) ---
    ds_aux = open_aux(cfg.paths.aux)

    # --- Optional: open TRAJ (not used yet) ---
    ds_traj = None
    if open_traj_file:
        ds_traj = open_traj(cfg.paths.traj)

    # --- Build continuous IMU product ---
    ds_cont = build_preprocessed_dataset(ds_aux, cfg)

    # --- Build cycle/segment products from continuous (needs measurement_code) ---
    ds_cycles, ds_segments = build_cycle_products(ds_cont, cfg)

    # --- Write outputs ---
    p_cont = out_dir / f"{cfg.platform}_preprocessed_imu.nc"
    p_cycles_nc = out_dir / f"{cfg.platform}_cycles.nc"
    p_segments_nc = out_dir / f"{cfg.platform}_segments.nc"

    write_netcdf(ds_cont, p_cont)
    write_netcdf(ds_cycles, p_cycles_nc)
    write_netcdf(ds_segments, p_segments_nc)

    if write_parquet_products:
        p_cycles_pq = out_dir / f"{cfg.platform}_cycles.parquet"
        p_segments_pq = out_dir / f"{cfg.platform}_segments.parquet"
        write_parquet(ds_cycles, p_cycles_pq)
        write_parquet(ds_segments, p_segments_pq)

    # Close opened datasets if they are lazy-backed by files
    try:
        ds_aux.close()
    except Exception:
        pass
    if ds_traj is not None:
        try:
            ds_traj.close()
        except Exception:
            pass

    return PreprocessOutputs(
        ds_continuous=ds_cont,
        ds_cycles=ds_cycles,
        ds_segments=ds_segments,
    )