from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Optional

import xarray as xr

from .config import load_config
from .io_coriolis import open_aux, open_traj
from .products import build_preprocessed_dataset
from .cycles import build_cycle_products
from .writers import write_netcdf, write_parquet
from .surface_fixes import add_surface_position_from_traj, SurfaceFixConfig

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

    # --- Build cycle/segment products from continuous ---
    ds_cycles, ds_segments = build_cycle_products(ds_cont, cfg)

    # --- add surface position constraints from traj fixes ---
    ds_traj = open_traj(cfg.paths.traj)  # già esiste nel runner se vuoi aprirlo sempre
    ds_cycles = add_surface_position_from_traj(
        ds_cycles,
        ds_traj,
        cfg=SurfaceFixConfig(
            max_gap_seconds=3*24*3600,
            max_abs_dt_nearest_seconds=6*3600,
        ),
    )
    
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

    # close datasets
    try:
        ds_aux.close()
    except Exception:
        pass
    if ds_traj is not None:
        try:
            ds_traj.close()
        except Exception:
            pass

    return PreprocessOutputs(ds_continuous=ds_cont, ds_cycles=ds_cycles, ds_segments=ds_segments)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ArgoBVP preprocess runner (AUX -> continuous + cycles + segments).")
    p.add_argument("--config", required=True, help="Path to YAML config (e.g. configs/4903848.yml)")
    p.add_argument("--out", required=True, help="Output directory (e.g. outputs/preprocess)")
    p.add_argument("--no-parquet", action="store_true", help="Do not write parquet products.")
    p.add_argument("--open-traj", action="store_true", help="Also open traj file (sanity check only).")
    return p


def main():
    args = _build_parser().parse_args()
    run_preprocess(
        config_path=args.config,
        out_dir=args.out,
        write_parquet_products=not args.no_parquet,
        open_traj_file=args.open_traj,
    )
    print("OK: wrote outputs to", args.out)


if __name__ == "__main__":
    main()