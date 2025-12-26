"""
Preprocessing utilities for Coriolis Argo trajectory auxiliary files (IMU, etc.).
"""

from .config import PreprocessConfig, load_config
from .io_coriolis import open_aux, open_traj
from .products import build_preprocessed_dataset
from .cycles import build_cycle_products
from .writers import write_netcdf, write_parquet

__all__ = [
    "PreprocessConfig",
    "load_config",
    "open_aux",
    "open_traj",
    "build_preprocessed_dataset",
    "build_cycle_products",
    "write_netcdf",
    "write_parquet",
]
