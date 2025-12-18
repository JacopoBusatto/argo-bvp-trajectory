"""
Preprocessing utilities for Coriolis Argo trajectory auxiliary files (IMU, etc.).
"""

from .config import PreprocessConfig, load_config
from .io_coriolis import open_aux, open_traj
from .products import build_preprocessed_dataset
from .cycles import build_cycle_products
from .bvp_ready import build_bvp_ready_dataset, BVPReadyConfig
from .writers import write_netcdf, write_parquet

__all__ = [
    "PreprocessConfig",
    "load_config",
    "open_aux",
    "open_traj",
    "build_preprocessed_dataset",
    "build_cycle_products",
    "build_bvp_ready_dataset",
    "BVPReadyConfig",
    "write_netcdf",
    "write_parquet",
]
