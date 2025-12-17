"""
Preprocessing utilities for Coriolis Argo trajectory auxiliary files (IMU, etc.).
"""

from .config import PreprocessConfig, load_config
from .io_coriolis import open_aux, open_traj
from .products import build_preprocessed_dataset
from .writers import write_netcdf, write_parquet

__all__ = [
    "PreprocessConfig",
    "load_config",
    "open_aux",
    "open_traj",
    "build_preprocessed_dataset",
    "write_netcdf",
    "write_parquet",
]
