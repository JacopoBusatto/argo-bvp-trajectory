from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import numpy as np
import xarray as xr


AUX_REQUIRED_VARS = [
    "JULD",
    "PRES",
    "CYCLE_NUMBER",
    "MEASUREMENT_CODE",  # <-- aggiunto: chiave per fasi/keypoints

    "LINEAR_ACCELERATION_COUNT_X",
    "LINEAR_ACCELERATION_COUNT_Y",
    "LINEAR_ACCELERATION_COUNT_Z",

    "ANGULAR_RATE_COUNT_X",
    "ANGULAR_RATE_COUNT_Y",
    "ANGULAR_RATE_COUNT_Z",

    "MAGNETIC_FIELD_COUNT_X",
    "MAGNETIC_FIELD_COUNT_Y",
    "MAGNETIC_FIELD_COUNT_Z",
]


def open_aux(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path)


def open_traj(path: str | Path) -> xr.Dataset:
    return xr.open_dataset(path)


def _as_1d(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values).reshape(-1)


def extract_aux_minimal(ds_aux: xr.Dataset, vars: Optional[Sequence[str]] = None) -> dict:
    """
    Extract minimal aux arrays as 1D numpy arrays.
    """
    vars = list(vars) if vars is not None else AUX_REQUIRED_VARS
    out = {}
    for v in vars:
        if v not in ds_aux:
            raise KeyError(f"Missing variable in aux: {v}")
        out[v] = _as_1d(ds_aux[v])
        out[f"{v}__attrs"] = dict(ds_aux[v].attrs)
    return out


def build_valid_mask(*arrays: np.ndarray) -> np.ndarray:
    """
    Valid mask for numeric arrays: finite for all numeric inputs.
    Non-numeric arrays do not affect the mask.
    """
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        a = np.asarray(a).reshape(-1)
        if np.issubdtype(a.dtype, np.number):
            mask &= np.isfinite(a)
    return mask