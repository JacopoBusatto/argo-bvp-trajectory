"""Build synthetic TRAJ datasets from TRUTH."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import xarray as xr


def build_traj_from_truth(truth_ds: xr.Dataset) -> xr.Dataset:
    """Create a Coriolis-like TRAJ dataset using TRUTH inputs."""
    if not isinstance(truth_ds, xr.Dataset):
        raise TypeError("truth_ds must be an xarray.Dataset")

    t = _as_1d_float(truth_ds, "t")
    n = t.size
    if n == 0:
        raise ValueError("truth_ds has no observations")

    start_juld = _resolve_start_juld(truth_ds)
    juld = start_juld + t / 86400.0

    pres = _get_pres(truth_ds, n)

    phase = _as_1d_str(truth_ds, "phase")
    surface_segments = _surface_segments(phase, "surface")
    if len(surface_segments) < 2:
        raise ValueError("truth_ds must contain surface1 and surface2 segments")

    surface1 = surface_segments[0]
    surface2 = surface_segments[-1]

    lat_truth = _as_1d_float(truth_ds, "lat")
    lon_truth = _as_1d_float(truth_ds, "lon")

    lat = np.full((n,), np.nan, dtype=float)
    lon = np.full((n,), np.nan, dtype=float)
    position_qc = np.full((n,), "9", dtype="U1")

    _fill_surface(lat, lon, position_qc, lat_truth, lon_truth, surface1, "1")
    _fill_surface(lat, lon, position_qc, lat_truth, lon_truth, surface2, "1")

    juld_qc = np.full((n,), "1", dtype="U1")
    measurement_code = _phase_to_measurement_code(phase)

    ds = xr.Dataset(
        data_vars={
            "JULD": ("N_MEASUREMENT", juld, {"units": "days"}),
            "JULD_QC": ("N_MEASUREMENT", juld_qc),
            "PRES": ("N_MEASUREMENT", pres, {"units": "decibar"}),
            "LATITUDE": ("N_MEASUREMENT", lat, {"units": "degree_north"}),
            "LONGITUDE": ("N_MEASUREMENT", lon, {"units": "degree_east"}),
            "POSITION_QC": ("N_MEASUREMENT", position_qc),
            "MEASUREMENT_CODE": ("N_MEASUREMENT", measurement_code),
        },
        attrs={
            "source": "synthetic TRUTH",
        },
    )
    return ds


def _as_1d_float(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing {name} in truth dataset")
    arr = np.asarray(ds[name].values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _as_1d_str(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing {name} in truth dataset")
    arr = np.asarray(ds[name].values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    out = []
    for value in arr.tolist():
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8").strip())
        else:
            out.append(str(value).strip())
    return np.asarray(out, dtype="U16")


def _resolve_start_juld(ds: xr.Dataset) -> float:
    if "start_juld" in ds.attrs:
        return float(ds.attrs["start_juld"])
    if "anchor_juld" in ds:
        anchor = np.asarray(ds["anchor_juld"].values, dtype=float)
        if anchor.size:
            return float(anchor.flat[0])
    raise KeyError("start_juld not found in TRUTH dataset")


def _get_pres(ds: xr.Dataset, n: int) -> np.ndarray:
    if "pres" in ds:
        pres = np.asarray(ds["pres"].values, dtype=float)
    elif "z" in ds:
        z = np.asarray(ds["z"].values, dtype=float)
        pres = np.maximum(-z, 0.0)
    else:
        raise KeyError("TRUTH dataset missing pres or z")
    if pres.shape[0] != n:
        raise ValueError("pres must match t length")
    return pres.astype("float32", copy=False)


def _surface_segments(phase: np.ndarray, surface_label: str) -> list[tuple[int, int]]:
    mask = phase == surface_label
    indices = np.where(mask)[0]
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx != prev + 1:
            segments.append((start, prev + 1))
            start = idx
        prev = idx
    segments.append((start, prev + 1))
    return segments


def _fill_surface(
    lat: np.ndarray,
    lon: np.ndarray,
    qc: np.ndarray,
    lat_truth: np.ndarray,
    lon_truth: np.ndarray,
    segment: tuple[int, int],
    qc_value: str,
) -> None:
    start, end = segment
    lat[start:end] = lat_truth[start:end]
    lon[start:end] = lon_truth[start:end]
    qc[start:end] = qc_value


def _phase_to_measurement_code(phase: Iterable[str]) -> np.ndarray:
    mapping = {
        "surface": 1.0,
        "descent": 2.0,
        "park": 3.0,
        "ascent": 4.0,
    }
    codes = []
    for label in phase:
        codes.append(mapping.get(str(label).strip(), 0.0))
    return np.asarray(codes, dtype="float64")


__all__ = ["build_traj_from_truth"]
