"""Surface window detection and anchor selection."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def find_surface_windows(
    juld: np.ndarray,
    pres: np.ndarray,
    p_surface: float,
    max_gap_seconds: float | None = None,
) -> list[tuple[int, int]]:
    """Find contiguous surface windows where pressure stays below p_surface."""
    juld_arr = np.asarray(juld, dtype=float)
    pres_arr = np.asarray(pres, dtype=float)
    if juld_arr.shape != pres_arr.shape:
        raise ValueError("juld and pres must have the same shape")
    if juld_arr.ndim != 1:
        raise ValueError("juld and pres must be 1D arrays")
    if juld_arr.size == 0:
        return []

    finite_mask = np.isfinite(juld_arr) & np.isfinite(pres_arr)
    if np.any(np.diff(juld_arr[finite_mask]) < 0):
        raise ValueError("juld must be monotonic non-decreasing")

    surface_mask = finite_mask & (pres_arr <= float(p_surface))
    indices = np.where(surface_mask)[0]
    if indices.size == 0:
        return []

    gap_days = None
    if max_gap_seconds is not None:
        gap_days = float(max_gap_seconds) / 86400.0

    windows: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        gap_ok = True
        if gap_days is not None:
            gap_ok = (juld_arr[idx] - juld_arr[prev]) <= gap_days
        if idx != prev + 1 or not gap_ok:
            windows.append((start, prev + 1))
            start = idx
        prev = idx
    windows.append((start, prev + 1))
    return windows


def select_anchor_points(
    juld: np.ndarray,
    pres: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    position_qc: Iterable[str],
    windows: list[tuple[int, int]],
    window_index: int,
    qc_ok: Iterable[str],
) -> tuple[dict[str, object], dict[str, object]]:
    """Select anchor points from surface windows using QC filtering."""
    if window_index < 0:
        raise ValueError("window_index must be >= 0")
    if window_index + 1 >= len(windows):
        raise ValueError("window_index must have a subsequent surface window")

    juld_arr = np.asarray(juld, dtype=float)
    pres_arr = np.asarray(pres, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    qc_arr = np.asarray(list(position_qc))

    _require_same_length(juld_arr, pres_arr, lat_arr, lon_arr, qc_arr)

    qc_ok_set = {str(code).strip() for code in qc_ok}

    start_window = windows[window_index]
    end_window = windows[window_index + 1]

    start_idx = _select_last_valid(
        start_window,
        juld_arr,
        lat_arr,
        lon_arr,
        qc_arr,
        qc_ok_set,
    )
    end_idx = _select_first_valid(
        end_window,
        juld_arr,
        lat_arr,
        lon_arr,
        qc_arr,
        qc_ok_set,
    )

    start_anchor = _build_anchor_dict(start_idx, juld_arr, lat_arr, lon_arr, qc_arr)
    end_anchor = _build_anchor_dict(end_idx, juld_arr, lat_arr, lon_arr, qc_arr)
    return start_anchor, end_anchor


def _select_last_valid(
    window: tuple[int, int],
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    qc: np.ndarray,
    qc_ok: set[str],
) -> int:
    indices = range(window[0], window[1])
    valid = [
        idx
        for idx in indices
        if _is_valid_fix(juld[idx], lat[idx], lon[idx], qc[idx], qc_ok)
    ]
    if not valid:
        raise ValueError(f"No valid GPS fix in window {window}")
    return int(valid[-1])


def _select_first_valid(
    window: tuple[int, int],
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    qc: np.ndarray,
    qc_ok: set[str],
) -> int:
    indices = range(window[0], window[1])
    for idx in indices:
        if _is_valid_fix(juld[idx], lat[idx], lon[idx], qc[idx], qc_ok):
            return int(idx)
    raise ValueError(f"No valid GPS fix in window {window}")


def _is_valid_fix(
    juld: float,
    lat: float,
    lon: float,
    qc: object,
    qc_ok: set[str],
) -> bool:
    if not (np.isfinite(juld) and np.isfinite(lat) and np.isfinite(lon)):
        return False
    qc_val = str(qc).strip()
    return qc_val in qc_ok


def _build_anchor_dict(
    idx: int,
    juld: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    qc: np.ndarray,
) -> dict[str, object]:
    return {
        "index": int(idx),
        "juld": float(juld[idx]),
        "lat": float(lat[idx]),
        "lon": float(lon[idx]),
        "position_qc": str(qc[idx]).strip(),
    }


def _require_same_length(*arrays: np.ndarray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("Input arrays must have the same length")


__all__ = ["find_surface_windows", "select_anchor_points"]
