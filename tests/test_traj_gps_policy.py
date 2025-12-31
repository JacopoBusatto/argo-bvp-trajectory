"""Tests for GPS/QC policy in synthetic TRAJ generation."""

import numpy as np

from argo_bvp.synth.generate_traj import build_traj_from_truth
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_traj_gps_policy() -> None:
    truth = generate_truth_cycle()
    traj = build_traj_from_truth(truth)

    phase = np.asarray(truth["phase"].values)
    surface_mask = phase == "surface"

    lat = np.asarray(traj["LATITUDE"].values, dtype=float)
    lon = np.asarray(traj["LONGITUDE"].values, dtype=float)
    qc = np.asarray(traj["POSITION_QC"].values, dtype=str)

    assert np.all(np.isnan(lat[~surface_mask]))
    assert np.all(np.isnan(lon[~surface_mask]))

    segments = _surface_segments(surface_mask)
    assert len(segments) >= 2
    surface1 = segments[0]
    surface2 = segments[-1]

    s1_valid = _valid_indices(lat, lon, qc, surface1, "1")
    s2_valid = _valid_indices(lat, lon, qc, surface2, "1")

    assert s1_valid.size > 0
    assert s2_valid.size > 0

    last_s1 = int(s1_valid[-1])
    first_s2 = int(s2_valid[0])

    assert qc[last_s1] == "1"
    assert qc[first_s2] == "1"
    assert np.isfinite(lat[last_s1]) and np.isfinite(lon[last_s1])
    assert np.isfinite(lat[first_s2]) and np.isfinite(lon[first_s2])


def _surface_segments(mask: np.ndarray) -> list[tuple[int, int]]:
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


def _valid_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    qc: np.ndarray,
    segment: tuple[int, int],
    qc_good: str,
) -> np.ndarray:
    start, end = segment
    segment_idx = np.arange(start, end)
    valid = (
        np.isfinite(lat[segment_idx])
        & np.isfinite(lon[segment_idx])
        & (qc[segment_idx] == qc_good)
    )
    return segment_idx[valid]
