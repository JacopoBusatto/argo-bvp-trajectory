"""Frame consistency checks between x/y and accelerations."""

import numpy as np

from argo_bvp.synth.generate_truth import generate_truth_cycle


def _second_derivative(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    v = np.gradient(x, t, edge_order=2)
    return np.gradient(v, t, edge_order=2)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def test_truth_frame_consistency() -> None:
    ds = generate_truth_cycle()
    t = np.asarray(ds["t"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    y = np.asarray(ds["y"].values, dtype=float)
    ax = np.asarray(ds["ax"].values, dtype=float)
    ay = np.asarray(ds["ay"].values, dtype=float)

    d2x = _second_derivative(t, x)
    d2y = _second_derivative(t, y)

    trim = slice(2, -2)
    corr_x = _corr(ax[trim], d2x[trim])
    corr_y = _corr(ay[trim], d2y[trim])

    assert corr_x > 0.8
    assert corr_y > 0.8
