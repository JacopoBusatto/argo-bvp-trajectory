"""Tests for Fubini 1D boundary-value integration."""

import numpy as np

from argo_bvp.integrate.fubini import integrate_fubini_1d


def _make_nonuniform_tau() -> np.ndarray:
    steps = np.array([0.0, 0.1, 0.3, 0.05, 0.2, 0.4, 0.12, 0.33, 0.17])
    return np.cumsum(steps)


def test_fubini_hits_anchors_trap_nonuniform() -> None:
    tau = _make_nonuniform_tau()
    t_s = tau + 123.4
    T = float(tau[-1])
    a = np.sin(2.0 * np.pi * tau / T)
    x0 = 0.0
    xT = 100.0

    x = integrate_fubini_1d(t_s, a, x0, xT, method="trap")
    assert np.isfinite(x).all()
    assert abs(x[0] - x0) < 1e-9
    assert abs(x[-1] - xT) < 1e-9


def test_fubini_hits_anchors_rect_nonuniform() -> None:
    tau = _make_nonuniform_tau()
    t_s = tau + 123.4
    T = float(tau[-1])
    a = np.sin(2.0 * np.pi * tau / T)
    x0 = 0.0
    xT = 100.0

    x = integrate_fubini_1d(t_s, a, x0, xT, method="rect")
    assert np.isfinite(x).all()
    assert abs(x[0] - x0) < 1e-9
    assert abs(x[-1] - xT) < 1e-9


def test_fubini_zero_acceleration_gives_linear() -> None:
    tau = _make_nonuniform_tau()
    t_s = tau + 123.4
    T = float(tau[-1])
    a = np.zeros_like(tau)
    x0 = -5.0
    xT = 42.0

    x_lin = x0 + (xT - x0) / T * tau

    x_trap = integrate_fubini_1d(t_s, a, x0, xT, method="trap")
    x_rect = integrate_fubini_1d(t_s, a, x0, xT, method="rect")

    assert np.max(np.abs(x_trap - x_lin)) < 1e-10
    assert np.max(np.abs(x_rect - x_lin)) < 1e-10
