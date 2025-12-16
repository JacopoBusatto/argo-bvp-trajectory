from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np


Array = np.ndarray


class IntegratorMethod(str, Enum):
    EULER = "euler"
    TRAPEZOID = "trapezoid"
    RK4 = "rk4"


def _as_2d(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got shape {x.shape}")
    return x


def _check_time(t: Array) -> Array:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1D array with at least 2 elements")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")
    return t


def integrate_2nd_order(
    t: Array,
    r0: Array,
    v0: Array,
    a_fun: Callable[[float, Array, Array], Array],
    method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID,
    backward: bool = False,
) -> Tuple[Array, Array]:
    """
    Integrate second-order kinematics:
        dr/dt = v
        dv/dt = a_fun(t, r, v)

    Parameters
    ----------
    t : (N,) array
        Time grid. Must be strictly increasing.
    r0 : (D,) array
        Initial position (forward) or final position (backward) depending on backward flag.
    v0 : (D,) array
        Initial velocity (forward) or final velocity (backward) depending on backward flag.
    a_fun : callable
        Acceleration model: a_fun(ti, r_i, v_i) -> (D,)
    method : {"euler","trapezoid","rk4"}
        Integration scheme.
    backward : bool
        If True, integrates from t[-1] to t[0] returning arrays aligned with original t.
        The inputs r0,v0 are interpreted as values at t[-1].

    Returns
    -------
    r, v : (N,D) arrays
        Position and velocity arrays aligned with t (increasing order).
    """
    method = IntegratorMethod(method)
    t = _check_time(t)

    r0 = np.asarray(r0, dtype=float).reshape(-1)
    v0 = np.asarray(v0, dtype=float).reshape(-1)
    if r0.shape != v0.shape:
        raise ValueError("r0 and v0 must have the same shape (D,)")

    # Work on a time grid direction
    if backward:
        t_work = t[::-1].copy()
    else:
        t_work = t.copy()

    N = t_work.size
    D = r0.size

    r = np.empty((N, D), dtype=float)
    v = np.empty((N, D), dtype=float)

    r[0] = r0
    v[0] = v0

    def a_at(i: int) -> Array:
        return np.asarray(a_fun(t_work[i], r[i], v[i]), dtype=float).reshape(D)

    if method == IntegratorMethod.EULER:
        for i in range(N - 1):
            dt  = t_work[i + 1] - t_work[i]
            a_i = a_at(i)
            v[i + 1] = v[i] + a_i * dt
            r[i + 1] = r[i] + v[i] * dt

    elif method == IntegratorMethod.TRAPEZOID:
        # predictor-corrector trapezoid on v, and trapezoid on r with v-avg
        for i in range(N - 1):
            dt = t_work[i + 1] - t_work[i]

            a_i = a_at(i)

            # predictor (Euler)
            v_pred = v[i] + a_i * dt
            r_pred = r[i] + v[i] * dt

            # correct acceleration at predicted state (semi-implicit trapezoid)
            a_ip1 = np.asarray(a_fun(t_work[i + 1], r_pred, v_pred), dtype=float).reshape(D)

            v[i + 1] = v[i] + 0.5 * (a_i + a_ip1) * dt
            r[i + 1] = r[i] + 0.5 * (v[i] + v[i + 1]) * dt

    elif method == IntegratorMethod.RK4:
        for i in range(N - 1):
            dt = t_work[i + 1] - t_work[i]
            ti = t_work[i]

            ri = r[i]
            vi = v[i]

            def f_r(tk: float, rk: Array, vk: Array) -> Array:
                return vk

            def f_v(tk: float, rk: Array, vk: Array) -> Array:
                return np.asarray(a_fun(tk, rk, vk), dtype=float).reshape(D)

            k1r = f_r(ti, ri, vi)
            k1v = f_v(ti, ri, vi)

            k2r = f_r(ti + 0.5 * dt, ri + 0.5 * dt * k1r, vi + 0.5 * dt * k1v)
            k2v = f_v(ti + 0.5 * dt, ri + 0.5 * dt * k1r, vi + 0.5 * dt * k1v)

            k3r = f_r(ti + 0.5 * dt, ri + 0.5 * dt * k2r, vi + 0.5 * dt * k2v)
            k3v = f_v(ti + 0.5 * dt, ri + 0.5 * dt * k2r, vi + 0.5 * dt * k2v)

            k4r = f_r(ti + dt, ri + dt * k3r, vi + dt * k3v)
            k4v = f_v(ti + dt, ri + dt * k3r, vi + dt * k3v)

            r[i + 1] = ri + (dt / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
            v[i + 1] = vi + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Re-align to original increasing time axis
    if backward:
        return r[::-1].copy(), v[::-1].copy()
    return r, v
