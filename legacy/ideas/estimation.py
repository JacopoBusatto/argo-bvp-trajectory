from __future__ import annotations
import numpy as np

Array = np.ndarray


def estimate_v0_from_surface_gps(
    t: Array,
    x: Array,
    y: Array,
    *,
    t0: float | None = None,
    t1: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Estimate horizontal velocity (v0x, v0y) from surface GPS fixes using
    least-squares linear regression:

        x(t) ≈ a_x + v_x t
        y(t) ≈ a_y + v_y t

    Parameters
    ----------
    t, x, y : arrays of same length
        Surface GPS timestamps and positions (projected coordinates recommended).
    t0, t1 : float, optional
        If provided, use only fixes with t in [t0, t1].

    Returns
    -------
    v0_xy : (2,) array
        Estimated [v0x, v0y].
    info : dict
        Diagnostics: intercepts, residual RMS, number of points, time span.
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if not (t.size == x.size == y.size):
        raise ValueError("t, x, y must have the same length")

    m = np.isfinite(t) & np.isfinite(x) & np.isfinite(y)
    if t0 is not None:
        m &= t >= float(t0)
    if t1 is not None:
        m &= t <= float(t1)

    t = t[m]
    x = x[m]
    y = y[m]
    if t.size < 2:
        raise ValueError("Need at least 2 valid GPS fixes to estimate velocity")

    # improve conditioning: fit using centered time
    tc = t - t.mean()
    G = np.column_stack([np.ones_like(tc), tc])  # [1, (t - mean(t))]

    # least squares
    px, *_ = np.linalg.lstsq(G, x, rcond=None)
    py, *_ = np.linalg.lstsq(G, y, rcond=None)

    ax, vx = px[0], px[1]
    ay, vy = py[0], py[1]

    xhat = G @ px
    yhat = G @ py
    rms = float(np.sqrt(np.mean((x - xhat) ** 2 + (y - yhat) ** 2)))

    info = dict(
        n=int(t.size),
        t_span=float(t.max() - t.min()),
        intercept=np.array([ax, ay], dtype=float),
        rms_xy=rms,
    )
    return np.array([vx, vy], dtype=float), info
