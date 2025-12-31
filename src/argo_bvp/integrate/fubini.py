"""1D Fubini boundary-value integrator."""

from __future__ import annotations

import numpy as np


def integrate_fubini_1d(
    t_s: np.ndarray,
    a: np.ndarray,
    x0: float,
    xT: float,
    method: str = "trap",
) -> np.ndarray:
    """Reconstruct x(t) from acceleration with endpoint constraints."""
    t = np.asarray(t_s, dtype=float)
    acc = np.asarray(a, dtype=float)

    if t.ndim != 1 or acc.ndim != 1:
        raise ValueError("t_s and a must be 1D arrays")
    if t.size != acc.size:
        raise ValueError("t_s and a must have the same length")
    if t.size < 2:
        raise ValueError("t_s must have at least 2 samples")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(acc)):
        raise ValueError("t_s and a must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_s must be strictly increasing")

    tau = t - t[0]
    T = float(tau[-1])
    if T <= 0:
        raise ValueError("T must be positive")

    dt = np.diff(tau)
    I = np.zeros_like(tau)
    K = np.zeros_like(tau)

    if method == "trap":
        for i in range(tau.size - 1):
            I[i + 1] = I[i] + 0.5 * (acc[i] + acc[i + 1]) * dt[i]
            K[i + 1] = K[i] + 0.5 * (
                tau[i] * acc[i] + tau[i + 1] * acc[i + 1]
            ) * dt[i]
    elif method == "rect":
        for i in range(tau.size - 1):
            I[i + 1] = I[i] + acc[i] * dt[i]
            K[i + 1] = K[i] + (tau[i] * acc[i]) * dt[i]
    else:
        raise ValueError("method must be 'trap' or 'rect'")

    J = tau * I - K
    C = T * I[-1] - K[-1]
    x = x0 + ((xT - x0) / T) * tau + J - (tau / T) * C

    x[0] = float(x0)
    x[-1] = float(xT)
    return x


__all__ = ["integrate_fubini_1d"]
