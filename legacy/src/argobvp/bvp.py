from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import root

from .integrators import integrate_2nd_order, IntegratorMethod


Array = np.ndarray


@dataclass(frozen=True)
class ShootingResult:
    v0_opt: Array
    success: bool
    message: str
    nfev: int
    r: Array
    v: Array


def shoot_v0_to_hit_rT(
    t: Array,
    r0: Array,
    rT_target: Array,
    v0_guess: Array,
    a_fun: Callable[[float, Array, Array], Array],
    method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID,
    dims: Tuple[int, ...] = (0, 1),
) -> ShootingResult:
    """
    Solve a boundary value constraint on position by shooting on v0.

    We enforce:
        r(T)[dims] = rT_target[dims]

    Unknowns:
        v0[dims]

    Parameters
    ----------
    t : (N,) array
        Time grid.
    r0 : (D,) array
        Initial position.
    rT_target : (D,) array
        Target final position at T=t[-1].
    v0_guess : (D,) array
        Initial guess for v0 (full D-vector; only 'dims' are optimized).
    a_fun : callable
        Acceleration model a(t, r, v).
    method : integration method
    dims : tuple of ints
        Which components of position/velocity are constrained/optimized.
        For Argo with prescribed z, typically dims=(0,1) i.e. XY only.

    Returns
    -------
    ShootingResult
    """
    t = np.asarray(t, dtype=float)
    r0 = np.asarray(r0, dtype=float).reshape(-1)
    rT_target = np.asarray(rT_target, dtype=float).reshape(-1)
    v0_guess = np.asarray(v0_guess, dtype=float).reshape(-1)

    if r0.shape != rT_target.shape or r0.shape != v0_guess.shape:
        raise ValueError("r0, rT_target, v0_guess must have same shape (D,)")

    D = r0.size
    dims = tuple(int(d) for d in dims)
    for d in dims:
        if d < 0 or d >= D:
            raise ValueError(f"Invalid dim {d} for D={D}")

    x0 = v0_guess[list(dims)].copy()

    def F(x):
        v0 = v0_guess.copy()
        v0[list(dims)] = x
        r, v = integrate_2nd_order(t=t, r0=r0, v0=v0, a_fun=a_fun, method=method, backward=False)
        return (r[-1, list(dims)] - rT_target[list(dims)])

    sol = root(F, x0, method="hybr")

    # Build final trajectory using solution (or best effort)
    v0_opt = v0_guess.copy()
    v0_opt[list(dims)] = sol.x
    r_opt, v_opt = integrate_2nd_order(t=t, r0=r0, v0=v0_opt, a_fun=a_fun, method=method, backward=False)

    return ShootingResult(
        v0_opt=v0_opt,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev) if hasattr(sol, "nfev") else -1,
        r=r_opt,
        v=v_opt,
    )
