import numpy as np
from .integrators import integrate_2nd_order, IntegratorMethod
from typing import Tuple


Array = np.ndarray

# +1 => z positive downward (our Argo convention)
# -1 => z positive upward (if you ever need it)
Z_SIGN: float = +1.0


def build_z_from_pressure(
    t: Array,
    p: Array,
    *,
    rho: float = 1025.0,
    g: float = 9.81,
    z0: float = 0.0,
) -> Array:
    """
    Convert pressure p(t) to depth z(t) using a simple hydrostatic relation:

        z = Z_SIGN * p / (rho * g)

    z>0 downward if Z_SIGN=+1.

    The output is shifted so that z(t[0]) == z0.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.shape != p.shape:
        raise ValueError("t and p must have the same shape")

    z = Z_SIGN * (p / (rho * g))
    z = z - z[0] + float(z0)
    return z


def integrate_z_from_accel(
    t: Array,
    az: Array,
    *,
    z0: float = 0.0,
    vz0: float = 0.0,
    method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID,
) -> tuple[Array, Array]:
    """
    Integrate vertical acceleration az(t) -> vz(t) -> z(t).

    az is assumed positive downward if Z_SIGN=+1.

    Returns:
      z(t), vz(t)
    """
    t = np.asarray(t, dtype=float)
    az = np.asarray(az, dtype=float)
    if t.shape != az.shape:
        raise ValueError("t and az must have the same shape")

    az_eff = Z_SIGN * az

    def a_fun(ti, r, v):
        # linear interpolation of az on the provided grid
        a = float(np.interp(float(ti), t, az_eff))
        return np.array([a], dtype=float)

    r, v = integrate_2nd_order(
        t=t,
        r0=np.array([float(z0)], dtype=float),
        v0=np.array([float(vz0)], dtype=float),
        a_fun=a_fun,
        method=method,
        backward=False,
    )
    return r[:, 0], v[:, 0]




def integrate_z_from_accel_samples(
    t: Array,
    az: Array,
    *,
    z0: float,
    vz0: float,
    method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID,
) -> Tuple[Array, Array]:
    """
    Integrate vertical motion from *sampled* vertical acceleration az[k] given on the same grid t[k].

    State:
        dz/dt = vz
        dvz/dt = az(t)

    This function assumes:
      - t is strictly increasing, 1D
      - az has same length as t
      - z positive downward

    Methods:
      - Euler:        vz_{k+1} = vz_k + dt * az_k
                      z_{k+1}  = z_k  + dt * vz_k
      - Trapezoid:    vz_{k+1} = vz_k + dt * 0.5*(az_k + az_{k+1})
                      z_{k+1}  = z_k  + dt * 0.5*(vz_k + vz_{k+1})

    Note: RK4 is not supported in "samples" mode because it requires a continuous a(t)
          (or a reconstruction model for intermediate stages).
    """
    t = np.asarray(t, dtype=float)
    az = np.asarray(az, dtype=float)

    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1D array with at least 2 elements")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if az.shape != t.shape:
        raise ValueError("az must have the same shape as t")

    if isinstance(method, str):
        method = IntegratorMethod(method)

    if method == IntegratorMethod.RK4:
        raise ValueError("RK4 is not supported for sampled acceleration (no continuous a(t)).")

    n = t.size
    z = np.empty(n, dtype=float)
    vz = np.empty(n, dtype=float)
    z[0] = float(z0)
    vz[0] = float(vz0)

    for k in range(n - 1):
        dt = t[k + 1] - t[k]

        if method == IntegratorMethod.EULER:
            vz[k + 1] = vz[k] + dt * az[k]
            z[k + 1] = z[k] + dt * vz[k]

        elif method == IntegratorMethod.TRAPEZOID:
            vz[k + 1] = vz[k] + dt * 0.5 * (az[k] + az[k + 1])
            z[k + 1] = z[k] + dt * 0.5 * (vz[k] + vz[k + 1])

        else:
            raise ValueError(f"Unsupported method for sampled integration: {method}")

    return z, vz
