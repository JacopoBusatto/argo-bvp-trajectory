import numpy as np

Array = np.ndarray

def argo_piecewise_z_profile(
    t: Array,
    *,
    z_max: float,
    t_dive_end: float,
    t_ascent_start: float,
    T: float | None = None,
    z0: float = 0.0,
) -> tuple[Array, Array, Array]:
    """
    Smooth Argo-like depth profile (z>0 downward), C^2 at phase boundaries.

    Phases:
      - descent: 0 -> z_max on [0, t_dive_end] using quintic smoothstep
      - parking: z_max constant on (t_dive_end, t_ascent_start)
      - ascent:  z_max -> 0 on [t_ascent_start, T] using quintic smoothstep

    Returns:
      z(t), vz(t), az(t) arrays.

    Smoothstep:
      s(u)  = 10u^3 - 15u^4 + 6u^5
      s'(u) = 30u^2 - 60u^3 + 30u^4
      s''(u)= 60u - 180u^2 + 120u^3

    Properties:
      s'(0)=s'(1)=0 and s''(0)=s''(1)=0 -> matches parking with vz=0, az=0.
    """
    t = np.asarray(t, dtype=float)
    if T is None:
        T = float(t[-1])
    else:
        T = float(T)

    t_dive_end = float(t_dive_end)
    t_ascent_start = float(t_ascent_start)

    if not (0.0 < t_dive_end < t_ascent_start < T):
        raise ValueError("Require 0 < t_dive_end < t_ascent_start < T")

    z = np.zeros_like(t, dtype=float)
    vz = np.zeros_like(t, dtype=float)
    az = np.zeros_like(t, dtype=float)

    def s(u):
        return 10*u**3 - 15*u**4 + 6*u**5

    def sp(u):
        return 30*u**2 - 60*u**3 + 30*u**4

    def spp(u):
        return 60*u - 180*u**2 + 120*u**3

    # --- descent
    m1 = (t >= 0.0) & (t <= t_dive_end)
    u1 = (t[m1] - 0.0) / t_dive_end
    z[m1] = z_max * s(u1)
    vz[m1] = z_max * sp(u1) / t_dive_end
    az[m1] = z_max * spp(u1) / (t_dive_end**2)

    # --- parking
    m2 = (t > t_dive_end) & (t < t_ascent_start)
    z[m2] = z_max
    vz[m2] = 0.0
    az[m2] = 0.0

    # --- ascent
    m3 = (t >= t_ascent_start) & (t <= T)
    Ta = (T - t_ascent_start)
    u3 = (t[m3] - t_ascent_start) / Ta
    # go from z_max down to 0 smoothly
    z[m3] = z_max * (1.0 - s(u3))
    vz[m3] = -z_max * sp(u3) / Ta
    az[m3] = -z_max * spp(u3) / (Ta**2)

    # shift so z(t[0]) == z0
    z = z - z[0] + float(z0)

    return z, vz, az
