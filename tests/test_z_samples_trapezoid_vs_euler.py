import numpy as np

from argobvp.integrators import IntegratorMethod
from argobvp.z_profiles import argo_piecewise_z_profile
from argobvp.z_sources import integrate_z_from_accel_samples


def test_z_samples_trapezoid_beats_euler_on_smooth_profile():
    # Use non-aligned times to avoid node coincidences
    T = 24_013.0
    t_dive_end = 3_973.0
    t_ascent_start = 18_037.0
    z_max = 1200.0

    # reference truth: evaluate analytic profile on each grid directly
    def run(dt: float):
        N = int(np.floor(T / dt))
        t = np.linspace(0.0, N * dt, N + 1)
        z_ref, vz_ref, az_ref = argo_piecewise_z_profile(
            t, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=float(t[-1])
        )
        z0 = float(z_ref[0])
        vz0 = float(vz_ref[0])

        z_e, _ = integrate_z_from_accel_samples(t, az_ref, z0=z0, vz0=vz0, method=IntegratorMethod.EULER)
        z_t, _ = integrate_z_from_accel_samples(t, az_ref, z0=z0, vz0=vz0, method=IntegratorMethod.TRAPEZOID)

        # Compare against the analytic truth on the same grid (no interpolation)
        err_e = float(np.linalg.norm(z_e - z_ref, ord=np.inf))
        err_t = float(np.linalg.norm(z_t - z_ref, ord=np.inf))
        return err_e, err_t

    # A few dt values; trapezoid should consistently improve for this smooth forcing
    dt_list = [6.0, 13.0, 29.0, 61.0]
    for dt in dt_list:
        err_e, err_t = run(dt)
        assert err_t < err_e


def test_z_samples_converges_when_dt_decreases():
    T = 24_013.0
    t_dive_end = 3_973.0
    t_ascent_start = 18_037.0
    z_max = 1200.0

    def err(dt: float, method: IntegratorMethod):
        N = int(np.floor(T / dt))
        t = np.linspace(0.0, N * dt, N + 1)
        z_ref, vz_ref, az_ref = argo_piecewise_z_profile(
            t, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=float(t[-1])
        )
        z0 = float(z_ref[0])
        vz0 = float(vz_ref[0])
        z_m, _ = integrate_z_from_accel_samples(t, az_ref, z0=z0, vz0=vz0, method=method)
        return float(np.linalg.norm(z_m - z_ref, ord=np.inf))

    # decreasing dt should reduce error
    dt_hi = 61.0
    dt_lo = 13.0
    assert err(dt_lo, IntegratorMethod.EULER) < err(dt_hi, IntegratorMethod.EULER)
    assert err(dt_lo, IntegratorMethod.TRAPEZOID) < err(dt_hi, IntegratorMethod.TRAPEZOID)
