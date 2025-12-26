import numpy as np

from argobvp.integrators import IntegratorMethod
from argobvp.z_profiles import argo_piecewise_z_profile
from argobvp.z_sources import build_z_from_pressure, integrate_z_from_accel


def test_z_from_pressure_matches_truth_no_noise():
    T = 24_000.0
    t_dive_end = 4_000.0
    t_ascent_start = 18_000.0
    z_max = 1200.0

    t = np.linspace(0.0, T, 2401)  # dt=10s
    z_true, vz_true, az_true = argo_piecewise_z_profile(
        t, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=T
    )

    rho, g = 1025.0, 9.81
    p = rho * g * z_true  # consistent with z>0 downward

    z_p = build_z_from_pressure(t, p, rho=rho, g=g, z0=z_true[0])

    # should match almost exactly (float rounding only)
    assert np.max(np.abs(z_p - z_true)) < 1e-10


def test_z_from_accel_converges_and_trapezoid_beats_euler():
    T = 24_000.0
    t_dive_end = 4_000.0
    t_ascent_start = 18_000.0
    z_max = 1200.0

    # "truth" on a very fine grid
    t_truth = np.linspace(0.0, T, 240_001)  # dt=0.1s
    z_true, vz_true, az_true = argo_piecewise_z_profile(
        t_truth, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=T
    )

    # coarser grids emulate IMU sampling / integration steps
    dt_list = [10.0, 20.0, 40.0]  # seconds
    err_euler = []
    err_trap = []

    for dt in dt_list:
        N = int(T / dt)
        t = np.linspace(0.0, T, N + 1)

        # sample az from truth
        az_s = np.interp(t, t_truth, az_true)
        z0 = float(np.interp(0.0, t_truth, z_true))
        vz0 = float(np.interp(0.0, t_truth, vz_true))

        # integrate with Euler
        z_e, _ = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=IntegratorMethod.EULER)
        # integrate with Trapezoid
        z_t, _ = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=IntegratorMethod.TRAPEZOID)

        # compare against truth sampled on the same grid
        z_ref = np.interp(t, t_truth, z_true)

        err_euler.append(float(np.linalg.norm(z_e - z_ref, ord=np.inf)))
        err_trap.append(float(np.linalg.norm(z_t - z_ref, ord=np.inf)))

        # trapezoid should be better for this smooth profile
        assert err_trap[-1] < err_euler[-1]

    # convergence: errors should decrease as dt decreases
    assert err_trap[0] <= err_trap[1] + 1e-12  # dt=10 <= dt=20 (allow tiny numerical tie)
    assert err_trap[1] <= err_trap[2] + 1e-12
