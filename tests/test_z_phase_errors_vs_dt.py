import numpy as np

from argobvp.integrators import IntegratorMethod
from argobvp.z_profiles import argo_piecewise_z_profile
from argobvp.z_sources import integrate_z_from_accel


def phase_errors_interp(t, z_num, t_truth, z_truth, *, t_dive_end, t_ascent_start, T):
    def err_at(tt: float) -> float:
        zN = float(np.interp(tt, t, z_num))
        zT = float(np.interp(tt, t_truth, z_truth))
        return abs(zN - zT)

    return {
        "dive_end": err_at(float(t_dive_end)),
        "ascent_start": err_at(float(t_ascent_start)),
        "final": err_at(float(T)),
    }


def test_z_phase_errors_vs_dt_show_convergence_trend():
    # Non-aligned phases
    T = 24_013.0
    t_dive_end = 3_973.0
    t_ascent_start = 18_037.0
    z_max = 1200.0

    # High-res truth
    t_truth = np.linspace(0.0, T, 240_131)  # ~0.1s
    z_true, vz_true, az_true = argo_piecewise_z_profile(
        t_truth,
        z_max=z_max,
        t_dive_end=t_dive_end,
        t_ascent_start=t_ascent_start,
        T=T,
    )

    # dt sweep (small->large)
    dt_list = [5.0, 10.0, 20.0, 40.0, 80.0]
    phases = ["dive_end", "ascent_start", "final"]
    methods = [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID, IntegratorMethod.RK4]

    # store RMS-over-phases error per dt
    e = {m: [] for m in methods}

    for dt in dt_list:
        N = int(np.floor(T / dt))
        t = np.linspace(0.0, N * dt, N + 1)
        T_eff = float(t[-1])

        az_s = np.interp(t, t_truth, az_true)
        z0 = float(np.interp(0.0, t_truth, z_true))
        vz0 = float(np.interp(0.0, t_truth, vz_true))

        for m in methods:
            z_m, _ = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=m)
            errs = phase_errors_interp(
                t, z_m, t_truth, z_true,
                t_dive_end=t_dive_end,
                t_ascent_start=t_ascent_start,
                T=T_eff,
            )
            vals = np.array([errs[p] for p in phases], dtype=float)
            e[m].append(float(np.sqrt(np.mean(vals**2))))

    # --- Assertions (robust) ---
    # 1) Each method should improve when dt decreases: compare smallest vs largest dt.
    for m in methods:
        assert e[m][0] < e[m][-1]

    # 2) Smooth global trend: errors should be non-decreasing as dt increases
    # Allow tiny numerical noise with a small tolerance.
    tol = 1e-9
    for m in methods:
        for i in range(len(dt_list) - 1):
            assert e[m][i] <= e[m][i + 1] + tol

    # 3) Sanity: RK4 should not be catastrophically worse than Euler (stability check)
    # (We do NOT enforce it being better because sampling/interp can dominate.)
    assert float(np.mean(e[IntegratorMethod.RK4])) <= float(np.mean(e[IntegratorMethod.EULER])) * 5.0
