import numpy as np

from argobvp.integrators import integrate_2nd_order, IntegratorMethod
from argobvp.metrics import nearest_index


def test_argo_style_forward_endpoint_and_phase_errors():
    # --- "Argo cycle" key times (seconds) ---
    T = 24_000.0  # ~6.7 hours (toy)
    t_dive_end = 4_000.0
    t_ascent_start = 18_000.0

    # --- Initial surface fix ---
    r0 = np.array([0.0, 0.0, 0.0])      # (x,y,z) in arbitrary units
    v0 = np.array([0.25, -0.10, 0.0])   # initial velocity

    # --- Smooth time-varying acceleration (toy ocean forcing) ---
    w1 = 2 * np.pi / 8_000.0
    w2 = 2 * np.pi / 2_500.0
    A = np.array([0.00030, -0.00018, 0.0])

    def a_true(ti, r, v):
        ti = float(ti)
        ax = A[0] * (np.sin(w1 * ti) + 0.3 * np.sin(w2 * ti))
        ay = A[1] * (np.cos(w1 * ti) + 0.2 * np.cos(w2 * ti))
        az = 0.0
        return np.array([ax, ay, az], dtype=float)

    # --- High-resolution "truth" (proxy for real trajectory) ---
    N_truth = 120_000  # dt = 0.2 s
    t_truth = np.linspace(0.0, T, N_truth + 1)
    r_truth, v_truth = integrate_2nd_order(
        t=t_truth, r0=r0, v0=v0, a_fun=a_true, method=IntegratorMethod.RK4, backward=False
    )

    # Targets at key times from truth
    idx_dive_end = nearest_index(t_truth, t_dive_end)
    idx_ascent_start = nearest_index(t_truth, t_ascent_start)

    r_target_final = r_truth[-1].copy()
    r_target_dive_end = r_truth[idx_dive_end].copy()
    r_target_ascent_start = r_truth[idx_ascent_start].copy()

    # --- Simulate having acceleration measurements on a coarser grid ---
    # This mimics "we have a(t) sampled", then we integrate forward.
    def make_sampled_a_fun_linear(t_grid):
        t_grid = np.asarray(t_grid, dtype=float)
        a_samples = np.array([a_true(tt, None, None) for tt in t_grid], dtype=float)

        ax = a_samples[:, 0]
        ay = a_samples[:, 1]
        az = a_samples[:, 2]

        def a_sampled(ti, r, v):
            ti = float(ti)
            # linear interpolation in time (clamped to endpoints)
            ax_i = np.interp(ti, t_grid, ax)
            ay_i = np.interp(ti, t_grid, ay)
            az_i = np.interp(ti, t_grid, az)
            return np.array([ax_i, ay_i, az_i], dtype=float)

        return a_sampled

    # Choose a realistic coarse dt family (minutes)
    dt = 60.0  # 1 minute sampling
    N = int(T / dt)
    t_coarse = np.linspace(0.0, T, N + 1)
    a_fun_sampled = make_sampled_a_fun_linear(t_coarse)

    # --- Integrate forward with sampled acceleration ---
    results = {}
    for m in [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID, IntegratorMethod.RK4]:
        r_num, v_num = integrate_2nd_order(
            t=t_coarse, r0=r0, v0=v0, a_fun=a_fun_sampled, method=m, backward=False
        )
        results[m] = (r_num, v_num)

    # --- Compute errors at key times (norm in XY only is also possible later) ---
    def err_at_time(r_num, t_grid, t_query, r_target):
        i = int(np.argmin(np.abs(t_grid - float(t_query))))
        return float(np.linalg.norm(r_num[i] - r_target))

    err = {}
    for m, (r_num, v_num) in results.items():
        err[m] = {
            "final": float(np.linalg.norm(r_num[-1] - r_target_final)),
            "dive_end": err_at_time(r_num, t_coarse, t_dive_end, r_target_dive_end),
            "ascent_start": err_at_time(r_num, t_coarse, t_ascent_start, r_target_ascent_start),
        }

    # --- Assertions: trapezoid should beat euler, RK4 should be best or comparable ---
    assert err[IntegratorMethod.TRAPEZOID]["final"] < err[IntegratorMethod.EULER]["final"]
    assert err[IntegratorMethod.TRAPEZOID]["dive_end"] < err[IntegratorMethod.EULER]["dive_end"]
    assert err[IntegratorMethod.TRAPEZOID]["ascent_start"] < err[IntegratorMethod.EULER]["ascent_start"]

    # RK4 usually best; allow equality-ish due to sampled acceleration limitations
    assert err[IntegratorMethod.RK4]["final"] <= err[IntegratorMethod.TRAPEZOID]["final"] * 1.05
