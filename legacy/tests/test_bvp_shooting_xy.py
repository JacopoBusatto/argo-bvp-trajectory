import numpy as np

from argobvp.integrators import integrate_2nd_order, IntegratorMethod
from argobvp.bvp import shoot_v0_to_hit_rT


def test_shooting_hits_final_position_xy():
    T = 10_000.0
    t = np.linspace(0.0, T, 2001)

    r0 = np.array([0.0, 0.0, 0.0])
    v0_true = np.array([0.12, -0.07, 0.0])

    w = 2 * np.pi / 4000.0
    A = np.array([2e-4, -1.5e-4, 0.0])

    def a_fun(ti, r, v):
        return np.array([A[0] * np.sin(w * ti), A[1] * np.cos(w * ti), 0.0])

    # Generate "truth" endpoint from true v0
    r_true, v_true = integrate_2nd_order(t, r0, v0_true, a_fun, method=IntegratorMethod.RK4, backward=False)
    rT_target = r_true[-1].copy()

    # Bad guess
    v0_guess = np.array([0.0, 0.0, 0.0])

    res = shoot_v0_to_hit_rT(
        t=t,
        r0=r0,
        rT_target=rT_target,
        v0_guess=v0_guess,
        a_fun=a_fun,
        method=IntegratorMethod.TRAPEZOID,
        dims=(0, 1),
    )

    assert res.success
    assert np.linalg.norm(res.r[-1, :2] - rT_target[:2]) < 1e-6
