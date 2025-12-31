import numpy as np

from argobvp.integrators import integrate_2nd_order, IntegratorMethod


def test_constant_acceleration_matches_analytic_rk4():
    # Analytic solution:
    # r(t) = r0 + v0*t + 0.5*a*t^2
    # v(t) = v0 + a*t
    t = np.linspace(0.0, 10.0, 2001)
    r0 = np.array([1.0, -2.0, 0.5])
    v0 = np.array([0.3, 0.1, -0.2])
    a = np.array([0.01, -0.02, 0.03])

    def a_fun(ti, r, v):
        return a

    r_num, v_num = integrate_2nd_order(
        t=t,
        r0=r0,
        v0=v0,
        a_fun=a_fun,
        method=IntegratorMethod.RK4,
        backward=False,
    )

    tt = t[:, None]
    r_true = r0[None, :] + v0[None, :] * tt + 0.5 * a[None, :] * tt**2
    v_true = v0[None, :] + a[None, :] * tt

    err_r = np.max(np.abs(r_num - r_true))
    err_v = np.max(np.abs(v_num - v_true))

    # RK4 on smooth poly should be extremely accurate at this resolution
    assert err_r < 1e-8
    assert err_v < 1e-10


def test_forward_then_backward_reconstructs_endpoints_trapezoid():
    t = np.linspace(0.0, 5.0, 501)
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([1.0, -0.5, 0.2])
    a = np.array([0.2, 0.1, -0.05])

    def a_fun(ti, r, v):
        return a

    # Forward
    r_f, v_f = integrate_2nd_order(t, r0, v0, a_fun, method="trapezoid", backward=False)
    r1 = r_f[-1].copy()
    v1 = v_f[-1].copy()

    # Backward from final state should recover the initial state
    r_b, v_b = integrate_2nd_order(t, r1, v1, a_fun, method="trapezoid", backward=True)

    assert np.max(np.abs(r_b[0] - r0)) < 1e-6
    assert np.max(np.abs(v_b[0] - v0)) < 1e-6
