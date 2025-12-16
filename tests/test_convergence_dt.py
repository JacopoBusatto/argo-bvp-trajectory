import numpy as np

from argobvp.integrators import integrate_2nd_order, IntegratorMethod


def _linreg_slope_loglog(x, y):
    """
    Fit log(y) = m log(x) + q, return m.
    """
    lx = np.log(x)
    ly = np.log(y)
    m, q = np.polyfit(lx, ly, 1)
    return m


def test_convergence_orders_endpoint_error():
    """
    Convergence test for endpoint error vs dt.

    We use an acceleration a(t) = sin(omega t) applied to each component,
    for which v(t), r(t) have closed-form expressions.

    Expect:
      - Euler (rectangles/left):   O(dt^1)
      - Trapezoid:                O(dt^2)
      - RK4:                      O(dt^4)  (often close, sometimes limited by problem smoothness / dt range)
    """
    omega = 1.7
    T = 10.0

    # initial conditions
    r0 = np.array([1.0, -2.0, 0.5])
    v0 = np.array([0.3, 0.1, -0.2])

    # acceleration amplitude per component (just to make it 3D and nontrivial)
    A = np.array([0.8, -0.4, 0.2])

    def a_fun(ti, r, v):
        return A * np.sin(omega * ti)

    # analytic solution:
    # v(t) = v0 + ∫ a dt = v0 + A * (1 - cos(omega t)) / omega
    # r(t) = r0 + ∫ v dt
    #      = r0 + v0 t + A * (t/omega - sin(omega t)/omega^2)
    def v_true(t):
        t = np.asarray(t)
        return v0[None, :] + (A[None, :] * (1.0 - np.cos(omega * t)[:, None]) / omega)

    def r_true(t):
        t = np.asarray(t)
        tt = t[:, None]
        return (
            r0[None, :]
            + v0[None, :] * tt
            + A[None, :] * (tt / omega - np.sin(omega * t)[:, None] / (omega**2))
        )

    # dt sequence (refine progressively)
    # Keep it moderate to avoid floating point saturation for RK4 at very fine dt.
    Ns = np.array([50, 100, 200, 400, 800])
    dts = T / Ns

    methods = [
        IntegratorMethod.EULER,       # rettangoli (left)
        IntegratorMethod.TRAPEZOID,   # trapezi
        IntegratorMethod.RK4,         # riferimento alto ordine
    ]

    # Compute endpoint errors for each method vs dt
    endpoint_errors = {m: [] for m in methods}

    for N in Ns:
        t = np.linspace(0.0, T, int(N) + 1)

        rT = r_true(np.array([T]))[0]
        vT = v_true(np.array([T]))[0]

        for m in methods:
            r_num, v_num = integrate_2nd_order(t, r0, v0, a_fun, method=m, backward=False)

            # endpoint error norm (L2 in 3D)
            er = np.linalg.norm(r_num[-1] - rT)
            ev = np.linalg.norm(v_num[-1] - vT)
            endpoint_errors[m].append(er + ev)

    # Convert to arrays
    for m in methods:
        endpoint_errors[m] = np.array(endpoint_errors[m], dtype=float)

    # Basic sanity: errors should decrease as dt decreases (mostly monotone)
    for m in methods:
        errs = endpoint_errors[m]
        assert errs[-1] < errs[0], f"{m} did not improve with refinement"

    # Estimate convergence slope: error ~ C * dt^p  -> log(err) = log(C) + p log(dt)
    p_euler = _linreg_slope_loglog(dts, endpoint_errors[IntegratorMethod.EULER])
    p_trap = _linreg_slope_loglog(dts, endpoint_errors[IntegratorMethod.TRAPEZOID])
    p_rk4 = _linreg_slope_loglog(dts, endpoint_errors[IntegratorMethod.RK4])

    # We accept broad tolerances because real slopes depend on dt range.
    assert 0.8 < p_euler < 1.2, f"Euler order off: {p_euler:.2f}"
    assert 1.7 < p_trap < 2.3, f"Trapezoid order off: {p_trap:.2f}"
    assert 3.2 < p_rk4 < 4.6, f"RK4 order off: {p_rk4:.2f}"

    # And: trapezoid should beat Euler at fine dt
    assert endpoint_errors[IntegratorMethod.TRAPEZOID][-1] < endpoint_errors[IntegratorMethod.EULER][-1]
