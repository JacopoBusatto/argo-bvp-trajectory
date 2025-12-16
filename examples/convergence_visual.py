import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import integrate_2nd_order, IntegratorMethod


def linreg_slope_loglog(x, y):
    lx = np.log(x)
    ly = np.log(y)
    m, q = np.polyfit(lx, ly, 1)
    return m, q


def main():
    omega = 1.7
    T = 10.0

    r0 = np.array([1.0, -2.0, 0.5])
    v0 = np.array([0.3, 0.1, -0.2])
    A = np.array([0.8, -0.4, 0.2])

    def a_fun(ti, r, v):
        return A * np.sin(omega * ti)

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

    Ns = np.array([25, 50, 100, 200, 400, 800, 1600])
    dts = T / Ns

    methods = [
        IntegratorMethod.EULER,       # rettangoli (left)
        IntegratorMethod.TRAPEZOID,   # trapezi
        IntegratorMethod.RK4,
    ]

    endpoint_err = {m: [] for m in methods}

    rT = r_true(np.array([T]))[0]
    vT = v_true(np.array([T]))[0]

    for N in Ns:
        t = np.linspace(0.0, T, int(N) + 1)
        for m in methods:
            r_num, v_num = integrate_2nd_order(t, r0, v0, a_fun, method=m, backward=False)
            er = np.linalg.norm(r_num[-1] - rT)
            ev = np.linalg.norm(v_num[-1] - vT)
            endpoint_err[m].append(er + ev)

    for m in methods:
        endpoint_err[m] = np.array(endpoint_err[m], dtype=float)

    # ---- Plot log-log ----
    plt.figure()
    for m in methods:
        plt.loglog(dts, endpoint_err[m], marker="o", label=str(m))

    # slope estimates
    for m in methods:
        slope, _ = linreg_slope_loglog(dts, endpoint_err[m])
        print(f"{m:10s}  slope p ≈ {slope:.3f}")

    plt.gca().invert_xaxis()  # dt small to the right is sometimes less intuitive; invert so refinement goes right-to-left
    plt.xlabel("Δt")
    plt.ylabel("Endpoint error ||r(T)-r*|| + ||v(T)-v*||")
    plt.title("Convergence vs Δt (rectangles vs trapezoid vs RK4)")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()


if __name__ == "__main__":
    main()
