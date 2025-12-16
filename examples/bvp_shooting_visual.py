import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import integrate_2nd_order, IntegratorMethod
from argobvp.bvp import shoot_v0_to_hit_rT


# -----------------------------
# "Truth" analytic XY (non-degenerate)
# -----------------------------
def xy_truth(t, T):
    """
    Non-degenerate 2D truth:
    - drift in BOTH x and y
    - low-frequency meander (cycle-scale)
    - higher-frequency oscillation
    """
    t = np.asarray(t, dtype=float)

    # Mean drift (both components!)
    U = np.array([0.18, -0.12])  # m/s (arbitrary units ok)

    # Low-frequency (cycle-scale)
    wL = 2 * np.pi / (1.5 * T)
    phiL = 0.2
    A_L = np.array([600.0, 400.0])  # meters amplitude

    # Higher-frequency
    wH = 2 * np.pi / 2500.0
    phiHx = 0.4
    phiHy = 1.1
    A_H = np.array([80.0, -120.0])

    x = U[0] * t + A_L[0] * np.sin(wL * t) + A_H[0] * np.sin(wH * t + phiHx)
    y = U[1] * t + A_L[1] * np.cos(wL * t + phiL) + A_H[1] * np.cos(wH * t + phiHy)
    return np.stack([x, y], axis=1)


def xy_v_truth(t, T):
    t = np.asarray(t, dtype=float)

    U = np.array([0.18, -0.12])

    wL = 2 * np.pi / (1.5 * T)
    phiL = 0.2
    A_L = np.array([600.0, 400.0])

    wH = 2 * np.pi / 2500.0
    phiHx = 0.4
    phiHy = 1.1
    A_H = np.array([80.0, -120.0])

    vx = U[0] + A_L[0] * wL * np.cos(wL * t) + A_H[0] * wH * np.cos(wH * t + phiHx)
    vy = U[1] - A_L[1] * wL * np.sin(wL * t + phiL) - A_H[1] * wH * np.sin(wH * t + phiHy)
    return np.stack([vx, vy], axis=1)


def xy_a_truth(t, T):
    t = np.asarray(t, dtype=float)

    wL = 2 * np.pi / (1.5 * T)
    phiL = 0.2
    A_L = np.array([600.0, 400.0])

    wH = 2 * np.pi / 2500.0
    phiHx = 0.4
    phiHy = 1.1
    A_H = np.array([80.0, -120.0])

    ax = -A_L[0] * (wL**2) * np.sin(wL * t) - A_H[0] * (wH**2) * np.sin(wH * t + phiHx)
    ay = -A_L[1] * (wL**2) * np.cos(wL * t + phiL) - A_H[1] * (wH**2) * np.cos(wH * t + phiHy)
    return np.stack([ax, ay], axis=1)


def main():
    # -----------------------------
    # Setup
    # -----------------------------
    T = 12_000.0

    # Fine grid only for plotting the truth curve nicely
    t_fine = np.linspace(0.0, T, 120_000 + 1)

    xy_fine = xy_truth(t_fine, T)
    r0 = np.array([xy_fine[0, 0], xy_fine[0, 1], 0.0])

    # True v0 from analytic derivative at t=0
    v0_true_xy = xy_v_truth(np.array([0.0]), T)[0]
    v0_true = np.array([v0_true_xy[0], v0_true_xy[1], 0.0])

    # Analytic acceleration function (consistent with the truth)
    def a_true(ti, r, v):
        axy = xy_a_truth(np.array([float(ti)]), T)[0]
        return np.array([axy[0], axy[1], 0.0], dtype=float)

    # Endpoint target from integrating with true v0 (very accurate)
    r_truth_fine, v_truth_fine = integrate_2nd_order(
        t=t_fine,
        r0=r0,
        v0=v0_true,
        a_fun=a_true,
        method=IntegratorMethod.RK4,
        backward=False,
    )
    rT_target = r_truth_fine[-1].copy()

    # -----------------------------
    # Coarser grid used in the inverse problem (shooting)
    # -----------------------------
    N = 2000  # dt ~ 6 s
    t = np.linspace(0.0, T, N + 1)

    # Reasonable guess from endpoints (mean velocity)
    v0_guess = np.array([(rT_target[0] - r0[0]) / T, (rT_target[1] - r0[1]) / T, 0.0])

    # Solve BVP by shooting, compare integrators
    methods = [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID, IntegratorMethod.RK4]
    sols = {}

    print("\n--- Shooting results ---")
    print(f"v0_true_xy = {v0_true[:2]}")
    print(f"v0_guess_xy= {v0_guess[:2]}\n")

    for m in methods:
        res = shoot_v0_to_hit_rT(
            t=t,
            r0=r0,
            rT_target=rT_target,
            v0_guess=v0_guess,
            a_fun=a_true,
            method=m,
            dims=(0, 1),
        )
        sols[m] = res
        print(f"{m:10s} success={res.success}  v0_opt_xy={res.v0_opt[:2]}  nfev={res.nfev}")

    # Truth on the coarse grid (for error vs time)
    r_truth_on_t, _ = integrate_2nd_order(
        t=t, r0=r0, v0=v0_true, a_fun=a_true, method=IntegratorMethod.RK4, backward=False
    )

    # -----------------------------
    # Plot 1: XY trajectories
    # -----------------------------
    plt.figure()
    plt.plot(r_truth_fine[:, 0], r_truth_fine[:, 1], label="truth (fine)", linewidth=2)

    for m in methods:
        r_m = sols[m].r
        plt.plot(r_m[:, 0], r_m[:, 1], label=f"shooting + {m}")

    plt.scatter([r0[0], rT_target[0]], [r0[1], rT_target[1]], s=40, label="boundary points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BVP via shooting on v0_xy: trajectories (non-degenerate truth)")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # Plot 2: Error vs time (XY)
    # -----------------------------
    plt.figure()
    for m in methods:
        r_m = sols[m].r
        e_xy = np.linalg.norm(r_m[:, :2] - r_truth_on_t[:, :2], axis=1)
        plt.plot(t / 3600.0, e_xy, label=str(m))

    plt.xlabel("time (hours)")
    plt.ylabel("||r_xy - r*_xy||")
    plt.title("Reconstruction error over time (XY)")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # Plot 3: Cost surface J(v0x,v0y)
    # -----------------------------
    # Use a cheaper time grid for cost surface only (makes it MUCH faster)
    N_cost = 300
    t_cost = np.linspace(0.0, T, N_cost + 1)

    # Center the window on the optimum so we don't waste space
    vxc = sols[IntegratorMethod.TRAPEZOID].v0_opt[0]
    vyc = sols[IntegratorMethod.TRAPEZOID].v0_opt[1]

    # Make a window around the guess (tune if needed)
    dvx = 0.12
    dvy = 0.12
    n = 31

    vx = np.linspace(vxc - dvx, vxc + dvx, n)
    vy = np.linspace(vyc - dvy, vyc + dvy, n)
    J = np.zeros((n, n), dtype=float)

    # Objective model for the cost surface (use trapezoid)
    for i, vxi in enumerate(vx):
        for j, vyj in enumerate(vy):
            v0_try = np.array([vxi, vyj, 0.0])
            r_try, _ = integrate_2nd_order(
                t=t_cost,
                r0=r0,
                v0=v0_try,
                a_fun=a_true,
                method=IntegratorMethod.TRAPEZOID,
                backward=False,
            )
            J[j, i] = np.linalg.norm(r_try[-1, :2] - rT_target[:2])

    plt.figure()
    plt.imshow(
        J,
        origin="lower",
        extent=[vx.min(), vx.max(), vy.min(), vy.max()],
        aspect="auto",
    )
    plt.colorbar(label="J = ||r_xy(T; v0) - rT_xy||")

    plt.scatter([v0_true[0]], [v0_true[1]], marker="x", s=80, label="v0_true")
    plt.scatter(
        [sols[IntegratorMethod.TRAPEZOID].v0_opt[0]],
        [sols[IntegratorMethod.TRAPEZOID].v0_opt[1]],
        marker="o",
        s=60,
        label="v0_opt (trapezoid)",
    )

    plt.xlabel("v0x")
    plt.ylabel("v0y")
    plt.title("Cost surface for shooting (trapezoid objective) — non-degenerate")
    plt.legend()
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    main()
