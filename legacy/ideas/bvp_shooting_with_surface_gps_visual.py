import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import integrate_2nd_order, IntegratorMethod
from argobvp.bvp import shoot_v0_to_hit_rT
from ideas.estimation import estimate_v0_from_surface_gps


# --- same non-degenerate truth as before ---
def xy_truth(t, T):
    t = np.asarray(t, float)
    U = np.array([0.18, -0.12])
    wL = 2 * np.pi / (1.5 * T)
    phiL = 0.2
    A_L = np.array([600.0, 400.0])
    wH = 2 * np.pi / 2500.0
    phiHx = 0.4
    phiHy = 1.1
    A_H = np.array([80.0, -120.0])
    x = U[0] * t + A_L[0] * np.sin(wL * t) + A_H[0] * np.sin(wH * t + phiHx)
    y = U[1] * t + A_L[1] * np.cos(wL * t + phiL) + A_H[1] * np.cos(wH * t + phiHy)
    return np.stack([x, y], axis=1)


def xy_v_truth(t, T):
    t = np.asarray(t, float)
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
    t = np.asarray(t, float)
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
    T = 12_000.0

    # --- "true" continuous acceleration model
    def a_true(ti, r, v):
        ax, ay = xy_a_truth(np.array([float(ti)]), T)[0]
        return np.array([ax, ay, 0.0], float)

    # --- Truth on fine grid (for plotting)
    t_fine = np.linspace(0.0, T, 120_000 + 1)
    xy_fine = xy_truth(t_fine, T)
    r0 = np.array([xy_fine[0, 0], xy_fine[0, 1], 0.0])

    v0_true_xy = xy_v_truth(np.array([0.0]), T)[0]
    v0_true = np.array([v0_true_xy[0], v0_true_xy[1], 0.0])

    r_truth_fine, _ = integrate_2nd_order(t_fine, r0, v0_true, a_true, method=IntegratorMethod.RK4)
    rT_target = r_truth_fine[-1].copy()

    # -----------------------------
    # TOY SURFACE GPS FIXES (before diving)
    # -----------------------------
    # simulate a short surface window: last 30 minutes before t=0 (we set t=0 as dive start)
    # We'll generate "surface fixes" around t in [-1800, 0] and add small GPS noise.
    t_surf = np.linspace(-1800.0, 0.0, 13)  # every 150 s
    xy_surf_true = xy_truth(t_surf, T)
    rng = np.random.default_rng(0)
    gps_noise = 2.0  # meters
    xy_surf_obs = xy_surf_true + rng.normal(0.0, gps_noise, size=xy_surf_true.shape)

    v0_surf_xy, info = estimate_v0_from_surface_gps(t_surf, xy_surf_obs[:, 0], xy_surf_obs[:, 1])
    print("\n--- Surface GPS estimate ---")
    print("v0_surf_xy =", v0_surf_xy, "  rms_xy(m)=", info["rms_xy"], "  n=", info["n"], "  span(s)=", info["t_span"])

    # -----------------------------
    # BVP SHOOTING (use v0_surf as guess)
    # -----------------------------
    # Coarse inverse grid
    t = np.linspace(0.0, T, 2000 + 1)
    v0_guess = np.array([v0_surf_xy[0], v0_surf_xy[1], 0.0])

    res = shoot_v0_to_hit_rT(
        t=t,
        r0=r0,
        rT_target=rT_target,
        v0_guess=v0_guess,
        a_fun=a_true,
        method=IntegratorMethod.TRAPEZOID,
        dims=(0, 1),
    )
    print("\n--- Shooting ---")
    print("success:", res.success, "nfev:", res.nfev, "msg:", res.message)
    print("v0_true_xy =", v0_true[:2])
    print("v0_guess_xy=", v0_guess[:2], "(from surface GPS)")
    print("v0_opt_xy  =", res.v0_opt[:2])

    # Truth on coarse grid (for time error)
    r_truth_on_t, _ = integrate_2nd_order(t, r0, v0_true, a_true, method=IntegratorMethod.RK4)

    # -----------------------------
    # PLOT A: XY trajectories
    # -----------------------------
    plt.figure()
    plt.plot(r_truth_fine[:, 0], r_truth_fine[:, 1], label="truth (fine)", linewidth=2)
    plt.plot(res.r[:, 0], res.r[:, 1], label="BVP shooting (trapezoid)", linewidth=2)
    plt.scatter([r0[0], rT_target[0]], [r0[1], rT_target[1]], s=40, label="boundary points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("BVP trajectory (using surface-GPS v0 as guess)")
    plt.legend()
    plt.grid(True)

    # -----------------------------
    # PLOT B: Error vs time (XY)
    # -----------------------------
    plt.figure()
    e_xy = np.linalg.norm(res.r[:, :2] - r_truth_on_t[:, :2], axis=1)
    plt.plot(t / 3600.0, e_xy, label="||r_rec - r_true|| (XY)")
    plt.xlabel("time (hours)")
    plt.ylabel("error")
    plt.title("Reconstruction error over time (XY)")
    plt.grid(True)
    plt.legend()

    # -----------------------------
    # PLOT C: Compare velocity arrows at start
    # -----------------------------
    plt.figure()
    plt.scatter([0], [0], s=30)
    scale = 1.0
    plt.quiver(
        0, 0, v0_true[0], v0_true[1],
        angles="xy", scale_units="xy", scale=scale, label="v0_true"
    )
    plt.quiver(
        0, 0, v0_guess[0], v0_guess[1],
        angles="xy", scale_units="xy", scale=scale, label="v0_surf (guess)"
    )
    plt.quiver(
        0, 0, res.v0_opt[0], res.v0_opt[1],
        angles="xy", scale_units="xy", scale=scale, label="v0_opt (shoot)"
    )
    plt.xlabel("vx")
    plt.ylabel("vy")
    plt.title("Initial velocity comparison (XY)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    # -----------------------------
    # PLOT D: Cost surface centered on v0_opt (FAST + always includes minimum)
    # -----------------------------
    vxc, vyc = res.v0_opt[0], res.v0_opt[1]

    dvx = 0.12
    dvy = 0.12
    n = 41  # much faster than 81

    vx = np.linspace(vxc - dvx, vxc + dvx, n)
    vy = np.linspace(vyc - dvy, vyc + dvy, n)
    J = np.zeros((n, n), float)

    for i, vxi in enumerate(vx):
        for j, vyj in enumerate(vy):
            v0_try = np.array([vxi, vyj, 0.0])
            r_try, _ = integrate_2nd_order(t, r0, v0_try, a_true, method=IntegratorMethod.TRAPEZOID)
            J[j, i] = np.linalg.norm(r_try[-1, :2] - rT_target[:2])

    plt.figure()
    plt.imshow(
        J, origin="lower",
        extent=[vx.min(), vx.max(), vy.min(), vy.max()],
        aspect="auto",
    )
    plt.colorbar(label="J = ||r_xy(T; v0) - rT_xy||")
    plt.scatter([v0_true[0]], [v0_true[1]], marker="x", s=80, label="v0_true")
    plt.scatter([v0_guess[0]], [v0_guess[1]], marker="^", s=70, label="v0_surf guess")
    plt.scatter([res.v0_opt[0]], [res.v0_opt[1]], marker="o", s=70, label="v0_opt")
    plt.xlabel("v0x")
    plt.ylabel("v0y")
    plt.title("Cost surface (centered on v0_opt) — fast + always in-window")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
