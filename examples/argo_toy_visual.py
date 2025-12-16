import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import integrate_2nd_order, IntegratorMethod


# -----------------------------
# 1) Z profile (imposed)
# -----------------------------
def z_profile(t, T, t_dive_end, t_ascent_start, z_park=-1000.0):
    """
    Piecewise linear z(t):
      - descent: 0 -> z_park
      - park:    z_park constant
      - ascent:  z_park -> 0
    z is negative downward (oceanographic convention).
    """
    t = np.asarray(t, dtype=float)
    z = np.zeros_like(t)

    # descent
    m1 = t <= t_dive_end
    z[m1] = (z_park / t_dive_end) * t[m1]

    # parking
    m2 = (t > t_dive_end) & (t <= t_ascent_start)
    z[m2] = z_park

    # ascent
    m3 = t > t_ascent_start
    z[m3] = z_park * (1.0 - (t[m3] - t_ascent_start) / (T - t_ascent_start))

    return z


# -----------------------------
# 2) XY truth (analytic)
# -----------------------------
def xy_truth(t):
    """
    Smooth 2D 'true' trajectory with drift + oscillations.
    Units are arbitrary (think meters if you want).
    """
    t = np.asarray(t, dtype=float)
    # drift
    U = np.array([0.18, -0.06])  # mean velocity
    # oscillations
    w1 = 2 * np.pi / 8000.0
    w2 = 2 * np.pi / 2500.0
    A1 = np.array([120.0, -80.0])
    A2 = np.array([40.0, 30.0])

    x = U[0] * t + A1[0] * np.sin(w1 * t) + A2[0] * np.sin(w2 * t + 0.7)
    y = U[1] * t + A1[1] * np.cos(w1 * t) + A2[1] * np.cos(w2 * t + 0.2)
    return np.stack([x, y], axis=1)


def xy_v_truth(t):
    t = np.asarray(t, dtype=float)
    U = np.array([0.18, -0.06])
    w1 = 2 * np.pi / 8000.0
    w2 = 2 * np.pi / 2500.0
    A1 = np.array([120.0, -80.0])
    A2 = np.array([40.0, 30.0])

    vx = U[0] + A1[0] * w1 * np.cos(w1 * t) + A2[0] * w2 * np.cos(w2 * t + 0.7)
    vy = U[1] - A1[1] * w1 * np.sin(w1 * t) - A2[1] * w2 * np.sin(w2 * t + 0.2)
    return np.stack([vx, vy], axis=1)


def xy_a_truth(t):
    t = np.asarray(t, dtype=float)
    w1 = 2 * np.pi / 8000.0
    w2 = 2 * np.pi / 2500.0
    A1 = np.array([120.0, -80.0])
    A2 = np.array([40.0, 30.0])

    ax = -A1[0] * (w1**2) * np.sin(w1 * t) - A2[0] * (w2**2) * np.sin(w2 * t + 0.7)
    ay = -A1[1] * (w1**2) * np.cos(w1 * t) - A2[1] * (w2**2) * np.cos(w2 * t + 0.2)
    return np.stack([ax, ay], axis=1)


# -----------------------------
# 3) Build sampled a(t) and interpolate linearly (Argo-like)
# -----------------------------
def make_linear_interp_accel(t_samples, a_samples):
    t_samples = np.asarray(t_samples, dtype=float)
    a_samples = np.asarray(a_samples, dtype=float)
    ax, ay = a_samples[:, 0], a_samples[:, 1]

    def a_fun(ti, r, v):
        ti = float(ti)
        ax_i = np.interp(ti, t_samples, ax)
        ay_i = np.interp(ti, t_samples, ay)
        return np.array([ax_i, ay_i, 0.0], dtype=float)  # az ignored (z imposed)

    return a_fun


# -----------------------------
# 4) Non-uniform time grid generator
# -----------------------------
def make_time_grid(T, dt_nominal, jitter=0.25, seed=0):
    """
    dt is non-uniform: dt_k = dt_nominal * (1 + jitter * uniform(-1,1))
    """
    rng = np.random.default_rng(seed)
    t = [0.0]
    while t[-1] < T:
        fac = 1.0 + jitter * (2.0 * rng.random() - 1.0)
        dt = max(1e-6, dt_nominal * fac)
        t.append(t[-1] + dt)
    t = np.array(t)
    t[-1] = T  # clamp exact end
    return t


def main():
    # --- Cycle times (seconds) ---
    T = 24_000.0
    t_dive_end = 4_000.0
    t_ascent_start = 18_000.0

    # --- "Truth" on a fine grid (for plotting reference) ---
    t_true = np.linspace(0.0, T, 120_000 + 1)  # dt=0.2 s
    xy_true = xy_truth(t_true)
    z_true = z_profile(t_true, T, t_dive_end, t_ascent_start, z_park=-1000.0)
    r_true_3d = np.column_stack([xy_true, z_true])

    # initial conditions from truth
    r0 = np.array([xy_true[0, 0], xy_true[0, 1], z_true[0]])
    v0_xy = xy_v_truth(np.array([0.0]))[0]
    v0 = np.array([v0_xy[0], v0_xy[1], 0.0])  # vz ignored

    # --- Acceleration samples (simulate instrument sampling) ---
    dt_acc = 60.0  # "acceleration measurement" every 60 s
    t_acc = np.arange(0.0, T + 1e-9, dt_acc)
    a_acc_xy = xy_a_truth(t_acc)  # exact accel at sample times
    a_fun = make_linear_interp_accel(t_acc, a_acc_xy)

    # --- Compare several integration dt (nominal) ---
    dt_list = [30.0, 60.0, 120.0, 300.0]  # seconds
    methods = [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID, IntegratorMethod.RK4]

    # Store endpoint errors and phase errors
    stats = {}

    for dt_nom in dt_list:
        # non-uniform integration grid (Argo-like irregular timing)
        t = make_time_grid(T, dt_nominal=dt_nom, jitter=0.30, seed=1)

        # truth sampled on this grid (for error computation)
        xy_t = xy_truth(t)
        z_t = z_profile(t, T, t_dive_end, t_ascent_start, z_park=-1000.0)
        r_t = np.column_stack([xy_t, z_t])

        # indices of key phases on integration grid
        i_dive_end = int(np.argmin(np.abs(t - t_dive_end)))
        i_ascent_start = int(np.argmin(np.abs(t - t_ascent_start)))

        stats[dt_nom] = {}

        for m in methods:
            r_num, v_num = integrate_2nd_order(
                t=t, r0=r0, v0=v0, a_fun=a_fun, method=m, backward=False
            )

            # impose z profile after integration
            r_num[:, 2] = z_t

            # errors
            e_time = np.linalg.norm(r_num - r_t, axis=1)
            e_final = float(np.linalg.norm(r_num[-1] - r_t[-1]))
            e_dive_end = float(np.linalg.norm(r_num[i_dive_end] - r_t[i_dive_end]))
            e_ascent_start = float(np.linalg.norm(r_num[i_ascent_start] - r_t[i_ascent_start]))

            stats[dt_nom][m] = dict(
                t=t, r_num=r_num, e_time=e_time,
                e_final=e_final, e_dive_end=e_dive_end, e_ascent_start=e_ascent_start
            )

    # -----------------------------
    # PLOTS
    # -----------------------------

    # (A) 3D trajectory for one dt
    dt_show = dt_list[-1]  # show the coarsest
    plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot(r_true_3d[:, 0], r_true_3d[:, 1], r_true_3d[:, 2], label="truth (fine)")

    for m in methods:
        ax.plot(
            stats[dt_show][m]["r_num"][:, 0],
            stats[dt_show][m]["r_num"][:, 1],
            stats[dt_show][m]["r_num"][:, 2],
            label=f"{m} (dt~{dt_show}s)"
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Toy Argo cycle: 3D trajectories (z imposed)")
    ax.legend()

    # (B) Error vs time for each method at different dt
    for dt_nom in dt_list:
        plt.figure()
        for m in methods:
            t = stats[dt_nom][m]["t"]
            e = stats[dt_nom][m]["e_time"]
            plt.plot(t / 3600.0, e, label=str(m))
        plt.axvline(t_dive_end / 3600.0, linestyle="--")
        plt.axvline(t_ascent_start / 3600.0, linestyle="--")
        plt.xlabel("time (hours)")
        plt.ylabel("||r_num - r_truth||")
        plt.title(f"Error over time (dt~{dt_nom}s, non-uniform grid)")
        plt.legend()
        plt.grid(True)

    # (C) Phase/endpoint errors vs dt (log-log)
    plt.figure()
    for m in methods:
        dts = np.array(dt_list, dtype=float)
        eF = np.array([stats[dt][m]["e_final"] for dt in dt_list], dtype=float)
        eD = np.array([stats[dt][m]["e_dive_end"] for dt in dt_list], dtype=float)
        eA = np.array([stats[dt][m]["e_ascent_start"] for dt in dt_list], dtype=float)

        plt.loglog(dts, eF, marker="o", label=f"{m} final")
        plt.loglog(dts, eD, marker="o", linestyle="--", label=f"{m} dive_end")
        plt.loglog(dts, eA, marker="o", linestyle=":", label=f"{m} ascent_start")

    plt.gca().invert_xaxis()
    plt.xlabel("dt nominal (s)")
    plt.ylabel("error norm")
    plt.title("Phase and endpoint errors vs dt (z imposed)")
    plt.legend()
    plt.grid(True, which="both")

    plt.show()


if __name__ == "__main__":
    main()
