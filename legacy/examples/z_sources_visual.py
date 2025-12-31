import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import IntegratorMethod
from argobvp.z_profiles import argo_piecewise_z_profile
from argobvp.z_sources import build_z_from_pressure, integrate_z_from_accel


def main():
    T = 24_000.0
    t_dive_end = 4_000.0
    t_ascent_start = 18_000.0
    z_max = 1200.0
    rho, g = 1025.0, 9.81

    # "truth"
    t_truth = np.linspace(0.0, T, 240_001)
    z_true, vz_true, az_true = argo_piecewise_z_profile(
        t_truth, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=T
    )
    p_true = rho * g * z_true
    z_p_truth = build_z_from_pressure(t_truth, p_true, rho=rho, g=g, z0=z_true[0])

    # compare a few dt
    dt_list = [10.0, 30.0, 60.0]
    methods = [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID, IntegratorMethod.RK4]

    plt.figure()
    plt.plot(t_truth / 3600.0, z_true, linewidth=2, label="z truth")
    plt.plot(t_truth / 3600.0, z_p_truth, linestyle="--", label="z from pressure (truth)")
    plt.xlabel("time (hours)")
    plt.ylabel("z (m, positive downward)")
    plt.title("Argo-like piecewise z(t): truth and pressure-derived")
    plt.grid(True)
    plt.legend()

    # z reconstructions from accel
    for dt in dt_list:
        N = int(T / dt)
        t = np.linspace(0.0, T, N + 1)
        az_s = np.interp(t, t_truth, az_true)
        z_ref = np.interp(t, t_truth, z_true)
        z0 = float(z_ref[0])
        vz0 = float(np.interp(0.0, t_truth, vz_true))

        plt.figure()
        plt.plot(t / 3600.0, z_ref, linewidth=2, label=f"truth (sampled) dt={dt}s")

        for m in methods:
            z_m, vz_m = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=m)
            plt.plot(t / 3600.0, z_m, label=f"accel integ {m}")

        plt.xlabel("time (hours)")
        plt.ylabel("z (m, positive downward)")
        plt.title(f"z(t) from accel integration — dt={dt}s")
        plt.grid(True)
        plt.legend()

        # error vs time
        plt.figure()
        for m in methods:
            z_m, _ = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=m)
            e = np.abs(z_m - z_ref)
            plt.plot(t / 3600.0, e, label=f"{m}")

        plt.xlabel("time (hours)")
        plt.ylabel("|z - z_truth| (m)")
        plt.title(f"Absolute error in z(t) — dt={dt}s")
        plt.grid(True)
        plt.legend()

    # endpoint error vs dt (log-log)
    plt.figure()
    for m in methods:
        errs = []
        for dt in dt_list:
            N = int(T / dt)
            t = np.linspace(0.0, T, N + 1)
            az_s = np.interp(t, t_truth, az_true)
            z_ref = np.interp(t, t_truth, z_true)
            z0 = float(z_ref[0])
            vz0 = float(np.interp(0.0, t_truth, vz_true))

            z_m, _ = integrate_z_from_accel(t, az_s, z0=z0, vz0=vz0, method=m)
            errs.append(abs(z_m[-1] - z_ref[-1]))

        plt.loglog(dt_list, errs, marker="o", label=str(m))

    plt.xlabel("dt (s)")
    plt.ylabel("|z(T) - z_truth(T)| (m)")
    plt.title("Endpoint error vs dt (z integration)")
    plt.grid(True, which="both")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
