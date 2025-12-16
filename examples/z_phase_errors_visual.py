import numpy as np
import matplotlib.pyplot as plt

from argobvp.integrators import IntegratorMethod
from argobvp.z_profiles import argo_piecewise_z_profile
from argobvp.z_sources import integrate_z_from_accel_samples, integrate_z_from_accel


def phase_err_interp(t_num, z_num, t_truth, z_truth, tt):
    """Error at exact time tt using interpolation on both arrays."""
    zN = float(np.interp(float(tt), t_num, z_num))
    zT = float(np.interp(float(tt), t_truth, z_truth))
    return abs(zN - zT)


def rms_phase_error(e_dict):
    """RMS across phases (dive_end, ascent_start, final)."""
    vals = np.array([e_dict["dive_end"], e_dict["ascent_start"], e_dict["final"]], dtype=float)
    return float(np.sqrt(np.mean(vals**2)))


def main():
    # --- Non-aligned phases (avoid node degeneracy) ---
    T = 24_013.0
    t_dive_end = 3_973.0
    t_ascent_start = 18_037.0
    z_max = 1200.0

    # --- High-res truth for pretty plotting (optional, only for display) ---
    t_truth = np.linspace(0.0, T, 240_131)  # ~0.1s
    z_true, vz_true, az_true = argo_piecewise_z_profile(
        t_truth, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=T
    )

    # dt sweep (intentionally non “nice”)
    dt_list = [6.0, 13.0, 29.0, 61.0, 127.0, 263.0]

    phases = {
        "dive_end": t_dive_end,
        "ascent_start": t_ascent_start,
        "final": T,
    }

    # --- What we compare ---
    # IMU-sample-consistent comparison:
    methods_samples = [IntegratorMethod.EULER, IntegratorMethod.TRAPEZOID]

    # Optional continuous benchmark:
    include_rk4_continuous = True

    # storage
    errs_phase = {m: {k: [] for k in phases} for m in methods_samples}
    errs_rms = {m: [] for m in methods_samples}

    if include_rk4_continuous:
        errs_phase["RK4_continuous"] = {k: [] for k in phases}
        errs_rms["RK4_continuous"] = []

    # pick one dt for time-series plots
    dt_show = 61.0

    for dt in dt_list:
        N = int(np.floor(T / dt))
        t = np.linspace(0.0, N * dt, N + 1)
        T_eff = float(t[-1])

        # --- IMPORTANT: build truth ON THIS GRID (no interp artefacts) ---
        z_ref, vz_ref, az_ref = argo_piecewise_z_profile(
            t, z_max=z_max, t_dive_end=t_dive_end, t_ascent_start=t_ascent_start, T=T_eff
        )
        z0 = float(z_ref[0])
        vz0 = float(vz_ref[0])

        # --- Integrate using SAMPLES mode (fair rectangle vs trapezoid) ---
        sols = {}
        for m in methods_samples:
            z_m, _ = integrate_z_from_accel_samples(t, az_ref, z0=z0, vz0=vz0, method=m)
            sols[m] = z_m

            e = {}
            for name, tt in phases.items():
                tt_eff = min(float(tt), T_eff)
                e[name] = phase_err_interp(t, z_m, t, z_ref, tt_eff)

            for name in phases:
                errs_phase[m][name].append(e[name])
            errs_rms[m].append(rms_phase_error(e))

        # --- Optional RK4 "continuous forcing" benchmark ---
        if include_rk4_continuous:
            z_rk4, _ = integrate_z_from_accel(t, az_ref, z0=z0, vz0=vz0, method=IntegratorMethod.RK4)
            sols["RK4_continuous"] = z_rk4

            e = {}
            for name, tt in phases.items():
                tt_eff = min(float(tt), T_eff)
                e[name] = phase_err_interp(t, z_rk4, t, z_ref, tt_eff)

            for name in phases:
                errs_phase["RK4_continuous"][name].append(e[name])
            errs_rms["RK4_continuous"].append(rms_phase_error(e))

        # --- Time-series plots for one representative dt ---
        if abs(dt - dt_show) < 1e-9:
            # z(t)
            plt.figure()
            plt.plot(t_truth / 3600.0, z_true, linewidth=2, label="truth (fine)")
            plt.plot(t / 3600.0, z_ref, linewidth=2, linestyle="--", label=f"truth (on grid) dt={dt_show}s")

            for m in methods_samples:
                plt.plot(t / 3600.0, sols[m], label=str(m) + " (samples)")
            if include_rk4_continuous:
                plt.plot(t / 3600.0, sols["RK4_continuous"], label="RK4 (continuous)", linestyle=":")

            for name, tt in phases.items():
                tt_eff = min(float(tt), T_eff)
                plt.axvline(tt_eff / 3600.0, linestyle="--")

            plt.xlabel("time (hours)")
            plt.ylabel("z (m, positive downward)")
            plt.title("z(t) from a_z(t): samples vs continuous integration")
            plt.grid(True)
            plt.legend()

            # |error(t)|
            plt.figure()
            for m in methods_samples:
                e_t = np.abs(sols[m] - z_ref)
                plt.plot(t / 3600.0, e_t, label=str(m) + " (samples)")
            if include_rk4_continuous:
                e_t = np.abs(sols["RK4_continuous"] - z_ref)
                plt.plot(t / 3600.0, e_t, label="RK4 (continuous)", linestyle=":")

            for name, tt in phases.items():
                tt_eff = min(float(tt), T_eff)
                plt.axvline(tt_eff / 3600.0, linestyle="--")

            plt.xlabel("time (hours)")
            plt.ylabel("|z - z_truth(grid)| (m)")
            plt.title("Absolute error in z(t) on the grid")
            plt.grid(True)
            plt.legend()

    # --- Phase errors vs dt (log-log) ---
    for name in phases:
        plt.figure()
        for m in methods_samples:
            plt.loglog(dt_list, errs_phase[m][name], marker="o", label=str(m) + " (samples)")
        if include_rk4_continuous:
            plt.loglog(dt_list, errs_phase["RK4_continuous"][name], marker="o", label="RK4 (continuous)")

        plt.xlabel("dt (s)")
        plt.ylabel(f"|z(t_{name}) - z_truth| (m)")
        plt.title(f"Phase error vs dt — {name}")
        plt.grid(True, which="both")
        plt.legend()

    # --- RMS over phases vs dt (log-log) ---
    plt.figure()
    for m in methods_samples:
        plt.loglog(dt_list, errs_rms[m], marker="o", label=str(m) + " (samples)")
    if include_rk4_continuous:
        plt.loglog(dt_list, errs_rms["RK4_continuous"], marker="o", label="RK4 (continuous)")
    plt.xlabel("dt (s)")
    plt.ylabel("RMS phase error (m)")
    plt.title("Aggregate phase error (RMS over phases) vs dt")
    plt.grid(True, which="both")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
