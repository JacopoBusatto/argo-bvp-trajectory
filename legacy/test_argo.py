from __future__ import annotations

import numpy as np

from argobvp.preprocess import load_config, open_aux
from argobvp.preprocess.products import build_preprocessed_dataset
from argobvp.preprocess.io_coriolis import extract_aux_minimal, build_valid_mask


def main() -> None:
    cfg_path = "configs/4903848.yml"
    dt_max = 300.0  # seconds for "normal-step" diagnostics

    # --- load + preprocess ---
    cfg = load_config("configs/4903848.yml")
    print("cfg.imu.gyro.scale:", cfg.imu.gyro.scale)
    print("cfg.imu.mag.hard_iron:", cfg.imu.mag.hard_iron)
    print("cfg.attitude.mode:", cfg.attitude.mode)
    print("cfg.attitude.dt_max:", cfg.attitude.dt_max)
    print("cfg.attitude.alpha:", cfg.attitude.alpha)
    ds_aux = open_aux(cfg.paths.aux)
    ds_pre = build_preprocessed_dataset(ds_aux, cfg)

    # --- cfg sanity ---
    print("gyro.scale from cfg:", cfg.imu.gyro.scale)

    # --- time steps dt (seconds) ---
    t = ds_pre.time.values.astype("datetime64[ns]").astype("int64") * 1e-9
    dt = np.diff(t, prepend=t[0])

    print(
        "dt stats [s]: min/median/p95/max =",
        float(np.min(dt)),
        float(np.median(dt)),
        float(np.percentile(dt, 95)),
        float(np.max(dt)),
    )
    print("fraction dt==0:", float(np.mean(dt == 0.0)))

    # --- raw gyro counts sanity (reference only) ---
    raw = extract_aux_minimal(ds_aux)

    j = raw["JULD"]
    p = raw["PRES"]
    c = raw["CYCLE_NUMBER"]
    gx = raw["ANGULAR_RATE_COUNT_X"]
    gy = raw["ANGULAR_RATE_COUNT_Y"]
    gz = raw["ANGULAR_RATE_COUNT_Z"]

    valid = build_valid_mask(j, p, c, gx, gy, gz)

    j_arr = np.asarray(j).reshape(-1)
    if np.issubdtype(j_arr.dtype, np.datetime64):
        valid &= ~np.isnat(j_arr)
    else:
        valid &= np.isfinite(j_arr.astype(float))

    gzx = np.asarray(gz).reshape(-1)[valid].astype(float)
    print("gyro counts std (Z):", float(np.std(gzx)))

    # --- yaw stats (diagnostic only in SAFE pipeline) ---
    yaw_deg = ds_pre["yaw"].values * 180.0 / np.pi
    print("yaw [deg] min / max:", float(np.min(yaw_deg)), float(np.max(yaw_deg)))
    print("yaw std [deg]:", float(np.std(yaw_deg)))

    # --- acc stats ---
    print("std acc_lin_ned_n [m/s^2]:", float(ds_pre["acc_lin_ned_n"].std()))
    print("std acc_lin_ned_e [m/s^2]:", float(ds_pre["acc_lin_ned_e"].std()))
    print("std acc_lin_ned_d [m/s^2]:", float(ds_pre["acc_lin_ned_d"].std()))

    # --- surface gravity check ---
    surf = ds_pre["pres"] < 5

    acc_mag = np.sqrt(
        (ds_pre["acc_body_x"].values ** 2)
        + (ds_pre["acc_body_y"].values ** 2)
        + (ds_pre["acc_body_z"].values ** 2)
    )
    print("median ||acc_body|| surface [m/s^2]:", float(np.median(acc_mag[surf.values])))

    # Important SAFE check: gravity should sit in acc_ned_d at surface
    print("median acc_ned_d surface [m/s^2]:", float(ds_pre["acc_ned_d"].where(surf).median()))

    # And after gravity removal: acc_lin_ned_d should be ~0 at surface
    print("median acc_lin_ned_d surface [m/s^2]:", float(ds_pre["acc_lin_ned_d"].where(surf).median()))

    # --- small preview ---
    print(
        ds_pre[["pres", "acc_lin_ned_n", "acc_lin_ned_e", "acc_lin_ned_d"]]
        .isel(obs=slice(0, 10))
        .to_dataframe()
    )

    # --- dy stats (all steps) ---
    dy = np.diff(ds_pre["yaw"].values)
    print("median |dy| [deg/step]:", float(np.median(np.abs(dy)) * 180.0 / np.pi))
    print("p95 |dy| [deg/step]:", float(np.percentile(np.abs(dy), 95) * 180.0 / np.pi))

    # --- dy stats only for "normal" dt ---
    mask = dt[1:] <= dt_max
    dy_norm = dy[mask]
    print(f"median |dy| [deg/step] (dt<={dt_max:.0f}s):", float(np.median(np.abs(dy_norm)) * 180.0 / np.pi))
    print(f"p95 |dy| [deg/step] (dt<={dt_max:.0f}s):", float(np.percentile(np.abs(dy_norm), 95) * 180.0 / np.pi))
    print("median acc_ned_d surface:", float(ds_pre.acc_ned_d.where(ds_pre.pres < 5).median()))
    print("median acc_lin_ned_d surface:", float(ds_pre.acc_lin_ned_d.where(ds_pre.pres < 5).median()))
    print("cfg.imu.gyro.scale:", cfg.imu.gyro.scale)
    print("cfg.imu.mag.hard_iron:", cfg.imu.mag.hard_iron)
    print("cfg.attitude.mode:", getattr(cfg, "attitude", None))


if __name__ == "__main__":
    main()



