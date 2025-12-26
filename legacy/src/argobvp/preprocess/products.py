import numpy as np
import xarray as xr

from .config import PreprocessConfig
from .io_coriolis import extract_aux_minimal, build_valid_mask
from .imu_calib import calibrate_accel_counts, calibrate_gyro_counts, calibrate_mag_counts
from .attitude import (
    roll_pitch_from_acc,
    r_body_to_ned_from_tilt,
    yaw_from_mag_simple,
)


def _juld_to_datetime64(juld: np.ndarray) -> np.ndarray:
    """
    Accepts numeric JULD (days since 1950-01-01) or datetime64.
    Returns datetime64[ns].
    """
    juld = np.asarray(juld).reshape(-1)

    if np.issubdtype(juld.dtype, np.datetime64):
        return juld.astype("datetime64[ns]")

    base = np.datetime64("1950-01-01T00:00:00", "ns")
    seconds = (juld.astype(float) * 86400.0).round().astype("int64")
    return base + seconds.astype("timedelta64[s]")


# ------------------------------------------------------------
# MAIN PRODUCT
# ------------------------------------------------------------

def build_preprocessed_dataset(ds_aux: xr.Dataset, cfg: PreprocessConfig) -> xr.Dataset:

    raw = extract_aux_minimal(ds_aux)

    juld = raw["JULD"]
    pres = raw["PRES"]
    cycle = raw["CYCLE_NUMBER"]
    mcode = raw["MEASUREMENT_CODE"]  # <-- aggiunto

    axc = raw["LINEAR_ACCELERATION_COUNT_X"]
    ayc = raw["LINEAR_ACCELERATION_COUNT_Y"]
    azc = raw["LINEAR_ACCELERATION_COUNT_Z"]

    gxc = raw["ANGULAR_RATE_COUNT_X"]
    gyc = raw["ANGULAR_RATE_COUNT_Y"]
    gzc = raw["ANGULAR_RATE_COUNT_Z"]

    mxc = raw["MAGNETIC_FIELD_COUNT_X"]
    myc = raw["MAGNETIC_FIELD_COUNT_Y"]
    mzc = raw["MAGNETIC_FIELD_COUNT_Z"]

    valid = build_valid_mask(
        juld, pres, cycle, mcode,
        axc, ayc, azc,
        gxc, gyc, gzc,
        mxc, myc, mzc
    )

    juld = juld[valid]
    pres = pres[valid]
    cycle = cycle[valid].astype(int)
    mcode = mcode[valid]

    # measurement_code è float64 nel netcdf: lo portiamo a int quando sensato
    # (ma lasciamo -1 se manca/NaN, anche se qui NaN è già escluso dal valid)
    mcode_int = np.asarray(np.round(mcode), dtype=int)

    counts_acc = {"X": axc[valid], "Y": ayc[valid], "Z": azc[valid]}
    counts_gyro = {"X": gxc[valid], "Y": gyc[valid], "Z": gzc[valid]}
    counts_mag = {"X": mxc[valid], "Y": myc[valid], "Z": mzc[valid]}

    # ------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------

    acc_g = calibrate_accel_counts(counts_acc, cfg.imu.accel)
    acc_body = {k: v * cfg.imu.g for k, v in acc_g.items()}  # m/s^2

    gyro_body = calibrate_gyro_counts(counts_gyro, cfg.imu.gyro)  # rad/s (or as cfg)
    mag_body = calibrate_mag_counts(counts_mag, cfg.imu.mag)

    # ------------------------------------------------------------
    # ATTITUDE (SAFE) -- allow identity path for synthetic data
    # ------------------------------------------------------------
    if str(cfg.platform).startswith("SYNTHETIC"):
        roll = np.zeros_like(acc_body["x"])
        pitch = np.zeros_like(acc_body["x"])
        yaw = np.zeros_like(acc_body["x"])
        a_ned = np.stack([acc_body["x"], acc_body["y"], acc_body["z"]], axis=1)
        gvec = np.array([0.0, 0.0, cfg.imu.g])
        a_lin_ned = a_ned - gvec[None, :]
    else:
        roll, pitch = roll_pitch_from_acc(acc_body)
        yaw = yaw_from_mag_simple(mag_body)
        R = r_body_to_ned_from_tilt(roll, pitch)
        a_body_vec = np.stack(
            [acc_body["x"], acc_body["y"], acc_body["z"]],
            axis=1
        )
        a_ned = np.einsum("nij,nj->ni", R, a_body_vec)
        gvec = np.array([0.0, 0.0, cfg.imu.g])
        a_lin_ned = a_ned - gvec[None, :]

    time = _juld_to_datetime64(juld)

    # ------------------------------------------------------------
    # Dataset (continuous)
    # ------------------------------------------------------------

    ds = xr.Dataset(
        coords=dict(
            obs=np.arange(time.size),
            time=("obs", time),
        ),
        data_vars=dict(
            juld=("obs", juld),
            pres=("obs", pres),
            cycle_number=("obs", cycle),
            measurement_code=("obs", mcode_int),  # <-- aggiunto

            acc_body_x=("obs", acc_body["x"]),
            acc_body_y=("obs", acc_body["y"]),
            acc_body_z=("obs", acc_body["z"]),

            acc_ned_n=("obs", a_ned[:, 0]),
            acc_ned_e=("obs", a_ned[:, 1]),
            acc_ned_d=("obs", a_ned[:, 2]),

            acc_lin_ned_n=("obs", a_lin_ned[:, 0]),
            acc_lin_ned_e=("obs", a_lin_ned[:, 1]),
            acc_lin_ned_d=("obs", a_lin_ned[:, 2]),

            gyro_body_x=("obs", gyro_body["x"]),
            gyro_body_y=("obs", gyro_body["y"]),
            gyro_body_z=("obs", gyro_body["z"]),

            mag_body_x=("obs", mag_body["x"]),
            mag_body_y=("obs", mag_body["y"]),
            mag_body_z=("obs", mag_body["z"]),

            roll=("obs", roll),
            pitch=("obs", pitch),
            yaw=("obs", yaw),  # diagnostic
        ),
        attrs=dict(
            platform=cfg.platform,
            frame="NED",
            gravity_removed="safe_tilt_only",
            yaw_usage="diagnostic_only",
        ),
    )

    # ---- units (minimal, but useful) ----
    ds["pres"].attrs["units"] = "dbar"
    ds["measurement_code"].attrs["comment"] = "Argo reference table 15; float-specific mapping in AUX global attrs"
    for v in [
        "acc_body_x","acc_body_y","acc_body_z",
        "acc_ned_n","acc_ned_e","acc_ned_d",
        "acc_lin_ned_n","acc_lin_ned_e","acc_lin_ned_d"
    ]:
        ds[v].attrs["units"] = "m s-2"
    for v in ["roll","pitch","yaw"]:
        ds[v].attrs["units"] = "rad"
    for v in ["gyro_body_x","gyro_body_y","gyro_body_z"]:
        ds[v].attrs["units"] = cfg.imu.gyro.units
    for v in ["mag_body_x","mag_body_y","mag_body_z"]:
        ds[v].attrs["units"] = "arb"

    return ds
