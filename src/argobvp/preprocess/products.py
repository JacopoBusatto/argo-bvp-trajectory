from __future__ import annotations

import numpy as np
import xarray as xr

from .config import PreprocessConfig
from .io_coriolis import extract_aux_minimal, build_valid_mask
from .imu_calib import calibrate_accel_counts, calibrate_gyro_counts, calibrate_mag_counts
from .attitude import (
    roll_pitch_from_acc_lowpass,
    yaw_from_mag_tilt_comp,
    complementary_filter_angles,
    r_body_to_ned,
)


ARGO_EPOCH_NS = np.datetime64("1950-01-01T00:00:00", "ns")


def _juld_days_to_datetime64(juld_days: np.ndarray) -> np.ndarray:
    """
    Convert numeric JULD (days since 1950-01-01) -> datetime64[ns].
    """
    juld_days = np.asarray(juld_days).reshape(-1).astype(float)
    seconds = np.round(juld_days * 86400.0).astype("int64")
    return ARGO_EPOCH_NS + seconds.astype("timedelta64[s]")


def _datetime64_to_juld_days(t: np.ndarray) -> np.ndarray:
    """
    Convert datetime64 -> numeric JULD days since 1950-01-01.
    """
    t = np.asarray(t).reshape(-1).astype("datetime64[ns]")
    dt_s = (t - ARGO_EPOCH_NS).astype("timedelta64[s]").astype(np.int64)
    return dt_s.astype(float) / 86400.0


def _normalize_juld(juld: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Accept either:
      - numeric JULD in days since 1950-01-01
      - datetime64 (already decoded)
    Return:
      (juld_days_float, time_datetime64[ns])
    """
    juld = np.asarray(juld).reshape(-1)

    if np.issubdtype(juld.dtype, np.datetime64):
        time = juld.astype("datetime64[ns]")
        juld_days = _datetime64_to_juld_days(time)
        return juld_days, time

    # numeric case
    juld_days = juld.astype(float)
    time = _juld_days_to_datetime64(juld_days)
    return juld_days, time


def build_preprocessed_dataset(ds_aux: xr.Dataset, cfg: PreprocessConfig) -> xr.Dataset:
    raw = extract_aux_minimal(ds_aux)

    juld_raw = raw["JULD"]
    pres = raw["PRES"]
    cycle = raw["CYCLE_NUMBER"]

    axc = raw["LINEAR_ACCELERATION_COUNT_X"]
    ayc = raw["LINEAR_ACCELERATION_COUNT_Y"]
    azc = raw["LINEAR_ACCELERATION_COUNT_Z"]

    gxc = raw["ANGULAR_RATE_COUNT_X"]
    gyc = raw["ANGULAR_RATE_COUNT_Y"]
    gzc = raw["ANGULAR_RATE_COUNT_Z"]

    mxc = raw["MAGNETIC_FIELD_COUNT_X"]
    myc = raw["MAGNETIC_FIELD_COUNT_Y"]
    mzc = raw["MAGNETIC_FIELD_COUNT_Z"]

    # base mask: finite essentials (works for numeric arrays; for datetime JULD it won't filter by JULD finiteness)
    valid = build_valid_mask(pres, cycle, axc, ayc, azc, gxc, gyc, gzc, mxc, myc, mzc)
    # additionally ensure JULD isn't NaT if it's datetime64
    if np.issubdtype(np.asarray(juld_raw).dtype, np.datetime64):
        valid &= ~np.isnat(np.asarray(juld_raw).reshape(-1))
    else:
        valid &= np.isfinite(np.asarray(juld_raw).reshape(-1).astype(float))

    juld_raw = np.asarray(juld_raw).reshape(-1)[valid]
    pres = pres[valid]
    cycle = cycle[valid].astype(int)

    counts_acc = {"X": axc[valid], "Y": ayc[valid], "Z": azc[valid]}
    counts_gyro = {"X": gxc[valid], "Y": gyc[valid], "Z": gzc[valid]}
    counts_mag = {"X": mxc[valid], "Y": myc[valid], "Z": mzc[valid]}

    # time normalization (Argo-consistent)
    juld_days, time = _normalize_juld(juld_raw)

    # calibrate
    acc_g = calibrate_accel_counts(counts_acc, cfg.imu.accel)  # in "g-units"
    acc_body = {k: v * cfg.imu.g for k, v in acc_g.items()}    # m/s^2

    gyro_body = calibrate_gyro_counts(counts_gyro, cfg.imu.gyro)  # units per cfg
    mag_body = calibrate_mag_counts(counts_mag, cfg.imu.mag)

    # attitude (MVP)
    roll, pitch = roll_pitch_from_acc_lowpass(acc_body, win=101)
    yaw = yaw_from_mag_tilt_comp(mag_body, roll, pitch)

    # rotation + gravity removal in NED
    if cfg.imu.frame != "NED":
        raise NotImplementedError("Only NED supported in MVP (as requested).")

    R = r_body_to_ned(roll, pitch, yaw)
    a_body_vec = np.stack([acc_body["x"], acc_body["y"], acc_body["z"]], axis=1)  # (N,3)
    a_ned = np.einsum("nij,nj->ni", R, a_body_vec)

    gvec = np.array([0.0, 0.0, cfg.imu.g], dtype=float)
    a_lin_ned = a_ned - gvec[None, :]

    ds = xr.Dataset(
        coords=dict(
            obs=np.arange(time.size, dtype=int),
            time=("obs", time),
        ),
        data_vars=dict(
            juld=("obs", juld_days),          # float days since 1950-01-01
            pres=("obs", pres),
            cycle_number=("obs", cycle),

            acc_body_x=("obs", acc_body["x"]),
            acc_body_y=("obs", acc_body["y"]),
            acc_body_z=("obs", acc_body["z"]),

            gyro_body_x=("obs", gyro_body["x"]),
            gyro_body_y=("obs", gyro_body["y"]),
            gyro_body_z=("obs", gyro_body["z"]),

            mag_body_x=("obs", mag_body["x"]),
            mag_body_y=("obs", mag_body["y"]),
            mag_body_z=("obs", mag_body["z"]),

            roll=("obs", roll),
            pitch=("obs", pitch),
            yaw=("obs", yaw),

            acc_ned_n=("obs", a_ned[:, 0]),
            acc_ned_e=("obs", a_ned[:, 1]),
            acc_ned_d=("obs", a_ned[:, 2]),

            acc_lin_ned_n=("obs", a_lin_ned[:, 0]),
            acc_lin_ned_e=("obs", a_lin_ned[:, 1]),
            acc_lin_ned_d=("obs", a_lin_ned[:, 2]),
        ),
        attrs=dict(
            platform=cfg.platform,
            frame=cfg.imu.frame,
            gravity_included_input="LINEAR_ACCELERATION_COUNT_*",
            gravity_removed_output="acc_lin_ned_*",
            gyro_units=cfg.imu.gyro.units,
            juld_reference="1950-01-01T00:00:00Z",
            juld_units="days",
        ),
    )

    # units metadata (minimal)
    ds["juld"].attrs["units"] = "days since 1950-01-01 00:00:00"
    ds["pres"].attrs["units"] = "dbar"

    for v in [
        "acc_body_x", "acc_body_y", "acc_body_z",
        "acc_ned_n", "acc_ned_e", "acc_ned_d",
        "acc_lin_ned_n", "acc_lin_ned_e", "acc_lin_ned_d",
    ]:
        ds[v].attrs["units"] = "m s-2"

    for v in ["roll", "pitch", "yaw"]:
        ds[v].attrs["units"] = "rad"

    return ds
