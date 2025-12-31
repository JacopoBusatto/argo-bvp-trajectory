from __future__ import annotations

import numpy as np
from .config import AccelCalib, GyroCalib, MagCalib


def _pick_axis(counts: dict, axis: str) -> np.ndarray:
    # axis in {"X","Y","Z"}
    return counts[axis]


def calibrate_accel_counts(counts_xyz: dict, calib: AccelCalib) -> dict:
    """
    counts_xyz: {"X":..., "Y":..., "Z":...} raw counts arrays (same length)
    returns: {"x":..., "y":..., "z":...} in m/s^2 in BODY frame (but with your desired axis mapping/sign)
    """
    out = {}
    for out_axis in ("x", "y", "z"):
        src_axis = calib.axis_map[out_axis]  # "X"/"Y"/"Z"
        sgn = int(calib.sign[out_axis])
        bias = float(calib.bias_counts[out_axis])
        gain = float(calib.gain[out_axis])

        raw = _pick_axis(counts_xyz, src_axis).astype(float)
        # Matlab-like: 4*gain*(counts + bias)/65536  [in g], then *g0 to get m/s^2? (we keep "in g" here?)
        # We'll output in "g units" for now and multiply by g later (so we don't hardcode g here).
        out[out_axis] = sgn * (calib.scale_g * gain * (raw + bias) / calib.denom)
    return out


def calibrate_gyro_counts(counts_xyz: dict, calib: GyroCalib) -> dict:
    """
    Returns angular rate in 'calib.units' (rad/s or deg/s), still in BODY frame.
    """
    out = {}
    for ax in ("x", "y", "z"):
        raw = counts_xyz[ax.upper()].astype(float)
        out[ax] = (raw + float(calib.bias_counts[ax])) * float(calib.scale)
    return out


def calibrate_mag_counts(counts_xyz: dict, calib: MagCalib) -> dict:
    """
    Apply hard-iron and (optional) soft-iron in XY like your Matlab.
    Returns corrected mag in "counts-like units" (relative).
    """
    mx = counts_xyz["X"].astype(float) + float(calib.hard_iron["x"])
    my = counts_xyz["Y"].astype(float) + float(calib.hard_iron["y"])
    mz = counts_xyz["Z"].astype(float) + float(calib.hard_iron["z"])

    xx = float(calib.soft_iron_xy["xx"])
    xy = float(calib.soft_iron_xy["xy"])
    yx = float(calib.soft_iron_xy["yx"])
    yy = float(calib.soft_iron_xy["yy"])

    # soft-iron correction on XY
    mx2 = mx * xx + my * xy
    my2 = mx * yx + my * yy

    return {"x": mx2, "y": my2, "z": mz}
