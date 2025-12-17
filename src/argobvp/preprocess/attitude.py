from __future__ import annotations
import numpy as np

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def wrap_pi(x):
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


# ------------------------------------------------------------
# Roll / Pitch from accelerometer (gravity vector)
# ------------------------------------------------------------

def roll_pitch_from_acc(acc_body: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute roll and pitch from accelerometer assuming:
    - acc includes gravity
    - body frame convention is internally consistent

    roll  = atan2( ay, az )
    pitch = atan2( -ax, sqrt(ay^2 + az^2) )

    Returns radians.
    """
    ax = acc_body["x"]
    ay = acc_body["y"]
    az = acc_body["z"]

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

    return roll, pitch


# ------------------------------------------------------------
# Rotation matrix: BODY -> NED (SAFE)
# ------------------------------------------------------------

def r_body_to_ned_from_tilt(
    roll: np.ndarray,
    pitch: np.ndarray,
) -> np.ndarray:
    """
    Rotation matrix from BODY to NED using ONLY roll & pitch.
    Yaw is intentionally ignored (SAFE gravity removal).

    Convention:
    - NED frame
    - D positive downward
    """
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)

    N = roll.size
    R = np.zeros((N, 3, 3))

    # Row-wise construction
    R[:, 0, 0] = cp
    R[:, 0, 1] = sr * sp
    R[:, 0, 2] = cr * sp

    R[:, 1, 0] = 0.0
    R[:, 1, 1] = cr
    R[:, 1, 2] = -sr

    R[:, 2, 0] = -sp
    R[:, 2, 1] = sr * cp
    R[:, 2, 2] = cr * cp

    return R


# ------------------------------------------------------------
# OPTIONAL: yaw from magnetometer (diagnostic only)
# ------------------------------------------------------------

def yaw_from_mag_simple(mag_body: dict) -> np.ndarray:
    """
    Very simple yaw from magnetometer, NO tilt compensation.
    Diagnostic only.
    """
    mx = mag_body["x"]
    my = mag_body["y"]
    return np.arctan2(-my, mx)
