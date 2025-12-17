from __future__ import annotations

import numpy as np


def roll_pitch_from_acc(acc_body: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    acc_body: {"x","y","z"} in m/s^2 OR in g-units (scale cancels for angles).
    Returns roll, pitch in radians (aerospace convention).
    """
    ax = acc_body["x"]
    ay = acc_body["y"]
    az = acc_body["z"]

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    return roll, pitch


def yaw_from_mag_tilt_comp(mag_body: dict, roll: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """
    Tilt-compensated heading (yaw) from magnetometer using roll/pitch.
    Assumes roll/pitch follow the same body-axis convention used in r_body_to_ned().
    Returns yaw in radians in [-pi, pi].
    """
    mx = mag_body["x"]
    my = mag_body["y"]
    mz = mag_body["z"]

    cr = np.cos(roll);  sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)

    # Rotate magnetic field into the horizontal plane (level frame)
    # One common formulation:
    mxh = mx * cp + mz * sp
    myh = mx * sr * sp + my * cr - mz * sr * cp

    yaw = np.arctan2(myh, mxh)
    return yaw


def yaw_from_mag(mag_body: dict) -> np.ndarray:
    """
    Non tilt-compensated heading (legacy / debug).
    """
    mx = mag_body["x"]
    my = mag_body["y"]
    return np.arctan2(my, mx)


def r_body_to_ned(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
    Returns rotation matrices R with shape (N, 3, 3), such that:
      v_ned = R @ v_body
    """
    cr = np.cos(roll);  sr = np.sin(roll)
    cp = np.cos(pitch); sp = np.sin(pitch)
    cy = np.cos(yaw);   sy = np.sin(yaw)

    N = roll.size
    R = np.zeros((N, 3, 3), dtype=float)

    # ZYX (yaw-pitch-roll) rotation
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr

    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr

    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr

    return R

def complementary_filter_angles(
    roll_acc: np.ndarray,
    pitch_acc: np.ndarray,
    yaw_mag: np.ndarray,
    gyro: dict,
    time: np.ndarray,
    alpha: float = 0.98,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple complementary filter:
    - gyro provides high-frequency dynamics
    - accel/mag provide low-frequency reference

    alpha ~ 0.98 is typical.
    """
    N = roll_acc.size

    roll = np.zeros(N)
    pitch = np.zeros(N)
    yaw = np.zeros(N)

    # initialize with accel/mag
    roll[0] = roll_acc[0]
    pitch[0] = pitch_acc[0]
    yaw[0] = yaw_mag[0]

    # time in seconds
    t = time.astype("datetime64[ns]").astype("int64") * 1e-9
    dt = np.diff(t, prepend=t[0])

    for i in range(1, N):
        # gyro integration
        roll_gyro = roll[i-1] + gyro["x"][i] * dt[i]
        pitch_gyro = pitch[i-1] + gyro["y"][i] * dt[i]
        yaw_gyro = yaw[i-1] + gyro["z"][i] * dt[i]

        # complementary fusion
        roll[i] = alpha * roll_gyro + (1 - alpha) * roll_acc[i]
        pitch[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc[i]
        yaw[i] = alpha * yaw_gyro + (1 - alpha) * yaw_mag[i]

    return roll, pitch, yaw

def smooth_running_mean(x: np.ndarray, win: int) -> np.ndarray:
    """
    Running mean (same length) with edge padding.
    win in samples, must be >= 1.
    """
    x = np.asarray(x, dtype=float)
    win = int(max(1, win))
    if win == 1:
        return x

    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    y = np.convolve(xpad, kernel, mode="valid")
    return y


def roll_pitch_from_acc_lowpass(acc_body: dict, win: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate roll/pitch from LOW-PASS accelerometer (gravity direction),
    then use those angles to rotate the *raw* accelerometer.
    """
    ax = smooth_running_mean(acc_body["x"], win)
    ay = smooth_running_mean(acc_body["y"], win)
    az = smooth_running_mean(acc_body["z"], win)

    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    return roll, pitch
