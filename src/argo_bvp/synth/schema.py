"""Synthetic TRAJ/AUX schema constants for Coriolis-like data."""

from __future__ import annotations

N_MEASUREMENT = "N_MEASUREMENT"
N_CYCLE = "N_CYCLE"

USED_TRAJ_VARS = (
    "JULD",
    "LATITUDE",
    "LONGITUDE",
    "POSITION_QC",
    "CYCLE_NUMBER",
    "MEASUREMENT_CODE",
    "PRES",
    "TEMP",
    "PSAL",
)

USED_AUX_VARS = (
    "JULD",
    "CYCLE_NUMBER",
    "MEASUREMENT_CODE",
    "PRES",
    "TEMP_COUNT_INERTIAL",
    "LINEAR_ACCELERATION_COUNT_X",
    "LINEAR_ACCELERATION_COUNT_Y",
    "LINEAR_ACCELERATION_COUNT_Z",
    "ANGULAR_RATE_COUNT_X",
    "ANGULAR_RATE_COUNT_Y",
    "ANGULAR_RATE_COUNT_Z",
    "MAGNETIC_FIELD_COUNT_X",
    "MAGNETIC_FIELD_COUNT_Y",
    "MAGNETIC_FIELD_COUNT_Z",
)

EXPECTED_UNITS = {
    "JULD": "relative julian days (parts of day)",
    "LATITUDE": "degree_north",
    "LONGITUDE": "degree_east",
    "POSITION_QC": "Argo reference table 2",
    "CYCLE_NUMBER": "cycle index (0..N)",
    "MEASUREMENT_CODE": "Argo reference table 15",
    "PRES": "decibar",
    "TEMP": "degree_Celsius",
    "PSAL": "psu",
    "TEMP_COUNT_INERTIAL": "count",
    "LINEAR_ACCELERATION_COUNT_X": "count",
    "LINEAR_ACCELERATION_COUNT_Y": "count",
    "LINEAR_ACCELERATION_COUNT_Z": "count",
    "ANGULAR_RATE_COUNT_X": "count",
    "ANGULAR_RATE_COUNT_Y": "count",
    "ANGULAR_RATE_COUNT_Z": "count",
    "MAGNETIC_FIELD_COUNT_X": "count",
    "MAGNETIC_FIELD_COUNT_Y": "count",
    "MAGNETIC_FIELD_COUNT_Z": "count",
}

__all__ = [
    "N_MEASUREMENT",
    "N_CYCLE",
    "USED_TRAJ_VARS",
    "USED_AUX_VARS",
    "EXPECTED_UNITS",
]
