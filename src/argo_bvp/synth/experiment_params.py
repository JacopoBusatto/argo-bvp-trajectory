"""Experiment parameters for synthetic trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentParams:
    cycle_hours: float
    start_juld: float
    lat0: float
    lon0: float
    dt_surface_s: float
    dt_descent_s: float
    dt_park_s: float
    dt_ascent_s: float
    transition_seconds: float
    surface1_minutes: float
    descent_hours: float
    park_depth_m: float
    park_hours: float
    ascent_hours: float
    surface2_minutes: float
    spiral_radius_m: float
    spiral_period_s: float
    park_arc_fraction: float
    park_radius_m: float
    park_z_osc_amplitude_m: float
    park_z_osc_period_s: float
    park_r_osc_amplitude_m: float
    park_r_osc_period_s: float
    park_z_osc_phase_rad: float
    park_r_osc_phase_rad: float
    acc_sigma_ms2: float
    seed: int


_SURFACE1_MINUTES = 15.0
_SURFACE2_MINUTES = 15.0
_DESCENT_HOURS = 3.
_ASCENT_HOURS = 3.
_PARK_HOURS = 17.5
_CYCLE_HOURS = (
    (_SURFACE1_MINUTES + _SURFACE2_MINUTES) / 60.0
    + _DESCENT_HOURS
    + _PARK_HOURS
    + _ASCENT_HOURS
)

DEFAULT_EXPERIMENT = ExperimentParams(
    cycle_hours=_CYCLE_HOURS,
    start_juld=0.0,
    lat0=42.0,
    lon0=12.0,
    dt_surface_s=10.0,
    dt_descent_s=5.0,
    dt_park_s=10.0,
    dt_ascent_s=5.0,
    transition_seconds=120.0,
    surface1_minutes=_SURFACE1_MINUTES,
    descent_hours=_DESCENT_HOURS,
    park_depth_m=1000.0,
    park_hours=_PARK_HOURS,
    ascent_hours=_ASCENT_HOURS,
    surface2_minutes=_SURFACE2_MINUTES,
    spiral_radius_m=10.0,
    spiral_period_s=900.0,
    park_arc_fraction=0.2,
    park_radius_m=2000.0,
    park_z_osc_amplitude_m=5.0,
    park_z_osc_period_s=900.0,
    park_r_osc_amplitude_m=10.0,
    park_r_osc_period_s=3600.0,
    park_z_osc_phase_rad=0.0,
    park_r_osc_phase_rad=0.0,
    acc_sigma_ms2=0.0,
    seed=0,
)


__all__ = ["ExperimentParams", "DEFAULT_EXPERIMENT"]
