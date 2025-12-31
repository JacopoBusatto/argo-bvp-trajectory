# Synthetic Data

This module generates a Coriolis-like synthetic dataset (TRUTH/TRAJ/AUX) with
configurable trajectory geometry and IMU signals. It is intended for validation,
integration testing, and sweep analysis.

## Physical model

The cycle is composed of five phases:

1) **surface1**: stationary surface segment with GPS fixes.
2) **descent**: spiral trajectory down to park depth.
3) **park**: circular arc at constant depth (plus small oscillations).
4) **ascent**: spiral trajectory back to the surface.
5) **surface2**: stationary surface segment with GPS fixes.

The synthetic ENU positions are converted to lat/lon using a local tangent-plane
approximation.

## Parameters (Experiment vs Instrument)

**Experiment parameters** live in `src/argo_bvp/synth/experiment_params.py` and
control the scenario and physics:

- Sampling: `dt_surface_s`, `dt_descent_s`, `dt_park_s`, `dt_ascent_s`
- Timing: `surface1_minutes`, `descent_hours`, `park_hours`, `ascent_hours`,
  `surface2_minutes`, `cycle_hours`
- Geometry: `spiral_radius_m`, `spiral_period_s`, `park_radius_m`,
  `park_arc_fraction`, `park_depth_m`
- Noise: `acc_sigma_ms2`, `seed`
- Smoothing: `transition_seconds`
- Parking oscillations: `park_z_osc_*`, `park_r_osc_*`

**Instrument parameters** live in `src/argo_bvp/instruments/registry.py` and
control conversion from IMU counts to SI:

- `lsb_to_ms2`
- `gyro_lsb_to_rads`
- `mag_lsb_to_uT`

NOTE (IT): non usiamo ancora gyro/mag nei calcoli, ma i canali esistono.

## Smoothing between phases (Level 1)

To avoid acceleration spikes at phase transitions, velocities are blended in a
window around each boundary using a smoothstep:

```
smoothstep(u) = 3u^2 - 2u^3,  u = (t - t0) / T
```

The blending window extends approximately ±3×`transition_seconds` around each
transition. Positions are reintegrated inside the window, preserving continuity
without adding a new phase label. Debug indices are stored in `is_transition`.

Limits: this is a kinematic blend, not a physical model; very large windows will
oversmooth true dynamics.

## Parking oscillations

During the parking phase only:

```
z(t) = z_base(t) + Az * sin(2πu/Tz + φz)
R(t) = R0 + dR * sin(2πu/Tr + φr)
```

with `u = t - t_park_start`. The arc angle law is unchanged; only radius and
depth are modulated. Phase labels remain unchanged.

## Outputs

`synth` writes:
- `SYNTH_<tag>_TRUTH.nc` (ground truth)
- `SYNTH_<tag>_TRAJ.nc` (surface GPS + pressure)
- `SYNTH_<tag>_AUX.nc` (IMU counts)
- `SYNTH_<tag>_{plan,3d,acc,depth}.png`

## Example

```bash
python -m argo_bvp.cli synth --outdir outputs/synthetic
```

```bash
python -m argo_bvp.cli synth \
  --outdir outputs/synthetic \
  --cycle-hours 24 \
  --dt-descent-s 5 \
  --dt-park-s 30 \
  --dt-ascent-s 5 \
  --acc-sigma-ms2 0.002
```
