Preprocess Overview
===================

Scope
-----
This document explains what the preprocessing pipeline does, which inputs it expects, and what products it generates. It consolidates the user-facing guidance that was previously in `docs/MANUALE_PREPROCESS.md`.

What the pipeline does
----------------------
- Reads Coriolis AUX (IMU + pressure + cycle/timing metadata).
- Calibrates IMU channels, estimates attitude (safe tilt-only), rotates to NED, and removes gravity (linear acceleration) where applicable.
- Segments observations into phases (e.g., park_drift, ascent) and derives per-cycle keypoints.
- Associates surface position fixes from the TRAJ file to the end-of-surface window.
- Exposes cycle- and segment-level products to downstream solvers (BVP, diagnostics).
- Explicitly flags cycles valid for BVP based on parking-phase sampling rules (see SAMPLING_VALIDITY).
- Builds a BVP-ready view that includes all IMU-attendible phases but requires an attendible parking phase.
- Does **not** integrate trajectories or solve the BVP.

Inputs
------
- AUX NetCDF: IMU counts, pressure, measurement codes, cycle numbers, timestamps.
- TRAJ NetCDF: surface position fixes and timing metadata.
- YAML config in `configs/`: paths to AUX/TRAJ, IMU calibration parameters, attitude options, and sample thresholds (`min_parking_samples_for_bvp`, `min_phase_samples_for_bvp`, default 10).

Outputs
-------
- Continuous IMU dataset (`*_preprocessed_imu.nc`):
  - time (datetime64), cycle_number, pressure
  - calibrated accelerations (body and NED), linear accelerations (NED), gyro and mag channels
  - attitude estimates (roll/pitch/yaw) and frame metadata
- Cycle product (`*_cycles.nc`):
  - keypoint times (cycle start, parking start/end, deepest point, ascent start, surface start/end)
  - representative pressures
  - surface position constraint and diagnostics (lat/lon at surface end, pos_source, pos_age_s, ...)
  - per-phase sampling diagnostics: `parking_n_obs`, `parking_attendible`, `park_sampled`, `valid_for_bvp`, plus `<phase>_n_obs` / `<phase>_attendible` for other detected phases (ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other)
  - attributes include `min_parking_samples_for_bvp`
- Segment product (`*_segments.nc`):
  - contiguous phase segments per cycle with idx0/idx1 bounds into the continuous dataset
  - phase names and flags (`is_parking_phase`)

How to run
----------
Typical end-to-end run from the repo root:

```
python -m argobvp.preprocess.runner --config configs/4903848.yml --out outputs/preprocess
```

This writes:
- `outputs/preprocess/<platform>_preprocessed_imu.nc`
- `outputs/preprocess/<platform>_cycles.nc`
- `outputs/preprocess/<platform>_segments.nc`
- optional Parquet mirrors (`--no-parquet` disables them)

BVP-ready extraction (all attendible phases, parking required):
```
python -m argobvp.preprocess.bvp_ready \
  --cont outputs/preprocess/<platform>_preprocessed_imu.nc \
  --cycles outputs/preprocess/<platform>_cycles.nc \
  --segments outputs/preprocess/<platform>_segments.nc \
  --out outputs/preprocess
```
This keeps only cycles with `valid_for_bvp=True` and includes all phases that meet the sampling thresholds; parking attendible is mandatory.

Notes on missing/invalid cycles
-------------------------------
- If no cycles have attendible parking, the BVP-ready step exits early and reports 0 valid cycles.
- Phases that are not attendible (insufficient IMU samples) are omitted from the BVP-ready time series; downstream solvers should treat them as vertical (no horizontal displacement) per the sampling/validity rules.

What the pipeline does not do
-----------------------------
- No trajectory integration (neither forward nor backward).
- No full BVP solver; only provides inputs for it.
- No correction of surface drift or inference of unsampled phases beyond explicit assumptions documented in SAMPLING_VALIDITY.
