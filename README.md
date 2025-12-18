ARGO-BVP-TRAJECTORY
===================

This repository contains a small, self-consistent numerical framework to:
1) integrate Argo-like trajectories from acceleration (second-order kinematics), and
2) solve a simple boundary-value constraint (BVP) by “shooting” on the unknown initial velocity.

The scientific objective is to reconstruct and quantify positional uncertainty at key Argo phases
(e.g., start of ascent, end of surface window), using IMU-derived acceleration and surface position fixes.

The project is built step-by-step with emphasis on:
- numerical correctness
- reproducibility
- diagnostic tooling (debug scripts + synthetic tests)
- a clean preprocessing pipeline from Coriolis NetCDF to analysis-ready products


------------------------------------------------------------
1. PROJECT STRUCTURE
------------------------------------------------------------

Standard "src-layout" structure:

```
argo-bvp-trajectory/
├── src/
│   └── argobvp/
│       ├── __init__.py
│       ├── integrators.py
│       ├── bvp.py
│       ├── metrics.py
│       └── preprocess/
│           ├── __init__.py
│           ├── config.py
│           ├── io_coriolis.py
│           ├── imu_calib.py
│           ├── attitude.py
│           ├── products.py
│           ├── cycles.py
│           ├── surface_fixes.py
│           ├── writers.py
│           ├── runner.py
│           ├── debug_phases.py
│           ├── debug_surface_fixes.py
│           └── check_traj_surface_fix.py
│
├── tests/
├── examples/
├── configs/
├── outputs/               (ignored by git)
├── pyproject.toml
├── .gitignore
└── README.md
```

------------------------------------------------------------
2. NUMERICAL MODEL (core)
------------------------------------------------------------

We integrate second-order kinematics:

    dr/dt = v
    dv/dt = a(t, r, v)

The integration is implemented on an arbitrary time grid t (strictly increasing).
We also provide a simple BVP solver via shooting on the unknown initial velocity v0,
typically in the horizontal plane (XY).


------------------------------------------------------------
3. CORE MODULES (src/argobvp/)
------------------------------------------------------------

3.1 integrators.py
- integrate_2nd_order(t, r0, v0, a_fun, method="trapezoid", backward=False)
Supported schemes:
- Euler (1st order)
- Trapezoid (2nd order; default reference)
- RK4 (high-accuracy benchmark)

3.2 bvp.py
- shoot_v0_to_hit_rT(...): solve r(T)[dims] = rT_target[dims] by shooting on v0[dims]

3.3 metrics.py
- nearest_index, endpoint_error, point_error_at_time, ...


------------------------------------------------------------
4. PREPROCESSING PIPELINE (Coriolis AUX + TRAJ)
------------------------------------------------------------

Goal:
Transform Coriolis Argo NetCDF files into compact, analysis-ready products:
- a continuous IMU dataset (time series),
- cycle-level keypoints,
- segment-level phase labels,
- surface position constraints from TRAJ fixes.

Inputs:
- AUX trajectory NetCDF (IMU channels + pressure + time + cycle)
- TRAJ NetCDF (surface fixes / positions + per-cycle timing metadata)

Configuration:
YAML configs live in `configs/` and define:
- input paths (traj + aux)
- IMU calibration parameters (bias/gain/scale)
- attitude mode parameters

Important:
- raw data files are NOT committed to git
- outputs/ is NOT committed to git


------------------------------------------------------------
4.1 Main runner
------------------------------------------------------------

Run the full preprocessing:

    python -m argobvp.preprocess.runner --config configs/4903848.yml --out outputs/preprocess

This produces:
- outputs/preprocess/<platform>_preprocessed_imu.nc
- outputs/preprocess/<platform>_cycles.nc
- outputs/preprocess/<platform>_segments.nc
and optionally parquet mirrors:
- outputs/preprocess/<platform>_cycles.parquet
- outputs/preprocess/<platform>_segments.parquet


------------------------------------------------------------
4.2 Continuous IMU product (ds_cont)
------------------------------------------------------------

Produced by:
- build_preprocessed_dataset(ds_aux, cfg)

Includes (subset, evolving):
- time coordinate (datetime64[ns])
- cycle_number
- pres (dbar)
- calibrated accelerometer channels (m/s^2)
- calibrated gyro channels (rad/s) [NOTE: scale may be provisional]
- magnetometer channels (corrected if configured)
- estimated attitude angles (roll/pitch/yaw depending on mode)
- accelerations rotated to a navigation frame (NED)
- optional gravity removal to obtain linear acceleration in NED

Frame convention:
- NED (North-East-Down), z positive downward (consistent with pressure).


------------------------------------------------------------
4.3 Cycle and segment products (ds_cycles, ds_segments)
------------------------------------------------------------

Built from the continuous dataset:
- build_cycle_products(ds_cont, cfg)

ds_cycles contains, for each cycle:
- t_cycle_start
- t_park_start
- t_profile_deepest
- t_ascent_start
- t_surface_start
- t_surface_end
- representative pressures (park and deepest)

ds_segments contains contiguous segments per cycle:
- cycle_number
- segment_name (e.g. park_drift, ascent, ...)
- idx0, idx1 (index bounds in the continuous dataset)
- t0, t1

Note:
Phase detection depends on what is available in the dataset (e.g. MEASUREMENT_CODE)
and may produce only a subset of phases for some platforms.


------------------------------------------------------------
4.4 Surface position constraints from TRAJ fixes
------------------------------------------------------------

The runner can add a surface position constraint to ds_cycles:
- add_surface_position_from_traj(ds_cycles, ds_traj, ...)

For each cycle, we associate a lat/lon position to t_surface_end using TRAJ fixes,
with robust guards:
- prefer interpolation only if surrounding fixes are sufficiently close in time
- otherwise fallback to the nearest fix
- store diagnostics such as pos_age_s and the actual time used

In the current dataset, fixes often occur ~20 minutes after t_surface_end,
so the typical mode is nearest-fix with pos_age_s ~ 1100–1200 s.


------------------------------------------------------------
5. DEBUG SCRIPTS
------------------------------------------------------------

Inspect phase segmentation and cycle keypoints:

    python -m argobvp.preprocess.debug_phases

Inspect surface-fix association summary:

    python -m argobvp.preprocess.debug_surface_fixes

Check TRAJ fix sampling (basic sanity):

    python -m argobvp.preprocess.check_traj_surface_fix


------------------------------------------------------------
6. INSTALLATION
------------------------------------------------------------

Create venv (Windows PowerShell):

    py -m venv .venv
    .venv\Scripts\activate

Install editable + dev tools:

    pip install -e ".[dev]"

Run tests:

    pytest -q


------------------------------------------------------------
7. CURRENT LIMITATIONS / NOTES
------------------------------------------------------------

- Gyro scale:
  `imu.gyro.scale` may be provisional if the exact count-to-rate conversion is unknown.
  Attitude estimation modes are designed to remain usable under this uncertainty.

- Phase coverage:
  Some datasets may provide only a subset of phases (e.g. park_drift + ascent)
  depending on what is present/encoded in MEASUREMENT_CODE and timing variables.

- Surface fixes:
  Surface position constraints may occur after the end of the surface window.
  The pipeline stores pos_age_s to make this explicit and usable in later BVP constraints.


------------------------------------------------------------
8. ROADMAP (next steps)
------------------------------------------------------------

1) Improve phase reconstruction:
   - use TRAJ timing variables (JULD_* per cycle) as a phase backbone
   - derive missing phases even when IMU samples are absent in that window

2) Add time-alignment diagnostics:
   - verify keypoints fall within the continuous IMU time range per cycle

3) Integrate trajectories and compute keypoint errors:
   - forward integration from IMU acceleration
   - BVP shooting in XY using surface constraints
