Preprocess Reference
====================

This reference lists the preprocessing modules, their responsibilities, and the main entry points and diagnostics. It reorganizes the detailed notes from `MANUALE_PREPROCESS.md` without changing their meaning.

Module responsibilities (src/argobvp/preprocess/)
-------------------------------------------------
- `config.py`: load/validate YAML config; expose paths, IMU calibration, attitude options, and sampling thresholds (`min_parking_samples_for_bvp`, `min_phase_samples_for_bvp`).
- `io_coriolis.py`: open AUX/TRAJ NetCDF (`decode_timedelta=False`) and minimal helpers.
- `imu_calib.py`: convert IMU counts to physical units (accel, gyro, mag) using bias/gain/scale.
- `attitude.py`: estimate orientation (safe tilt-only by default); rotate accelerations to NED.
- `products.py`: orchestrate the continuous IMU dataset construction (time, pressure, calibrated channels, attitude, rotated accelerations).
- `cycles.py`: derive per-cycle keypoints and contiguous phase segments; compute per-phase sampling diagnostics (`<phase>_n_obs`, `<phase>_attendible`, including parking) plus `park_sampled`, `valid_for_bvp`; store idx0/idx1 for segments. Parking attendible is the gate for BVP validity.
- `surface_fixes.py`: attach surface position constraints from TRAJ fixes to cycles with interpolation/nearest heuristics and diagnostics (`pos_source`, `pos_age_s`, etc.).
- `writers.py`: write NetCDF/Parquet outputs.
- `runner.py`: end-to-end preprocess driver (AUX -> continuous -> cycles/segments -> surface fixes -> outputs).
- `bvp_ready.py`: extract a BVP-ready view that requires attendible parking and includes all attendible phases; enforces sampling thresholds and writes `<platform>_bvp_ready.nc`. Phases below threshold are omitted and expected to be treated as vertical by the solver.

Entry points
------------
- Full preprocess:
  ```
  python -m argobvp.preprocess.runner --config configs/<platform>.yml --out outputs/preprocess
  ```
- BVP-ready extraction (parking only):
  ```
  python -m argobvp.preprocess.bvp_ready --help
  ```

Debug/diagnostic scripts
------------------------
- `debug_phases.py`: inspect detected phases and segment order.
- `debug_surface_fixes.py`: summarize surface-fix association, methods used (interp/nearest/missing), and timing offsets.
- `check_traj_surface_fix.py`: quick sanity on TRAJ fix sampling and timing gaps.
- `examples/diagnostics/plot_bvp_ready_global.py`: map of surface fixes and per-cycle parking acceleration/pressure plots (first valid cycles), with parking sample summaries.

Data products summary
---------------------
- `*_preprocessed_imu.nc` (ds_continuous): calibrated IMU channels, attitude, rotated/linear accelerations, time/cycle/pressure.
- `*_cycles.nc` (ds_cycles): per-cycle keypoints, representative pressures, surface position constraints, parking sampling diagnostics, validity flags.
- `*_cycles.nc` (ds_cycles): per-cycle keypoints, representative pressures, surface position constraints, per-phase sampling diagnostics (`<phase>_n_obs`, `<phase>_attendible`, parking gate via `valid_for_bvp`).
- `*_segments.nc` (ds_segments): contiguous phase slices with index bounds into the continuous dataset and `is_parking_phase` flag.
- `*_bvp_ready.nc`: parking-phase-only samples and per-cycle metadata for cycles that pass the parking sample threshold.
