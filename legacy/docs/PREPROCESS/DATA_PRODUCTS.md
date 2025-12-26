Preprocessing Data Products
===========================

This guide describes the datasets written by the preprocess pipeline and the BVP-ready builder. It is a data dictionary: variables, dimensions, units, meaning, and origin. Parquet mirrors are optional; they contain the same variables flattened to rows.

Products
--------
- `*_preprocessed_imu.nc` (`ds_continuous`)
- `*_cycles.nc` (`ds_cycles`)
- `*_segments.nc` (`ds_segments`)
- `*_bvp_ready.nc` (`ds_bvp_ready`)
- Parquet mirrors: optional for cycles/segments when `write_parquet_products=True` in `runner.py`; flattened tables (one row per coord).

Conventions and thresholds
--------------------------
- Frame: NED (North-East-Down), Down positive.
- Time: datetime64[ns] where present.
- Pressure: dbar.
- Parking attendible (samples ≥ `min_parking_samples_for_bvp`) plus surface anchors gate `valid_for_bvp`.
- Other phases attendible if samples ≥ `min_phase_samples_for_bvp`; omitted phases are expected vertical by the solver.

Dataset: *_preprocessed_imu.nc (ds_continuous)
-----------------------------------------------
Purpose: continuous time series of calibrated IMU data and derived quantities. Written by `products.build_preprocessed_dataset`.

Coordinates / index
- `obs` (length N): integer index.
- `time` (`obs`, datetime64[ns]): observation time.

Core variables
- `juld` (`obs`, days since 1950-01-01): original time reference.
- `pres` (`obs`, dbar): pressure from AUX.
- `cycle_number` (`obs`, int): cycle id.
- `measurement_code` (`obs`, int): Argo MEASUREMENT_CODE (phase hints).

Dynamics (BODY / NED)
- `acc_body_x/y/z` (`obs`, m/s^2): calibrated accelerations in body frame.  
- `acc_ned_n/e/d` (`obs`, m/s^2): accelerations rotated to NED (includes gravity).  
- `acc_lin_ned_n/e/d` (`obs`, m/s^2): gravity-removed linear acceleration (NED).  
- `gyro_body_x/y/z` (`obs`, cfg units: rad/s or deg/s): calibrated angular rates.  
- `mag_body_x/y/z` (`obs`, arb): calibrated/biased-corrected magnetometer.

Attitude (diagnostic)
- `roll`, `pitch`, `yaw` (`obs`, rad): roll/pitch from accelerometer; yaw from simple mag (diagnostic only).

Attributes
- `platform`, `frame`="NED", `gravity_removed`="safe_tilt_only", `yaw_usage`="diagnostic_only".

Dataset: *_cycles.nc (ds_cycles)
--------------------------------
Purpose: per-cycle keypoints, pressures, surface fixes, and sampling/validity flags. Written by `cycles.build_cycle_products`, then augmented by `surface_fixes.add_surface_position_from_traj`.

Coordinates
- `cycle` (length C): cycle index coord; also `cycle_number` data var (int).

Keypoint times (datetime64[ns])
- `t_cycle_start`, `t_park_start`, `t_park_end`, `t_descent_to_profile_start`, `t_profile_deepest`, `t_ascent_start`, `t_surface_start`, `t_surface_end`. Origin: `cycles.py` from MEASUREMENT_CODE/pressure heuristics.

Pressures
- `pres_park_rep` (dbar): representative parking pressure (code 301 or median during park).
- `pres_profile_deepest` (dbar): deepest pressure (code 503 or max).

Surface fixes + diagnostics (added by `surface_fixes.py`)
- `lat_surface_start`, `lon_surface_start` (deg): last fix before descent (`t_cycle_start` window).
- `lat_surface_end`, `lon_surface_end` (deg): position tied to `t_surface_end`.
- `t_surface_start_fix`, `t_surface_end_fix` (datetime64[ns]): timestamps of chosen fixes.
- `pos_source` (str): "interp" | "nearest" | "missing".
- `t_pos_used` (datetime64[ns]): timestamp of fix used.
- `pos_age_s` (s): `t_pos_used - t_surface_end`.
- `dt_before_s`, `dt_after_s`, `gap_s`, `dt_nearest_s` (s): diagnostics.
- `t_fix_before/after/nearest` (datetime64[ns]): fix times used for diagnostics.
- `anchors_attendible` (bool): true if both surface anchors are finite.

Sampling / validity (parking gate + per-phase)
- Parking: `parking_n_obs` (int), `parking_attendible` (bool), `park_sampled` (bool >0), `valid_for_bvp` (bool = parking_attendible & anchors_attendible).
- Threshold attrs: `min_parking_samples_for_bvp`, `min_phase_samples_for_bvp` (ints).
- Per-phase counts/flags (from `cycles.py`): `<phase>_n_obs` (int), `<phase>_attendible` (bool) for phases detected by the classifier:  
  `parking` (handled above), `ascent`, `descent_to_profile`, `profile_drift`, `surface`, `in_air`, `grounded`, `other`.

Dataset: *_segments.nc (ds_segments)
------------------------------------
Purpose: contiguous phase segments with indices into continuous dataset. Written by `cycles.build_cycle_products`.

Coordinates
- `segment` (length S): segment index.

Variables
- `cycle_number` (`segment`, int): owning cycle.
- `segment_name` (`segment`, str): phase label (e.g., park_drift, ascent, etc.).
- `idx0`, `idx1` (`segment`, int): observation bounds in `ds_continuous` (idx1 exclusive).
- `t0`, `t1` (`segment`, datetime64[ns]): times of first/last obs in segment.
- `is_parking_phase` (`segment`, bool): convenience flag (segment_name == park_drift).

Attributes
- `platform`, `pres_surface_max`, notes about idx0/idx1 referencing continuous obs.

Dataset: *_bvp_ready.nc (ds_bvp_ready)
--------------------------------------
Purpose: observations from all attendible phases (parking mandatory) for BVP solver input. Built by `bvp_ready.build_bvp_ready_dataset`.

Coordinates
- `obs` (length N): observation index within BVP-ready.
- `cycle` (length C_sel): cycles passing parking gate and containing attendible samples.

Per-cycle variables
- `cycle_number` (`cycle`, int)
- `row_start`, `row_size` (`cycle`, int): slice into obs for that cycle.
- `t0`, `t1` (`cycle`, datetime64[ns]): first/last obs time in BVP slice.
- Key times: `t_cycle_start`, `t_descent_to_profile_start`, `t_profile_deepest`, `t_ascent_start`, `t_surface_start`, `t_surface_end`, `t_park_start`, `t_park_end` (where present in cycles).
- Surface fixes: `lat_surface_start`, `lon_surface_start`, `lat_surface_end`, `lon_surface_end`, `t_surface_start_fix`, `t_surface_end_fix`, `pos_age_s`, `pos_source`.
- Phase availability: `<phase>_attendible`, `<phase>_n_obs` for the same phase set as in cycles (parking, ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other).

Per-observation variables
- `time` (`obs`, datetime64[ns])
- `z_from_pres` (`obs`, dbar): pressure as depth proxy (no conversion).
- `acc_n`, `acc_e` (`obs`, m/s^2): horizontal accelerations from chosen source (linear/gravity-removed preferred).
- `phase_name` (`obs`, str): phase label of each obs.
- `cycle_number_for_obs` (`obs`, int), `cycle_index` (`obs`, int), `obs_index` (`obs`, int): provenance and original index.

Attributes
- `platform`, thresholds (`min_parking_samples_for_bvp`, `min_phase_samples_for_bvp`), `acc_source`, `phase_name_map`, notes about inclusion rule: parking attendible mandatory; other phases included if attendible, omitted otherwise (solver assumes vertical).

Parquet mirrors (optional)
--------------------------
- Written by `writers.write_parquet` when `write_parquet_products=True` in `runner.py` (cycles/segments only). Flattened tables; same variable names; file names `<platform>_cycles.parquet`, `<platform>_segments.parquet`.

Examples (copy-paste snippets)
-----------------------------

Open datasets:
```python
import xarray as xr
ds_cont = xr.open_dataset("outputs/preprocess/<platform>_preprocessed_imu.nc")
ds_cyc = xr.open_dataset("outputs/preprocess/<platform>_cycles.nc")
ds_seg = xr.open_dataset("outputs/preprocess/<platform>_segments.nc")
ds_bvp = xr.open_dataset("outputs/preprocess/<platform>_bvp_ready.nc")
```

List variables:
```python
print(list(ds_cyc.data_vars))
print(list(ds_bvp.data_vars))
```

Filter valid cycles:
```python
valid_cyc = ds_cyc.where(ds_cyc.valid_for_bvp, drop=True)
print(valid_cyc.cycle_number.values)
```

Count samples per phase (cycles):
```python
for ph in ["parking", "ascent", "descent_to_profile", "profile_drift", "surface", "in_air", "grounded", "other"]:
    nvar = f"{ph}_n_obs"
    avar = f"{ph}_attendible"
    if nvar in ds_cyc:
        print(ph, int(ds_cyc[nvar].sum()), "attendible:", ds_cyc[avar].sum().item())
```

Inspect surface fix age:
```python
age = ds_cyc.pos_age_s
print("pos_age_s (s) stats:", float(age.min()), float(age.median()), float(age.max()))
```

Summarize BVP-ready phases:
```python
phases = ds_bvp.phase_name.values.astype(str)
import numpy as np
uniq, cnt = np.unique(phases, return_counts=True)
for u, c in zip(uniq, cnt):
    print(u, int(c))
```

Check parking gate counts:
```python
mp = ds_cyc.attrs.get("min_parking_samples_for_bvp", None)
print("min_parking_samples_for_bvp:", mp, "parking_attendible cycles:", int(ds_cyc.parking_attendible.sum()))
```
