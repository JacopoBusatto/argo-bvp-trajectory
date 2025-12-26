Preprocess API Reference
========================

Scope: functions and classes in `src/argobvp/preprocess/`. Inputs/outputs are stated explicitly with shapes, units, and assumptions. Frames are NED (North-East-Down) unless noted.

Call graph / workflow
---------------------
- `runner.run_preprocess` → opens AUX (and TRAJ), calls `products.build_preprocessed_dataset`, then `cycles.build_cycle_products`, then `surface_fixes.add_surface_position_from_traj`, then writes NetCDF/Parquet via `writers`.  
- `bvp_ready.build_bvp_ready_dataset` runs after preprocess outputs: it requires `valid_for_bvp=True` (parking attendible plus surface anchors) and includes all attendible phases; non-attendible phases are omitted and expected vertical by the solver.

Module reference
----------------

### config.py
Purpose: define typed configs for IMU calibration, attitude, paths, and preprocessing thresholds; load YAML.

Classes:
- `AccelCalib(scale_g: float = 4.0, denom: float = 65536.0, bias_counts: Dict[str, float] = None, gain: Dict[str, float] = None, axis_map: Dict[str, str] = None, sign: Dict[str, int] = None)`  
  Bias/gain/axis/sign mapping for accelerometer counts → g units (later scaled by `imu.g`). Defaults to identity mapping; bias/gain defaults zero/one. No side effects.
- `GyroCalib(scale: float = 1.0, units: Literal["rad/s", "deg/s"] = "rad/s", bias_counts: Dict[str, float] = None)`  
  Simple linear scale and bias per axis; units kept in attrs.
- `MagCalib(hard_iron: Dict[str, float] = None, soft_iron_xy: Dict[str, float] = None)`  
  Hard-iron offsets and optional 2×2 soft-iron matrix on XY.
- `IMUConfig(frame: Frame = "NED", g: float = 9.80665, accel: AccelCalib = AccelCalib(), gyro: GyroCalib = GyroCalib(), mag: MagCalib = MagCalib())`  
  IMU settings; frame stored (uppercase) but all downstream assumes NED.
- `PathsConfig(traj: str, aux: str)`  
  Required file paths.
- `AttitudeConfig(mode: Literal["safe_tilt_only", "complementary"] = "safe_tilt_only", dt_max: float = 300.0, alpha: float = 0.98)`  
  Parameters for attitude estimation (currently only safe tilt-only used).
- `PreprocessConfig(platform: str, paths: PathsConfig, imu: IMUConfig = IMUConfig(), pres_surface_max: float = 5.0, min_parking_samples_for_bvp: int = 10, min_phase_samples_for_bvp: int = 10, time_reference: str = "1950-01-01T00:00:00Z", attitude: AttitudeConfig = AttitudeConfig())`  
  Main config; thresholds gate parking and other phases.

Functions:
- `load_config(path: str | Path) -> PreprocessConfig`  
  Inputs: YAML file path. Outputs: populated `PreprocessConfig`; lowercases YAML keys; validates presence of `gyro.scale`; sets defaults for thresholds and time reference. No files written.

Assumptions: NED frame, pressures in dbar, times convertible to datetime64.

### io_coriolis.py
Purpose: open AUX/TRAJ NetCDF with minimal assumptions; extract minimal AUX arrays; build validity mask.

Constants: `AUX_REQUIRED_VARS` list of mandatory AUX variable names.

Functions:
- `open_aux(path: str | Path) -> xr.Dataset`  
  Opens NetCDF with `decode_timedelta=False`. Input: file path. Output: xarray Dataset. No side effects.
- `open_traj(path: str | Path) -> xr.Dataset`  
  Same as above for TRAJ.
- `extract_aux_minimal(ds_aux: xr.Dataset, vars: Optional[Sequence[str]] = None) -> dict`  
  Inputs: AUX dataset, optional variable list. Outputs: dict with 1D numpy arrays per variable and a `<var>__attrs` copy. Raises KeyError if missing.
- `build_valid_mask(*arrays: np.ndarray) -> np.ndarray`  
  Inputs: arrays of equal length; numeric entries must be finite. Output: boolean mask same shape as first array; non-numeric arrays do not affect mask.

Assumptions: AUX variables present; no time decoding side effects; caller handles closing Dataset.

### imu_calib.py
Purpose: convert raw IMU counts to calibrated physical units using config.

Functions:
- `calibrate_accel_counts(counts_xyz: dict, calib: AccelCalib) -> dict`  
  Inputs: dict with keys "X","Y","Z" arrays; calib mapping/bias/gain/scale; uses `calib.scale_g/denom` to output in g-units; sign/axis mapping applied. Output: dict `{"x","y","z"}` in g-units (multiplied by `imu.g` later). Assumes arrays equal length.
- `calibrate_gyro_counts(counts_xyz: dict, calib: GyroCalib) -> dict`  
  Inputs: counts per axis; linear scale and bias. Output: dict `{"x","y","z"}` in `calib.units` (rad/s or deg/s).
- `calibrate_mag_counts(counts_xyz: dict, calib: MagCalib) -> dict`  
  Inputs: counts per axis; applies hard-iron offsets and soft-iron XY matrix. Output: dict `{"x","y","z"}` in corrected, relative units.

Assumptions: arrays aligned; no NaNs after upstream valid mask.

### attitude.py
Purpose: derive attitude angles and rotation matrices (safe tilt-only).

Functions:
- `wrap_pi(x)`  
  Wrap angle to [-pi, pi]. Input/Output: numpy array or scalar radians.
- `roll_pitch_from_acc(acc_body: dict) -> tuple[np.ndarray, np.ndarray]`  
  Inputs: dict with `"x","y","z"` accelerations including gravity (m/s²). Output: roll, pitch arrays (radians). Assumes consistent body frame.
- `r_body_to_ned_from_tilt(roll: np.ndarray, pitch: np.ndarray) -> np.ndarray`  
  Inputs: roll, pitch arrays. Output: (N,3,3) rotation matrices BODY→NED using only tilt; yaw ignored. Assumes NED with Down positive.
- `yaw_from_mag_simple(mag_body: dict) -> np.ndarray`  
  Inputs: dict `"x","y","z"` (relative units). Output: yaw array (radians), diagnostic only; no tilt compensation.

Assumptions: time-aligned arrays; no gyro fusion; NED convention.

### products.py
Purpose: build the continuous IMU dataset from AUX using calibration and attitude.

Functions:
- `_juld_to_datetime64(juld: np.ndarray) -> np.ndarray` (internal helper)  
  Converts numeric JULD (days since 1950-01-01) or datetime64 to datetime64[ns].
- `build_preprocessed_dataset(ds_aux: xr.Dataset, cfg: PreprocessConfig) -> xr.Dataset`  
  Inputs: AUX dataset with required vars; config. Steps: apply valid mask; calibrate accel/gyro/mag; compute roll/pitch/yaw; rotate to NED; remove gravity; convert times. Output: xarray Dataset `ds_continuous` with coords `obs`, `time`; data vars include `pres` (dbar), `cycle_number` (int), `measurement_code` (int), calibrated accelerations (`acc_body_*`, `acc_ned_*`, `acc_lin_ned_*`, units m/s²), gyro (cfg units), mag (arb), attitude angles (rad). Attributes: platform (from cfg), frame="NED", gravity_removed="safe_tilt_only", yaw_usage="diagnostic_only". No files written.

Assumptions: NED frame; gravity vector [0,0,g] with Down positive; time monotonic per cycle not enforced; valid mask removes NaNs.

### cycles.py
Purpose: derive per-cycle keypoints, per-phase sampling stats, and contiguous segments from continuous dataset.

Key constants: phase labels (`PHASE_*`) and `PHASES_FOR_ATTENDIBILITY` (parking, ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other).

Classes:
- `PhaseRules` dataclass with code mappings and `phase_for_code(self, code: int) -> str`.

Functions:
- `assign_phase(measurement_code: np.ndarray, pres: np.ndarray, pres_surface_max: float, rules: PhaseRules = DEFAULT_RULES) -> np.ndarray`  
  Inputs: measurement codes, pressures, surface threshold, rules. Output: array of phase names per observation; uses code mapping then surface fallback.
- `build_cycle_products(ds_continuous: xr.Dataset, cfg: PreprocessConfig) -> Tuple[xr.Dataset, xr.Dataset]`  
  Inputs: continuous dataset with `time`, `pres`, `cycle_number`, `measurement_code`; config thresholds. Outputs:  
  * `ds_cycles`: coord `cycle`; keypoint times (`t_cycle_start`, `t_park_start/end`, `t_descent_to_profile_start`, `t_profile_deepest`, `t_ascent_start`, `t_surface_start/end`), pressures (`pres_park_rep`, `pres_profile_deepest`), per-phase sampling counts/flags (`parking_n_obs`, `parking_attendible`, `park_sampled`, `valid_for_bvp`, `<phase>_n_obs`, `<phase>_attendible` for other phases), attrs `pres_surface_max`, `min_parking_samples_for_bvp`, `min_phase_samples_for_bvp`.  
  * `ds_segments`: coord `segment`; vars `cycle_number`, `segment_name`, `idx0`, `idx1` (obs bounds), `t0`, `t1`, `is_parking_phase`.  
  Side effects: none (in-memory).  
  Invariants: indices refer to rows of `ds_continuous`; `valid_for_bvp` starts as `parking_attendible`; `park_sampled` is any parking samples; phase attendibility uses `min_phase_samples_for_bvp` except parking uses parking threshold. (After `surface_fixes.add_surface_position_from_traj`, `valid_for_bvp` is ANDed with surface anchor availability.)

Assumptions: per-cycle grouping by `cycle_number`; NaT if missing codes; pressures in dbar.

### surface_fixes.py
Purpose: attach surface position constraints from TRAJ fixes to cycles (surface-before + surface-after anchors).

Classes:
- `SurfaceFixConfig(...)` with candidate var names, QC filters, and timing thresholds for interpolation/nearest.

Functions:
- `add_surface_position_from_traj(ds_cycles: xr.Dataset, ds_traj: xr.Dataset, *, cfg: SurfaceFixConfig | None = None) -> xr.Dataset`  
  Inputs: cycle dataset (must contain `t_surface_end`, `t_cycle_start`), TRAJ dataset with lat/lon/time, optional config. Output: new `ds_cycles` with `lat_surface_start`, `lon_surface_start`, `t_surface_start_fix`, `lat_surface_end`, `lon_surface_end`, `t_surface_end_fix`, `pos_source` ("interp"/"nearest"/"missing"), `t_pos_used`, `pos_age_s`, diagnostics (`dt_before_s`, `dt_after_s`, `gap_s`, `dt_nearest_s`, `t_fix_before/after/nearest`), plus `anchors_attendible` and updated `valid_for_bvp`. No files written. Assumptions: times decoded to datetime64[ns]; QC optional; interpolation only when time gaps within thresholds; nearest only within max dt; surface-before anchor uses the last valid fix before `t_cycle_start` within the surface-before window.

### writers.py
Purpose: file I/O helpers.

Functions:
- `write_netcdf(ds: xr.Dataset, path: str | Path) -> None`  
  Writes NetCDF to path; creates parent dirs.
- `write_parquet(ds: xr.Dataset, path: str | Path) -> None`  
  Requires pandas/pyarrow. Flattens dataset to DataFrame and writes Parquet; creates parent dirs.

Side effects: filesystem writes.

### runner.py
Purpose: orchestrate full preprocess run from AUX/TRAJ to outputs.

Classes:
- `PreprocessOutputs(ds_continuous: xr.Dataset, ds_cycles: xr.Dataset, ds_segments: xr.Dataset)` dataclass wrapper.

Functions:
- `run_preprocess(config_path: str | Path, out_dir: str | Path, *, write_parquet_products: bool = True, open_traj_file: bool = False) -> PreprocessOutputs`  
  Steps: load config; open AUX; optionally open TRAJ (sanity); build continuous dataset; build cycles/segments; open TRAJ (always for surface fixes); attach surface fixes; write NetCDF (and Parquet if enabled) to `out_dir`; returns datasets in memory. Side effects: filesystem writes.
- `main()`  
  CLI entry: parses args (`--config`, `--out`, `--no-parquet`, `--open-traj`), calls `run_preprocess`, prints destination.

Assumptions: input files exist; `outputs/` ignored by git; caller closes datasets if reusing.

### bvp_ready.py
Purpose: build a BVP-ready dataset that includes all IMU-attendible phases, with parking attendible as mandatory gate.

Classes:
- `BVPReadyConfig(acc_source: str = "acc_lin", acc_n_name: str | None = None, acc_e_name: str | None = None, min_parking_samples_for_bvp: int | None = None, min_phase_samples_for_bvp: int | None = None)`  
  Acc selection and optional threshold overrides (fall back to `ds_cycles` attrs/default=10).

Functions:
- `build_bvp_ready_dataset(ds_continuous: xr.Dataset, ds_cycles: xr.Dataset, ds_segments: xr.Dataset, *, cfg: BVPReadyConfig | None = None) -> xr.Dataset`  
  Inputs: preprocess outputs; config. Behavior:  
  * Requires `valid_for_bvp=True` (parking attendible and surface anchors present).  
  * Determines attendible phases per cycle via `<phase>_attendible`; includes all observations from attendible phases (parking, ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other as present). Non-attendible phases are omitted and expected vertical by the solver.  
  * Outputs Dataset with coords `obs`, `cycle`; per-obs vars `time`, `z_from_pres` (pressure), `acc_n/e` (chosen source), `phase_name`, `cycle_number_for_obs`, `cycle_index`, `obs_index`; per-cycle vars `row_start/row_size` (into obs), key times (`t_*`), surface fixes, phase attendible/counts, and cycle_number.  
  * Attrs: thresholds, platform, acc_source, phase_name_map, notes.  
  Early exit: if no valid cycles, prints message and exits with code 2 (caller via `main`).
- `main()`  
  CLI entry: args `--cont`, `--cycles`, `--segments`, `--acc-source`, `--acc-n-var`, `--acc-e-var`, `--out`; builds dataset, prints selection summary by phase, writes NetCDF, exits non-zero when no valid cycles.

Assumptions: NED accelerations; pressure used as depth proxy (units copied); time arrays datetime64[ns]; thresholds enforced per attrs or config; parking mandatory.

BVP-ready rule
--------------
- Parking attendible (samples ≥ `min_parking_samples_for_bvp`) is the gate: cycles failing this are excluded.  
- Other phases are included only if attendible (samples ≥ `min_phase_samples_for_bvp` per phase); omitted phases are treated as vertical (zero horizontal displacement) by the downstream solver.  
- The BVP-ready dataset therefore contains all available IMU-supported motion, with explicit phase labeling for filtering.
