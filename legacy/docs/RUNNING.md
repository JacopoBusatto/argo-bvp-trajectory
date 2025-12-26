Running the pipeline
====================

Installation
------------

Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Linux/Mac:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Minimal run (preprocess)
------------------------
From repo root, with a YAML config in `configs/`:
```bash
python -m argobvp.preprocess.runner --config configs/<platform>.yml --out outputs/preprocess
```
Expected outputs (written to `--out`):
- `<platform>_preprocessed_imu.nc` (continuous IMU)
- `<platform>_cycles.nc` (per-cycle keypoints, sampling/validity, surface fixes)
- `<platform>_segments.nc` (contiguous phase segments)
- Optional Parquet mirrors for cycles/segments (unless `--no-parquet` is used)

Build BVP-ready
---------------
After preprocess:
```bash
python -m argobvp.preprocess.bvp_ready \
  --cont outputs/preprocess/<platform>_preprocessed_imu.nc \
  --cycles outputs/preprocess/<platform>_cycles.nc \
  --segments outputs/preprocess/<platform>_segments.nc \
  --out outputs/preprocess
```
Behavior when 0 valid cycles: prints `No valid cycles for BVP-ready output (0/<N>).` and exits with non-zero code (no file written).

How to interpret validity
-------------------------
- `valid_for_bvp`: True if the parking phase is attendible (parking samples ≥ `min_parking_samples_for_bvp`). Parking is the mandatory gate.
- `parking_n_obs`: number of IMU samples in the parking phase for that cycle.
- `min_parking_samples_for_bvp`, `min_phase_samples_for_bvp` (attrs): thresholds used to decide attendibility for parking and other phases.
- `<phase>_n_obs` / `<phase>_attendible`: per-phase counts/flags (ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other). Phases below threshold are omitted from BVP-ready and should be treated as vertical by the solver.

Diagnostics
-----------
Plot surface track and attendible-phase acceleration/pressure (first valid cycles):
```bash
python examples/diagnostics/plot_bvp_ready_global.py \
  --cycles outputs/preprocess/<platform>_cycles.nc \
  --bvp outputs/preprocess/<platform>_bvp_ready.nc \
  --outdir diagnostics_plots   # omit --outdir to show interactively
```

Troubleshooting
---------------
- **File not found**: check `--config` paths and `--out` directory; ensure AUX/TRAJ paths in YAML are correct.
- **0 valid cycles**: parking phase may be under-sampled; inspect `parking_n_obs` and thresholds in `<platform>_cycles.nc`.
- **netCDF opening issues**: verify files exist and are not truncated; use `xr.open_dataset(path, decode_timedelta=False)` if CLOCK_OFFSET warnings appear (already set in `io_coriolis`).

CLI reference
-------------

Preprocess runner
```
python -m argobvp.preprocess.runner --help
```
Usage: `python -m argobvp.preprocess.runner --config <path.yml> --out <dir> [--no-parquet] [--open-traj]`
- `--config <path>`: YAML config (paths, calibration, thresholds). No default. Common mistake: wrong relative path.
- `--out <dir>`: output directory; created if missing. Common mistake: forgetting write permissions.
- `--no-parquet`: disable Parquet outputs for cycles/segments (NetCDF always written). Default: Parquet enabled.
- `--open-traj`: also open TRAJ for sanity; runner always re-opens TRAJ for surface fixes. Default: False.
See also: docs/PREPROCESS/OVERVIEW.md and docs/PREPROCESS/REFERENCE.md for pipeline details.

BVP-ready builder
```
python -m argobvp.preprocess.bvp_ready --help
```
Usage: `python -m argobvp.preprocess.bvp_ready --cont <imu.nc> --cycles <cycles.nc> --segments <segments.nc> --out <path-or-dir> [--acc-source acc_lin|acc|lin|raw] [--acc-n-var <name>] [--acc-e-var <name>]`
- `--cont <imu.nc>`: path to continuous IMU NetCDF.
- `--cycles <cycles.nc>`: path to cycles NetCDF (must include validity/attendibility fields).
- `--segments <segments.nc>`: path to segments NetCDF.
- `--out`: output file or directory; if directory, writes `<platform>_bvp_ready.nc`.
- `--acc-source`: choose acceleration source; default `acc_lin` (gravity-removed). Common mistake: picking `acc` when linear is desired.
- `--acc-n-var` / `--acc-e-var`: explicit variable overrides; use if your IMU vars are non-standard.
Behavior on 0 valid cycles: prints message and exits non-zero (no file written). Parking attendible is mandatory; other phases included only if attendible (see docs/PREPROCESS/SAMPLING_VALIDITY.md).
