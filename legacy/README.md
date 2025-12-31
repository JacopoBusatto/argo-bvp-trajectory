ARGO-BVP-TRAJECTORY
===================

What this project does today
----------------------------
- Calibrates Argo IMU AUX data, estimates attitude (safe tilt-only), rotates accelerations to NED, and removes gravity.
- Segments observations into phases, derives per-cycle keypoints, and attaches surface position fixes from TRAJ.
- Flags cycles usable for BVP based on parking-phase sampling density (parking phase mandatory); builds a BVP-ready view that includes all IMU-attendible phases (parking, ascent, descent, deep/profile, surface/in-air when sampled).
- Provides lightweight kinematics/integration helpers (`integrators.py`, `bvp.py`) for downstream experiments.
- Handles the case where some floats yield zero valid cycles (e.g., parking under-sampled); the BVP-ready step will exit early and report it.

What is not implemented yet
---------------------------
- End-to-end trajectory reconstruction/BVP solution across datasets (inputs are prepared; solver remains minimal).
- Advanced phase inference (e.g., reconstructing missing phases from TRAJ timing) or surface-drift corrections.
- Final gyro scale determination for all platforms (current defaults are usable but provisional).

Docs
----
See `docs/README.md` for the documentation index:
- Preprocess overview and reference
- Sampling/validity rules for phase selection (parking mandatory; other phases attendible when IMU-sampled)

Quick start
-----------
Install (with optional dev tools):
```
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Run preprocessing:
```
python -m argobvp.preprocess.runner --config configs/<platform>.yml --out outputs/preprocess
```

Build a BVP-ready file (all attendible phases, parking required):
```
python -m argobvp.preprocess.bvp_ready \
  --cont outputs/preprocess/<platform>_preprocessed_imu.nc \
  --cycles outputs/preprocess/<platform>_cycles.nc \
  --segments outputs/preprocess/<platform>_segments.nc \
  --out outputs/preprocess
```

Project layout (simplified)
---------------------------
- `src/argobvp/` — core kinematics/BVP helpers and preprocessing pipeline
- `docs/` — documentation index and preprocess guides
- `configs/` — YAML configs (paths + calibration)
- `examples/` — visualization and diagnostics scripts
- `outputs/` — generated files (git-ignored)
