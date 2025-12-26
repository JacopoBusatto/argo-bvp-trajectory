Synthetic Coriolis-like benchmark
=================================

This synthetic generator builds AUX/TRAJ files with 30 cycles at varying IMU sampling rates to benchmark the preprocess and BVP-ready pipeline.

Files generated (under `outputs/synthetic/`)
- `SYNTHETIC01_AUX.nc`: AUX-like IMU dataset (counts scaled for identity calibration).
- `SYNTHETIC01_TRAJ.nc`: TRAJ-like surface fixes (+20 min after surface end).
- `config_synthetic.yml`: config pointing to AUX/TRAJ with identity calibration and thresholds.
- `ground_truth.nc`: key times and true N/E displacements per cycle.
- `preprocess/`: outputs from running the preprocess and BVP-ready steps.

How to run
----------
From repo root:
```bash
python examples/synthetic/make_synthetic_coriolis_benchmark.py
```

What it does
------------
- Builds 30 cycles: 10 sampling levels × 3 durations (1, 5, 10 days).
- Descent/ascent: 3 h each; parking fills the remainder; IMU always ON in parking.
- Sampling intervals per level: descent/ascent from 1800 s down to 1 s; parking from 3600 s down to 60 s.
- Generates depth-dependent horizontal velocities and integrates positions; produces realistic pressures and attitude; writes AUX/TRAJ + ground truth.
- Runs the preprocess runner and BVP-ready builder, printing a brief summary (valid cycles, per-phase attendibility) and the size of the BVP-ready output.

Notes
-----
- Output directory: `outputs/synthetic/` (created if missing).
- Calibration is identity; counts are already scaled to desired physical units.
- Parking is always attendible; higher sampling levels yield denser ascent/descent samples.
