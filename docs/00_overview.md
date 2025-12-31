# Overview

This project reconstructs submerged trajectories of ARGO floats using onboard IMU
accelerations and boundary constraints from GPS. The goal is to estimate the
underwater path and quantify the error between the reconstructed parking segment
and the observed surface positions at dive and emerge.

## Coordinate frame and anchors

- Local frame: a tangent-plane ENU (east, north, up) frame, derived from a reference
  latitude/longitude (`lat0`, `lon0`) stored in the cycle file.
- Surface anchors:
  - **dive** = last GPS fix with acceptable QC in the first surface window.
  - **emerge** = first GPS fix with acceptable QC in the following surface window.
- ENU ↔ lat/lon conversion uses a small-angle equirectangular approximation, valid
  for km-scale trajectories.

## Instrument configuration

Instrument-specific conversions are centralized in `src/argo_bvp/instruments/`:

- `InstrumentParams`: `lsb_to_ms2`, `gyro_lsb_to_rads`, `mag_lsb_to_uT`.
- `INSTRUMENTS` registry: add a new entry to represent a new sensor calibration.

In preprocessing, counts are converted to SI using the selected instrument. This
is the single entry point for switching between synthetic and real sensors.

## What is integrated

- Horizontal motion (`x`, `y`) is reconstructed from linear accelerations.
- Vertical motion is **not** integrated: `z` is derived from pressure and used
  for phase classification only.

## BVP integration (high level)

We solve a 1D boundary value problem (Fubini formulation) for each horizontal
axis, constrained by the dive and emerge anchor positions. The method integrates
accelerations over time and enforces endpoint positions, making it robust to
drift in double integration.

## Pipelines

Synthetic pipeline (current):
1) `synth` → generate TRUTH, TRAJ, AUX (Coriolis-like)
2) `preprocess` → build a cycle NetCDF with anchors and IMU in SI
3) `integrate` → reconstruct ENU trajectory from the cycle file
4) `sweep` → grid of parameter variations (synth → preprocess → integrate)
5) `analyze-sweep` → metrics + heatmaps + trajectory plots

Instrument/data pipeline (target, partial):
1) `preprocess` → build cycle files from real TRAJ/AUX
2) `integrate` → reconstruct ENU trajectory
3) `analysis` → metrics and diagnostics

NOTE (IT): la pipeline su dati reali non è ancora completa; serve la parte
body-frame → ENU e l’ingestione multi-ciclo.

## Current limits and open items

- No body-frame → ENU rotation (gyro/mag not used yet).
- Integration is 2D only; vertical is pressure-derived.
- Real-data ingestion and multi-cycle workflows are not implemented.
- See `docs/99_todo.md` for the roadmap.
