## Overview

This project reconstructs submerged trajectories of ARGO floats by integrating IMU
accelerations and solving a boundary value problem (BVP) constrained by GPS fixes
at the start and end of each cycle.

The current focus is a minimal, solid foundation: a cycle-file NetCDF schema
for a single dive cycle, along with validation and I/O helpers.

## Pipeline (high level)

1) Raw sources (TRAJ, AUX, IMU) -> parsed anchors + time series
2) Cycle file (NetCDF schema v1) -> validated, analysis-ready dataset
3) BVP integration -> reconstructed submerged trajectory
4) Outputs -> trajectories, diagnostics, and derived metrics
