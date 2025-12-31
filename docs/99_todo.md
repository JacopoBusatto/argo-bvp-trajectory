# Roadmap (TODO)

This roadmap lists concrete next steps for moving from the current synthetic
pipeline to a real-data workflow.

## 1) Body-frame → ENU using gyro + magnetometer

**Goal**: rotate IMU accelerations from body frame into ENU.

Deliverables:
- Implement a rotation pipeline (bias handling + attitude estimation).
- Use gyro integration + magnetometer correction (e.g., complementary filter).
- Add tests with synthetic rotations and known ground truth.
- Extend cycle schema or metadata to store orientation if needed.

## 2) Real-data workflow (multi-cycle, QC, reporting)

**Goal**: process real TRAJ/AUX across multiple cycles with reproducible QC.

Deliverables:
- Batch preprocessing for many cycles.
- QC report per cycle (anchor selection, gaps, invalid segments).
- Summary tables and plots (per-cycle RMS, anchor deltas).
- CLI commands to manage datasets and export results.

## 3) Sweep analysis outputs

**Goal**: stabilize the sweep analysis into a standard evaluation report.

Deliverables:
- Finalize metric definitions and thresholds.
- Publish summary plots (heatmaps + trajectories) with consistent styling.
- Add a compact report generator (Markdown or HTML).

## 4) Packaging and reproducibility

**Goal**: make the tool easy to install and run for collaborators.

Deliverables:
- Pin dependencies and document supported Python versions.
- Provide a minimal example dataset.
- Add CI checks for tests and basic linting.
