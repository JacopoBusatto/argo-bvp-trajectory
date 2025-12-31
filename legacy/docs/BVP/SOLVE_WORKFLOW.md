Solve workflow (first-pass)
===========================

Scope
-----
This describes how to run the first working solver pipeline on BVP-ready outputs and what the outputs mean. It integrates horizontal kinematics over attendible phases, with parking as the mandatory gate. It does **not** implement an advanced BVP solver; it uses a simple shooting update when a surface end fix is available.

Inputs
------
- `*_bvp_ready.nc` from preprocess (`valid_for_bvp=True` cycles only). Required vars: `time`, `z_from_pres`, `phase_name`, `cycle_number`, `cycle_number_for_obs`, horizontal accelerations (`acc_n`, `acc_e` by default; can override).
- Optional: `*_cycles.nc` for surface fixes if not present in BVP-ready.

Assumptions
-----------
- Parking is attendible (mandatory). Other phases (ascent, descent, profile, surface, etc.) are included only if attendible; missing phases are treated as vertical (zero horizontal displacement) by the solver, consistent with `SAMPLING_VALIDITY`.
- Local tangent-plane approximation around the surface fix; pressure is used as depth proxy (z positive downward ≈ dbar).
- Time must increase; duplicate timestamps at phase boundaries are dropped.

How to run
----------
1) Ensure preprocess outputs and BVP-ready exist (see `docs/RUNNING.md` for preprocess and BVP-ready commands).
2) Run solver (CLI):
```bash
python -m argobvp.solve.runner \
  --bvp-ready outputs/preprocess/<platform>_bvp_ready.nc \
  --cycles outputs/preprocess/<platform>_cycles.nc \
  --out outputs/solve/<platform>_solved.nc
```
Options:
- `--acc-n-var`, `--acc-e-var`: choose acceleration variables from BVP-ready (default: `acc_n`, `acc_e`, typically gravity-removed).
- `--cycles`: optional, used to fetch surface fixes if missing in BVP-ready.

What the solver does
--------------------
- For each cycle, slices observations from BVP-ready and maps `phase_name` → macro phases (parking/ascent/descent/surface/profile/other).
- Requires parking data; otherwise the cycle is skipped.
- Chooses a local reference at the surface fix (if available) and integrates horizontal kinematics `v' = a`, `r' = v` with simple per-step updates.
- If ascent data and a surface end fix exist: applies a one-step shooting update for initial velocity `v0` using the boundary condition at surface end; records misfit.
- If no ascent fix: uses zero initial velocity (IVP-only) and marks `used_bvp=False`.
- Stores per-phase sampling stats (n_obs, median dt), integration_type (which phases were present), and simple metrics (misfit_end, delta_start/end).
- Depth is carried from pressure (`z_m ≈ pres dbar`, positive downward).

Outputs
-------
- NetCDF: `outputs/solve/<platform>_solved.nc`
- Coordinates: `obs`, `cycle`.
- Per-observation vars: `time`, `phase_name`, `macro_phase`, `x_east_m`, `y_north_m`, `z_m`, `lat`, `lon`, `cycle_number`.
- Per-cycle vars: `integration_type`, `has_descent_data`, `has_parking_data`, `has_ascent_data`, `used_bvp`, `misfit_end_m`, `delta_start_m`, `delta_end_m`, `T_total_s`, per-phase `*_n_obs` / `*_median_dt_s`, `lon0`, `lat0`.
- Attributes: platform, acc_source, notes.

Quick inspection
----------------
- Run `examples/solve/plot_3d_solved.py --solved outputs/solve/<platform>_solved.nc` to view a 3D ENZ plot (colored by macro_phase) for selected cycles.
- Metrics to check:
  - `misfit_end_m`: residual to surface fix when BVP was applied (NaN if no boundary).
  - `integration_type`: which phases contributed.
  - Phase counts/median dt: check sampling density.

Limitations
-----------
- Simple tangent-plane geometry; ignores Coriolis/Earth curvature beyond local approximation.
- Vertical dynamics not integrated; pressure used directly as depth proxy.
- BVP step is a single-shot linear update; no iterative refinement or constraints.
- Surface anchors: if only surface end is available, start is approximated there (vertical assumption for descent).
