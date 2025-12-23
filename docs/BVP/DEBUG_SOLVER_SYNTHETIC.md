# Synthetic BVP Debug Status

## 1) Context and goal
We are building a synthetic Coriolis-like benchmark to validate the Argo BVP trajectory solver. All synthetic cycles are replicas of the **same physical truth trajectory**; only IMU sampling frequency and (optionally) noise differ. This allows isolating solver behaviour from environmental variability.

## 2) What is now working
- **Analytic ground-truth generator**: produces a curved 3D trajectory with spiral descent/ascent and a curved parking arc; accelerations are physically consistent (~1e-3 m/s²).
- **Preprocessing pipeline**: synthetic body accelerations are passed through to NED without spurious rotation; `bvp_ready` contains realistic `acc_e` / `acc_n`.
- **Diagnostics**: truth vs solved plots, anchor comparison plots, and acceleration histograms are emitted for post-run inspection.

## 3) Critical problems identified

### 3.1 Wrong start anchor (conceptual bug)
- The solver currently enforces (x, y) = (0, 0) at **park_start**.
- Correct anchor must be **immersion** (surface_before end / start of descent).
- This mis-anchoring forces all cycles to coincide at parking start and breaks the intended geometry.

### 3.2 Non-monotonic time axis (numerical bug)
- `dt` values inside the solver include large negative and positive jumps (order ±86400 s).
- Integration runs on an unsorted / non-monotonic time axis.
- This yields straight-line trajectories and huge coordinates (~1e15 m), despite curved truth and nonzero accelerations.

## 4) Evidence
- `dt` statistics were explicitly computed and show large sign changes.
- `acc_e` / `acc_n` in `bvp_ready` have the correct magnitude (~0.002 m/s²), confirming preprocessing is OK.
- Solved trajectories remain straight despite the curved truth, indicating the issue lies in anchoring and time handling inside the solver.

## 5) What must be fixed next
- Solver must:
  - Anchor the BVP at **immersion** (surface_before end), not park_start.
  - Guarantee a **strictly monotonic time axis** before integration.
- No further changes to the synthetic generator are required until these solver fixes are in place.

## 6) Current repository status
- The synthetic generator and preprocessing are considered correct for this benchmark.
- Further work should focus exclusively on the solver to address the anchoring and time-ordering bugs.
