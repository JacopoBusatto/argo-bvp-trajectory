ARGO-BVP-TRAJECTORY
===================

This repository contains a small, self-consistent numerical framework to:
1) integrate Argo-like trajectories from acceleration (second-order kinematics), and
2) solve a simple boundary-value constraint (BVP) by “shooting” on the unknown initial velocity.

The final scientific objective is to quantify the positional error at key Argo phases
(end of descent, start of ascent) with respect to surface GPS fixes at the start/end of a cycle.

The project is intentionally built step by step, with strong emphasis on:
- numerical correctness
- reproducibility
- verification through analytical and synthetic test cases


------------------------------------------------------------
1. PROJECT STRUCTURE
------------------------------------------------------------

The repository follows a standard "src-layout" Python structure:

```html
argo-bvp-trajectory/
├── src/
│   └── argobvp/
│       ├── __init__.py
│       ├── integrators.py
│       ├── bvp.py
│       └── metrics.py
│
├── tests/
│   └── test_*.py
│
├── examples/
│   ├── convergence_visual.py
│   ├── bvp_shooting_visual.py
│   └── argo_toy_visual.py
│
├── pyproject.toml
├── .gitignore
└── README.md


Public API is re-exported in `argobvp/__init__.py`:
- integrate_2nd_order, IntegratorMethod
- endpoint_error, point_error_at_time, nearest_index
(see src/argobvp/__init__.py). :contentReference[oaicite:3]{index=3}


------------------------------------------------------------
2. NUMERICAL MODEL
------------------------------------------------------------

We integrate second-order kinematics:

    dr/dt = v
    dv/dt = a(t, r, v)

where:
- r(t) is the position vector (x, y, z)
- v(t) is the velocity vector
- a(t, r, v) is a prescribed acceleration model

This is implemented as a discrete-time integrator on an arbitrary time grid `t`
(assumed strictly increasing).


------------------------------------------------------------
3. CORE MODULES (src/argobvp/)
------------------------------------------------------------

3.1 integrators.py
------------------

Main function:

    integrate_2nd_order(t, r0, v0, a_fun, method="trapezoid", backward=False)

It integrates the system on a given time grid `t`.
Supported integration schemes (IntegratorMethod):

- EULER
  Explicit Euler (rectangle/left rule). First-order accurate.
  (Update v with a(t_i), update r with v(t_i)). :contentReference[oaicite:4]{index=4}

- TRAPEZOID
  Predictor-corrector trapezoid on v, and trapezoid on r using average velocity.
  Second-order accurate in time; the main reference for Argo-like workflows. :contentReference[oaicite:5]{index=5}

- RK4
  Classical 4th-order Runge–Kutta for the coupled (r,v) system; used as benchmark. 

Backward integration:
If `backward=True`, the code reverses the time grid internally, integrates,
then flips the output so arrays are returned aligned with the original (increasing) `t`. 


3.2 bvp.py
----------

This module relies on `scipy.optimize.root` for the nonlinear solve,
therefore SciPy is required.

Implements a simple Boundary Value constraint on position via “shooting” on v0.

Main function:

    shoot_v0_to_hit_rT(t, r0, rT_target, v0_guess, a_fun, method="trapezoid", dims=(0,1))

Goal:
Choose the components v0[dims] so that the integrated endpoint matches the target:

    r(T)[dims] = rT_target[dims]

Implementation:
- Define a residual F(v0[dims]) = r(T)[dims] - rT_target[dims]
- Solve F(x)=0 using `scipy.optimize.root` (hybr). :contentReference[oaicite:8]{index=8}
- Return the optimized v0 and the reconstructed (r(t), v(t)) trajectory.

This is the prototype of an Argo-cycle BVP:
for example with prescribed z-profile, we typically shoot only on XY (dims=(0,1)). :contentReference[oaicite:9]{index=9}


3.3 metrics.py
--------------

Small utilities used across tests and examples:

- nearest_index(t, t_query): nearest time index :contentReference[oaicite:10]{index=10}
- endpoint_error(r, r_target): ||r(T) - r_target|| :contentReference[oaicite:11]{index=11}
- point_error_at_time(t, r, t_query, r_target): ||r(t_query) - r_target|| :contentReference[oaicite:12]{index=12}


------------------------------------------------------------
4. TEST SUITE (tests/)
------------------------------------------------------------

Tests are run with pytest and are designed to be:
- fast
- deterministic
- verification-oriented

Current tests include:
- analytical validation with constant acceleration
- forward/backward integration consistency
- convergence order vs Δt (Euler / Trapezoid / RK4)
- Argo-style phase error diagnostics on synthetic cycles
- BVP shooting correctness in XY

------------------------------------------------------------
5. VISUAL EXAMPLES (examples/)
------------------------------------------------------------

These scripts are for interactive inspection and intuition-building.
They are NOT part of the automated test suite.

5.1 examples/convergence_visual.py
----------------------------------

Purpose:
Show convergence with Δt → 0 and compare integrators (rectangles vs trapezoid vs RK4).

What it does:
- Uses a smooth acceleration a(t) = A sin(ω t)
- Computes an endpoint error (position + velocity mismatch) for a sequence of dt values
- Produces a log-log plot and prints fitted slopes p in error ~ O(dt^p). :contentReference[oaicite:13]{index=13}

Run:

    python examples/convergence_visual.py


5.2 examples/bvp_shooting_visual.py
-----------------------------------

Purpose:
Demonstrate a BVP in XY solved by shooting on v0_xy.

High-level workflow:
1) Define a “truth” XY curve (drift + low-frequency meander + high-frequency oscillation)
2) Build a consistent analytic acceleration a_true(t) (second derivative of the truth)
3) Generate a target endpoint rT_target using a fine RK4 “truth integration”
4) On a coarser grid, solve for v0_xy such that r(T) matches rT_target (for each integrator)
5) Plot:
   - XY trajectories (truth vs reconstructed)
   - reconstruction error ||r_xy - r*_xy|| over time
   - cost surface J(v0x,v0y) = ||r_xy(T; v0) - rT_xy|| 

Run:

    python examples/bvp_shooting_visual.py


5.3 examples/argo_toy_visual.py
-------------------------------

Purpose:
A “toy Argo” visual demo that bundles:
- the same non-degenerate XY truth family,
- shooting on v0_xy,
- and a cost-surface visualization implemented as a helper function `cost_surface(...)`. 

Run:

    python examples/argo_toy_visual.py


------------------------------------------------------------
6. INSTALLATION & ENVIRONMENT
------------------------------------------------------------

Python requirement:
- Python 3.10 or 3.11 (see `pyproject.toml`)

Create and activate venv (Windows PowerShell):

    py -m venv .venv
    .venv\Scripts\activate

Install the project (editable) + development tools:

    pip install -e ".[dev]"

Run tests:

    pytest -q

Run examples:

    python examples/convergence_visual.py
    python examples/bvp_shooting_visual.py
    python examples/argo_toy_visual.py


------------------------------------------------------------
7. DESIGN CHOICES (WHY THESE METHODS?)
------------------------------------------------------------

- Euler is included as the simplest baseline (“rectangles / left rule”) and to show first-order behavior.
- Trapezoid is the default reference because it is:
  - simple,
  - robust,
  - second-order accurate,
  - a natural “next step” from rectangles for Argo-like discrete sampling.
- RK4 is included primarily as a high-accuracy benchmark to define “truth” trajectories.

------------------------------------------------------------
8. VERTICAL INTEGRATION MODES (z-dynamics)
------------------------------------------------------------

Two distinct integration modes are provided on purpose.

They correspond to two different physical assumptions:

- sampled mode: integrate exactly what the IMU measures (discrete az samples)
- continuous mode: integrate a reconstructed continuous az(t)

The two modes solve different problems and should not be mixed.

In this project, the vertical coordinate z (positive downward)
can be reconstructed using different assumptions on the available data
and on the physical meaning of the acceleration signal.

This is particularly important for Argo floats equipped with IMU
(accelerometer + gyroscope), where vertical motion can be inferred
both from pressure measurements and from vertical acceleration.


------------------------------------------------------------
8.1 Sampled acceleration integration (IMU-consistent)
------------------------------------------------------------

Function:
    integrate_z_from_accel_samples(...)

Location:
    src/argobvp/z_sources.py

This function integrates the vertical motion assuming that the vertical
acceleration az is available as *discrete samples* at times t[k],
as provided by an IMU.

The governing equations are:

    dz/dt  = vz
    dvz/dt = az(t)

but az(t) is assumed to be known only at the sampling instants.

Supported numerical schemes:

- Euler (rectangle rule):
      vz[k+1] = vz[k] + dt * az[k]
      z[k+1]  = z[k]  + dt * vz[k]

- Trapezoid:
      vz[k+1] = vz[k] + dt * (az[k] + az[k+1]) / 2
      z[k+1]  = z[k]  + dt * (vz[k] + vz[k+1]) / 2

Important properties:

- No interpolation of az is performed.
- The method operates directly on sampled data.
- This is the most appropriate approach when az comes from IMU measurements.
- The comparison between Euler and Trapezoid is clean and reproducible,
  since it reflects only the numerical quadrature choice.

Limitations:

- RK4 is intentionally NOT supported in this mode, because it requires
  evaluating az at intermediate times, which is not defined for purely
  sampled signals.

This mode is therefore the recommended default for Argo-like IMU data.


------------------------------------------------------------
8.2 Continuous acceleration integration (model-based)
------------------------------------------------------------

Function:
    integrate_z_from_accel(...)

Location:
    src/argobvp/z_sources.py

In this mode, the sampled acceleration az is implicitly promoted to a
continuous function az(t) via interpolation.

The vertical ODE system is then integrated assuming that az(t) can be
evaluated at arbitrary times, including intermediate stages of the
integration scheme.

Supported numerical schemes include:

- Euler
- Trapezoid
- RK4

Important characteristics:

- The numerical result depends both on the integration scheme AND on
  the chosen interpolation model for az(t).
- This mode is suitable when az(t) represents a reconstructed or modeled
  continuous signal (e.g. filtered IMU data, spline reconstruction,
  or analytical forcing in synthetic tests).
- RK4 can be used as a high-accuracy benchmark under these assumptions.

Caveat:

- Comparisons between Euler, Trapezoid, and RK4 in this mode do NOT isolate
  the effect of the integration scheme alone, since the interpolation of
  az(t) plays a central role.
- In particular, Euler may occasionally outperform Trapezoid on specific
  metrics due to cancellation effects or interpolation artifacts.

For this reason, this mode is mainly used for:
- synthetic benchmarks
- sensitivity experiments
- comparison against analytical solutions

Key conceptual difference:

- integrate_z_from_accel_samples integrates discrete data:
      "What does the IMU actually measure?"

- integrate_z_from_accel integrates a continuous model:
      "What would the trajectory be if az(t) were a smooth function?"

Both are correct, but they answer different questions.


------------------------------------------------------------
8.3 Relation to pressure-based depth
------------------------------------------------------------

In real Argo datasets, depth (or pressure) provides an independent
measurement of z(t).

The framework is designed so that:

- z(t) reconstructed from acceleration can be directly compared to
  z(t) derived from pressure.
- Tests can be constructed to quantify the error introduced by:
    - numerical integration
    - acceleration sampling
    - bias or noise in az

Future developments will allow:
- switching between pressure-based z and acceleration-based z
- using pressure-derived z as a constraint in boundary value problems
  for trajectory reconstruction


------------------------------------------------------------
8.4 Recommended usage summary
------------------------------------------------------------

For real Argo IMU data:
    → use integrate_z_from_accel_samples

For synthetic tests or continuous forcing models:
    → use integrate_z_from_accel

For method-comparison tests (Euler vs Trapezoid):
    → always use sampled mode

For high-accuracy reference solutions:
    → use RK4 in continuous mode

This separation ensures numerical clarity, reproducibility,
and physical interpretability of the results.

------------------------------------------------------------
9. ROADMAP / NEXT STEPS
------------------------------------------------------------

Planned next steps (already discussed in the project):

1) Prescribed z-profile variant:
   - solve only for XY while z(t) is prescribed by an Argo-like depth profile.

2) Longer convergence tests:
   - error vs Δt curves specifically at key phases (end-of-descent / start-of-ascent / final)

3) Real Argo integration:
   - given measured a(t) (or reconstructed), forward-integrate and quantify endpoint/phase errors
     relative to surface GPS positions.

This document will evolve together with the code.

------------------------------------------------------------
10. Vertical integration tests
------------------------------------------------------------

Specific tests are included to validate vertical (z) reconstruction
under Argo-like conditions.

These tests focus on physically meaningful diagnostics rather than
formal numerical order only.

Main vertical tests include:

- test_z_from_accel_samples.py
    Verifies vertical reconstruction when az is treated as sampled IMU data.
    Demonstrates that trapezoidal integration outperforms Euler
    in a robust, sampling-consistent setting.

- test_z_phase_errors_vs_dt.py
    Quantifies vertical reconstruction errors at key Argo phases:
        - end of descent
        - start of ascent
        - end of cycle
    as a function of the sampling interval Δt.

    Errors are evaluated using time interpolation to avoid
    grid-alignment artefacts.

These tests are designed to be:
- robust to phase-node alignment,
- representative of real Argo sampling,
- suitable for uncertainty budgeting.

------------------------------------------------------------
11. Vertical reconstruction examples
------------------------------------------------------------

Additional visual scripts focus on vertical (z) dynamics.

- examples/z_phase_errors_visual.py

Purpose:
Visualize how vertical reconstruction errors depend on:
- integration method,
- sampling interval Δt,
- Argo phase (descent end, ascent start, final).

The script shows:
- reconstructed z(t) vs analytic reference,
- absolute error |z - z_truth| over time,
- phase errors vs Δt (log-log),
- separation between Euler, Trapezoid, and RK4 in appropriate regimes.

The example highlights the difference between:
- sampled-data integration (IMU-consistent),
- continuous-forcing integration (benchmark).

These scripts are meant for diagnostic exploration and
scientific interpretation, not for automated validation.
