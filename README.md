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

argo-bvp-trajectory/
│
├── src/argobvp/
│   ├── __init__.py
│   ├── integrators.py
│   ├── bvp.py
│   └── metrics.py
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
└── README.md (this document)

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
8. ROADMAP / NEXT STEPS
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
