ARGO-BVP-TRAJECTORY
==================

This repository contains a small, self-consistent numerical framework
to study the integration and reconstruction of Argo float trajectories
using second-order kinematics.

The final scientific objective is to quantify the positional error
at the end of the diving and parking phases of an Argo float,
with respect to the known surface GPS positions at the beginning
and at the end of a cycle.

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
│   └── integrators.py
│
├── tests/
│   ├── test_integrators_constant_acc.py
│   └── test_convergence_dt.py
│
├── examples/
│   ├── convergence_visual.py
│   └── trajectory_visual.py
│
├── pyproject.toml
├── .gitignore
└── README.md (this document)


------------------------------------------------------------
2. NUMERICAL MODEL
------------------------------------------------------------

The motion of a particle (or float) is described by a second-order
kinematic system:

    dr/dt = v
    dv/dt = a(t, r, v)

where:
- r(t) is the position vector (x, y, z)
- v(t) is the velocity vector
- a(t, r, v) is a prescribed acceleration

The acceleration may represent:
- a synthetic analytical forcing (for tests)
- a reconstructed acceleration from observations
- in future steps, a combination of physical processes


------------------------------------------------------------
3. INTEGRATION MODULE (src/argobvp/integrators.py)
------------------------------------------------------------

The core function is:

    integrate_2nd_order(t, r0, v0, a_fun, method, backward=False)

It integrates the second-order system on a given time grid `t`.

Available integration schemes:

- EULER
    Explicit Euler scheme.
    Equivalent to "rectangle / left rule" integration.
    First-order accurate in time.

- TRAPEZOID
    Predictor-corrector trapezoidal scheme.
    Second-order accurate in time.
    This is the main reference method for Argo applications.

- RK4
    Classical fourth-order Runge-Kutta.
    Used mainly as a high-accuracy benchmark.

The same function supports:

- Forward integration:
    Given (r0, v0) at t[0], integrate up to t[-1].

- Backward integration:
    Given (rN, vN) at t[-1], integrate backward to t[0].
    Internally, this is handled by reversing the time grid,
    without changing the numerical scheme.


------------------------------------------------------------
4. AUTOMATED TESTS (tests/)
------------------------------------------------------------

All tests are run with pytest.

4.1 Constant acceleration test
--------------------------------

File:
    tests/test_integrators_constant_acc.py

Purpose:
- Verify correctness against an analytical solution
  for constant acceleration.
- Verify consistency between forward and backward integration.

This test checks that:
- RK4 reproduces the analytical solution with very small error.
- Forward integration followed by backward integration
  reconstructs the initial conditions within tolerance.


4.2 Convergence test (Δt → 0)
------------------------------

File:
    tests/test_convergence_dt.py

Purpose:
- Verify the numerical order of each integration scheme.
- Compare "rectangles vs trapezoids" in a quantitative way.

Method:
- Use a smooth, non-polynomial acceleration:
      a(t) = A sin(ω t)
- Compute the endpoint error as a function of Δt.
- Fit the slope in log-log space.

Expected behavior:
- Euler (rectangles):     error ~ O(Δt¹)
- Trapezoid:              error ~ O(Δt²)
- RK4:                    error ~ O(Δt⁴)

This test provides the numerical basis for error-budget estimates.


------------------------------------------------------------
5. VISUAL EXAMPLES (examples/)
------------------------------------------------------------

The examples directory contains scripts meant for
interactive and visual inspection of the numerical behavior.
These scripts are NOT part of the automated test suite.

5.1 Convergence visualization
------------------------------

File:
    examples/convergence_visual.py

What it does:
- Computes endpoint errors for different Δt values.
- Produces a log-log plot of error vs Δt.
- Prints estimated convergence slopes.

This script visually confirms:
- the superiority of trapezoidal integration over rectangles
- the expected order of each method.


5.2 Trajectory visualization
----------------------------

File:
    examples/trajectory_visual.py

What it does:
- Compares numerical trajectories with an analytical solution.
- Shows:
    - error as a function of time
    - 3D trajectories (true vs numerical)
- Compares coarse vs fine time discretization.

This script is particularly useful to build intuition
about error accumulation along the trajectory.


------------------------------------------------------------
6. ENVIRONMENT AND REPRODUCIBILITY
------------------------------------------------------------

The project uses a local Python virtual environment:

    .venv/

The environment is NOT committed to git.

Dependencies are declared in pyproject.toml.
To recreate the environment:

    py -3.11 -m venv .venv
    .venv\\Scripts\\activate
    pip install -e .[dev]

To run tests:

    pytest

To run visual examples:

    python examples/convergence_visual.py
    python examples/trajectory_visual.py


------------------------------------------------------------
7. FUTURE EXTENSIONS
------------------------------------------------------------

Planned next steps include:

- Boundary Value Problem (BVP) formulation:
    reconstruction of trajectories given surface constraints.

- Piecewise Argo trajectories:
    descent → parking → ascent.

- Use of real Argo-derived accelerations
    and estimation of final position error.

- Quantification of uncertainty at:
    - end of descent
    - beginning of ascent
    relative to surface GPS fixes.

This document will evolve together with the code.
