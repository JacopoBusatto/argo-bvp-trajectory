# Sampling assumptions and cycle validity rules for Argo-IMU preprocessing

## Scope of this document

This document defines the **sampling assumptions**, **cycle validity criteria**, and
**integration strategies** adopted in the `argobvp` preprocessing pipeline.

The preprocessing stage does **not** reconstruct trajectories.
Its sole purpose is to prepare raw Argo-IMU and trajectory data so that they can be:

1. integrated by a downstream Boundary Value Problem (BVP) solver, and  
2. used to estimate position errors at physically meaningful key events
   (e.g. start/end of the parking phase).

Only concepts strictly required to achieve these goals are defined here.

---

## Physical and observational context

Argo cycles may last from several hours up to **multiple days (up to ~10 days)**.

IMU sampling is **not continuous over the entire cycle**:
- the IMU may be switched off during descent,
- the IMU may be switched off during ascent,
- in some cycles, the IMU may not sample the parking phase at all.

As a consequence, **trajectory integration cannot be assumed possible over the full cycle**,
and must be restricted to phases where dynamical measurements are available.

---

## Definition of the parking phase

The **parking phase** is defined as the time interval between:

- `t_park_start`: first timestamp associated with the parking measurement code,
- `t_park_end`: last timestamp associated with the parking measurement code  
  (or, equivalently, the start of ascent when explicitly identified).

The parking phase is the **only phase of the Argo cycle** where:
- long-term horizontal motion is expected,
- the duration is sufficiently long to accumulate measurable displacement,
- IMU sampling (when present) provides meaningful dynamical information.

---

## Fundamental validity criterion

> **A cycle is considered valid for trajectory reconstruction if and only if the IMU samples the parking phase.**

Formally:

- if IMU data are available during the parking phase → the cycle is **valid**,
- if the parking phase is not sampled by the IMU → the cycle is **invalid** and excluded.

No attempt is made to integrate trajectories for cycles that do not satisfy this criterion.

---

## Assumptions for unsampled phases

### Descent phase

If the IMU is switched off during descent:

> The horizontal displacement during descent is assumed to be negligible.

Therefore, the position at the **start of the parking phase** is assumed to coincide with
the last known surface position prior to descent.

Formally:
\[
\mathbf{r}_{\text{park,start}} := \mathbf{r}_{\text{surface,start}}
\]

This assumption is:
- physically plausible for Argo floats (quasi-vertical descent),
- explicitly declared,
- only applied when IMU data are unavailable.

---

### Ascent phase

If the IMU is switched off during ascent:

> The horizontal displacement during ascent is not reconstructed.

The position at the **end of the parking phase** is associated with the surface position
measured after emersion, using a nearest-neighbour or controlled interpolation strategy.

Formally:
\[
\mathbf{r}_{\text{park,end}} := \mathbf{r}_{\text{surface,end}}
\]

The temporal offset between the physical event (`t_park_end`) and the observation
(`t_pos_used`) is explicitly tracked and stored.

---

## Accepted and rejected cycle types

Based on the rules above, only two classes of cycles are defined.

### Valid cycles

A cycle is **valid** if:
- the parking phase is sampled by the IMU,
- IMU data during the parking phase are continuous enough to define a usable integration window.

For valid cycles:
- trajectory integration is performed **only over the parking phase**,
- descent and ascent are treated using the assumptions defined above.

---

### Invalid cycles

A cycle is **invalid** if:
- the parking phase is not sampled by the IMU.

For invalid cycles:
- no trajectory reconstruction is attempted,
- the cycle is excluded from the BVP analysis.

No partial or heuristic reconstruction is performed for these cycles.

---

## Implications for long cycles

Since Argo cycles may last several days, this strategy ensures that:

- no integration is attempted over long intervals without dynamical measurements,
- accumulated drift over unsampled phases is avoided,
- all reconstructed displacements are supported by IMU observations.

The preprocessing stage explicitly encodes where assumptions are made and where
dynamical information is available.

---

## Responsibilities of the preprocessing stage

The preprocessing pipeline must, for each cycle:

1. Identify the parking phase (`t_park_start`, `t_park_end`).
2. Determine whether the parking phase is sampled by the IMU.
3. Flag the cycle as:
   - `valid_for_bvp = True` if parking is sampled,
   - `valid_for_bvp = False` otherwise.
4. Provide:
   - surface position constraints before and after parking,
   - temporal diagnostics linking physical events and observations.
5. **Not** perform any trajectory integration. 

The preprocessing output must contain all information required by the BVP solver
to operate **only on valid cycles**.

### Encoded outputs

The preprocessing products record the parking sampling and validity decisions
explicitly so the solver can filter cycles without re-deriving them:

- `ds_cycles.park_sampled` — `True` if IMU samples the parking phase,
  otherwise `False`.
- `ds_cycles.valid_for_bvp` — mirrors `park_sampled`; marks cycles eligible
  for BVP analysis.
- `ds_cycles.t_park_start`, `ds_cycles.t_park_end` — parking window bounds
  derived from parking measurement codes.
- `ds_segments.is_parking_phase` — flags which contiguous segments correspond
  to the parking phase.

---

## Role of the BVP solver

The BVP solver operates exclusively on cycles flagged as valid by the preprocessing.

It:
- integrates the equations of motion over the parking phase,
- applies surface position constraints as boundary conditions,
- reconstructs positions at the start and/or end of the parking phase,
- estimates position errors at key physical events.

No cycle validity checks are performed within the BVP solver itself.

---

## Summary

- Only cycles with IMU-sampled parking phases are considered.
- Descent and ascent are handled via explicit, declared assumptions.
- Long cycles do not pose a problem if dynamical data are available during parking.
- Preprocessing defines *what can be reconstructed*.
- The BVP solver defines *how it is reconstructed*.

This separation ensures clarity, robustness, and scientific transparency.