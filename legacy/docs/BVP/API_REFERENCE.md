Boundary Value / Kinematics API Reference
=========================================

Scope: lightweight helpers in `src/argobvp/` for integration and simple BVP shooting. These are building blocks; end-to-end solvers are not guaranteed unless implemented by the user.

Modules and functions
---------------------

### integrators.py
Purpose: integrate second-order kinematics `dr/dt = v`, `dv/dt = a(t,r,v)` on a prescribed time grid.

- Enum `IntegratorMethod(str, Enum)`  
  Values: `EULER`, `TRAPEZOID`, `RK4`.

- `integrate_2nd_order(t: Array, r0: Array, v0: Array, a_fun: Callable[[float, Array, Array], Array], method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID, backward: bool = False) -> Tuple[Array, Array]`  
  Inputs:  
  * `t` (N, float): strictly increasing time grid (s or consistent units).  
  * `r0` (D,), `v0` (D,): initial position/velocity (interpreted at `t[0]`; if `backward=True`, interpreted at `t[-1]`).  
  * `a_fun(ti, ri, vi)` → (D,): acceleration model; must return same dimension.  
  * `method`: `"euler"`, `"trapezoid"`, or `"rk4"`.  
  * `backward`: if True, integrates from `t[-1]` to `t[0]` but returns arrays aligned to increasing `t`.  
  Outputs: `r`, `v` (N,D) arrays aligned to increasing `t`. Units: position/velocity follow inputs; acceleration per `a_fun`.  
  Assumptions: `t` strictly increasing; shapes consistent; no automatic frame handling; stability per chosen scheme; no event handling or constraints.

### metrics.py
Purpose: simple error utilities.

- `nearest_index(t: Array, t_query: float) -> int`  
  Returns index of closest time to `t_query`. Inputs: `t` numeric array, `t_query` scalar.
- `endpoint_error(r: Array, r_target: Array, ord: int = 2) -> float`  
  Computes ‖r[-1] - r_target‖_ord. Inputs: `r` (N,D), `r_target` (D,). Output: scalar.
- `point_error_at_time(t: Array, r: Array, t_query: float, r_target: Array, ord: int = 2) -> float`  
  Picks nearest time to `t_query` and returns norm to `r_target`. Inputs: `t` (N,), `r` (N,D), `t_query` scalar, `r_target` (D,). Output: scalar.

Assumptions: Euclidean norms (ord default 2); no frame knowledge.

### bvp.py
Purpose: basic shooting on initial velocity to meet a terminal position constraint.

- Dataclass `ShootingResult(v0_opt: Array, success: bool, message: str, nfev: int, r: Array, v: Array)`  
  Holds solution (or best effort) initial velocity and resulting trajectory.

- `shoot_v0_to_hit_rT(t: Array, r0: Array, rT_target: Array, v0_guess: Array, a_fun: Callable[[float, Array, Array], Array], method: IntegratorMethod | str = IntegratorMethod.TRAPEZOID, dims: Tuple[int, ...] = (0, 1)) -> ShootingResult`  
  Inputs:  
  * `t` (N, float): time grid (strictly increasing).  
  * `r0` (D,), `rT_target` (D,): initial and desired terminal position.  
  * `v0_guess` (D,): initial guess for velocity; only `dims` are optimized.  
  * `a_fun(t, r, v)` → (D,): acceleration model.  
  * `method`: integration scheme passed to `integrate_2nd_order`.  
  * `dims`: tuple of indices constrained/optimized (e.g., (0,1) for XY if Z is prescribed).  
  Outputs: `ShootingResult` with optimized `v0_opt`, success flag/message/nfev, and full `r`, `v` arrays.  
  Assumptions: no bounds on velocity; uses SciPy `root` (hybr); depends on a continuous, smooth `a_fun`; no robustness to poor conditioning.

How preprocess connects to BVP-ready
------------------------------------
- Required inputs (from `*_bvp_ready.nc`): time vector (`time`), horizontal accelerations (`acc_n`, `acc_e`), phase labels (`phase_name`), pressure (`z_from_pres` as depth proxy), per-cycle slices (`row_start`, `row_size`), key times (`t_*`), surface constraints (`lat_surface_start`, `lon_surface_start`, `lat_surface_end`, `lon_surface_end`, `t_surface_start_fix`, `t_surface_end_fix`, `pos_age_s`, `pos_source`), and phase attendibility flags/cnts (`<phase>_attendible`, `<phase>_n_obs`), thresholds (`min_parking_samples_for_bvp`, `min_phase_samples_for_bvp`).
- Inclusion rule: parking attendible is mandatory; other phases (ascent, descent_to_profile, profile_drift, surface, in_air, grounded, other) are included only if attendible. Missing/non-attendible phases are expected to be treated as vertical (zero horizontal displacement) by the solver per `SAMPLING_VALIDITY`.
- Typical usage: build per-cycle time/acc slices from BVP-ready, choose acceleration model (often directly `acc_n/e` in NED), optionally enforce boundary conditions from surface fixes, and apply `integrate_2nd_order` or `shoot_v0_to_hit_rT` on horizontal components. Frame remains NED; depth from pressure if needed.

Notes on maturity
-----------------
- These helpers are intentionally minimal: no state estimation, no noise models, no constraint handling beyond shooting on `v0`.  
- Users must design their own acceleration models, boundary handling, and validation.  
- Stability and accuracy depend on the chosen integration method and time step; no adaptive stepping is provided.
