# Integration

Integration reconstructs horizontal ENU positions from a cycle file using the
Fubini boundary value formulation. The vertical coordinate is not integrated;
depth is derived from pressure.

## Scientific / logical view

- Integrate only underwater samples (phase != surface).
- Enforce anchor constraints: start anchor at ENU origin, end anchor at the
  ENU position derived from `anchor_lat/lon`.
- Use linear accelerations in ENU (east, north) as the input.

## Technical view

Key modules and functions:

- `argo_bvp.integrate.fubini`
  - `integrate_fubini_1d(t_s, a, x0, xT, method)`
  - Handles non-uniform sampling and enforces endpoints.
- `argo_bvp.integrate.park_xy`
  - `reconstruct_xy_enu_fubini(...)`
  - `reconstruct_cycle_xy_from_ds(ds_cycle, method)`
  - Adds `x_enu`, `y_enu`, `lat_rec`, `lon_rec`, `underwater_mask` to the dataset.
- `argo_bvp.run_integrate`
  - `integrate_cycle_file(cycle_path, outdir, method)`

## Assumptions and limits

- `lin_acc` is already expressed in ENU; no body-frame rotation is applied.
- Only 2D integration (x/y); z is inferred from pressure.
- Surface samples are excluded from integration and kept as NaN for plotting.

## Outputs

`integrate` writes:
- `REC_<tag>_W###.nc`
- `<tag>_W###_{plan,3d,acc}.png`
