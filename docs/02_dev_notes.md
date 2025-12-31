## Dev Notes

### Implemented (Milestone 1)

- Created the `argo_bvp` package layout with schema helpers and NetCDF I/O.
- Implemented schema v1 helpers:
  - `make_empty_cycle_dataset` to build a compliant template dataset.
  - `validate_cycle_dataset` to enforce dimensions, required variables, and metadata.
- Added NetCDF read/write helpers with validation on both paths.
- Implemented TRAJ/AUX readers, surface window detection, and a minimal cycle builder.
- Added pytest coverage for schema integrity, validation failures, and I/O round-trip.
- Added preprocessing tests for anchor selection and minimal cycle building.
- Wrote initial documentation and README.

### Decisions

- Use xarray + h5netcdf for NetCDF I/O to avoid the netCDF4 backend.
- Keep validation strict for schema metadata and units, but avoid enforcing data values.
- Include optional scalar variables (`lat0`, `lon0`, `g`) as ready-to-fill placeholders.
- Normalize unit attributes in validation (decode bytes, strip whitespace); JULD units can round-trip via encoding, so validation falls back to encoding units and accepts CF-equivalent representations.

### TODO (next)

- Define a synthetic data generator for controlled tests.
- Add BVP solver scaffolding and diagnostics output formats.
