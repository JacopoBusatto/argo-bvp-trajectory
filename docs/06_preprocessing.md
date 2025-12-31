# Preprocessing

Preprocessing builds a single cycle dataset from TRAJ and AUX inputs. The output
is a validated NetCDF file conforming to the cycle schema and ready for integration.

## Scientific / logical view

- Identify two consecutive surface windows in TRAJ using pressure.
- Select GPS anchors with QC filtering:
  - **dive**: last valid fix in window `i`
  - **emerge**: first valid fix in window `i+1`
- Subset AUX between the two anchor times.
- Convert IMU counts to SI using the selected instrument parameters.
- Tag phases using a simple pressure-based heuristic.

## Technical view

Key modules and functions:

- `argo_bvp.preprocess.surface_windows`
  - `find_surface_windows(juld, pres, p_surface, max_gap_seconds)`
  - `select_anchor_points(juld, pres, lat, lon, position_qc, window_index, qc_ok)`
- `argo_bvp.preprocess.cycle_builder`
  - `build_cycle_from_traj_aux(traj, aux, window_index, instrument, config)`
  - Assigns anchors, converts IMU counts, fills `phase`, creates time coordinate `t`.
- `argo_bvp.io.cycle_io`
  - `write_cycle_netcdf`, `read_cycle_netcdf`

## Instrument parameters

Conversion from counts to SI happens in preprocessing. Use the instruments
registry in `src/argo_bvp/instruments/registry.py`:

- `lsb_to_ms2`: linear acceleration counts → m/s^2
- `gyro_lsb_to_rads`: gyro counts → rad/s
- `mag_lsb_to_uT`: magnetometer counts → μT

To add a new instrument, create a new entry in `INSTRUMENTS` and reference it
via `--instrument` in the CLI.

## Configuration knobs

The preprocessing step accepts a config mapping. Important keys:

- `p_surface`: surface pressure threshold (dbar)
- `max_gap_seconds`: split surface windows when gaps are large
- `qc_ok`: QC codes accepted for GPS anchors
- `traj_vars`: mapping for TRAJ variable names
- `aux_vars`: mapping for AUX variable names
- `park_eps_dbar`: tolerance for park phase detection

## Outputs

`preprocess` writes:
- `CYCLE_<tag>_W###.nc`

The cycle file includes anchors, pressure, phase tags, IMU counts, and IMU data
in SI units.
