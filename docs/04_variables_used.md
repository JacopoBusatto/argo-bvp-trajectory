# Variables Used (I/O Reference)

This document lists the variables required/produced by the current pipeline
and how they map to each file type.

## TRAJ (input to preprocess)

Minimal variables required by `preprocess`:

| Variable | Dimensions | Units | Purpose |
| --- | --- | --- | --- |
| `JULD` | `N_MEASUREMENT` | days | Time axis for surface windows and anchors |
| `PRES` | `N_MEASUREMENT` | dbar | Surface detection and phase proxy |
| `LATITUDE` | `N_MEASUREMENT` | degree_north | GPS positions at surface |
| `LONGITUDE` | `N_MEASUREMENT` | degree_east | GPS positions at surface |
| `POSITION_QC` | `N_MEASUREMENT` | Argo QC | Valid/invalid GPS fixes |

Notes:
- `LATITUDE`/`LONGITUDE` are expected only at surface (NaN underwater).
- Phase is not required; in synthetic TRAJ it is encoded via `MEASUREMENT_CODE`,
  but preprocessing does not depend on it.

## AUX (input to preprocess)

Minimal variables required by `preprocess`:

| Variable | Dimensions | Units | Purpose |
| --- | --- | --- | --- |
| `JULD` | `N_MEASUREMENT` | days | Time axis for IMU samples |
| `PRES` | `N_MEASUREMENT` | dbar | Phase proxy and depth context |
| `LINEAR_ACCELERATION_COUNT_X` | `N_MEASUREMENT` | count | IMU linear acceleration (counts) |
| `LINEAR_ACCELERATION_COUNT_Y` | `N_MEASUREMENT` | count | IMU linear acceleration (counts) |
| `LINEAR_ACCELERATION_COUNT_Z` | `N_MEASUREMENT` | count | IMU linear acceleration (counts) |
| `ANGULAR_RATE_COUNT_X` | `N_MEASUREMENT` | count | Gyro counts (currently unused) |
| `ANGULAR_RATE_COUNT_Y` | `N_MEASUREMENT` | count | Gyro counts (currently unused) |
| `ANGULAR_RATE_COUNT_Z` | `N_MEASUREMENT` | count | Gyro counts (currently unused) |
| `MAGNETIC_FIELD_COUNT_X` | `N_MEASUREMENT` | count | Magnetometer counts (currently unused) |
| `MAGNETIC_FIELD_COUNT_Y` | `N_MEASUREMENT` | count | Magnetometer counts (currently unused) |
| `MAGNETIC_FIELD_COUNT_Z` | `N_MEASUREMENT` | count | Magnetometer counts (currently unused) |

Notes:
- Gyro/magnetometer are present but currently zero in the synthetic data.
- Conversion from counts to SI happens in preprocessing using the selected
  `InstrumentParams`.

## CYCLE_*.nc (preprocess output)

Key variables (from `cycle_schema`):

| Variable | Dimensions | Units | Meaning |
| --- | --- | --- | --- |
| `anchor_juld` | `anchor=2` | days | Dive/emerge GPS times |
| `anchor_lat` | `anchor=2` | degree_north | Dive/emerge GPS lat |
| `anchor_lon` | `anchor=2` | degree_east | Dive/emerge GPS lon |
| `anchor_position_qc` | `anchor=2` | Argo QC | QC codes at anchors |
| `lat0` | scalar | degree_north | ENU origin (dive anchor) |
| `lon0` | scalar | degree_east | ENU origin (dive anchor) |
| `juld` | `obs` | days | Time axis for cycle samples |
| `t` (coord) | `obs` | s | Seconds since dive anchor |
| `pres` | `obs` | dbar | Pressure time series |
| `phase` | `obs` | code | 0=unknown,1=surface,2=descent,3=park,4=ascent |
| `lin_acc_count` | `obs,vec` | count | Linear acceleration counts |
| `ang_rate_count` | `obs,vec` | count | Gyro counts |
| `mag_field_count` | `obs,vec` | count | Magnetometer counts |
| `lin_acc` | `obs,vec` | m/s^2 | Linear acceleration in SI |
| `ang_rate` | `obs,vec` | rad/s | Angular rate in SI |
| `mag_field` | `obs,vec` | uT | Magnetic field in SI |

## REC_*.nc (integration output)

Added by integration:

| Variable | Dimensions | Units | Meaning |
| --- | --- | --- | --- |
| `x_enu` | `obs` | m | Reconstructed ENU east |
| `y_enu` | `obs` | m | Reconstructed ENU north |
| `lat_rec` | `obs` | degree_north | Reconstructed latitude |
| `lon_rec` | `obs` | degree_east | Reconstructed longitude |
| `underwater_mask` | `obs` | bool | True for non-surface samples |

## TRUTH_*.nc (synthetic ground truth)

Key variables:

| Variable | Dimensions | Units | Meaning |
| --- | --- | --- | --- |
| `t` | `obs` | s | Time since start |
| `x`, `y`, `z` | `obs` | m | ENU positions (synthetic) |
| `ax`, `ay`, `az` | `obs` | m/s^2 | ENU accelerations (synthetic) |
| `pres` | `obs` | dbar | Pressure proxy |
| `phase` | `obs` | string | surface/descent/park/ascent |
| `lat`, `lon` | `obs` | deg | Lat/lon derived from ENU |
| `anchor_*` | `anchor=2` | various | Start/end anchor values |

TRUTH is used for validation and sweep analysis; it is not part of the
real-data pipeline.

## File → step → purpose

| File | Produced by | Purpose |
| --- | --- | --- |
| `SYNTH_<tag>_TRUTH.nc` | `synth` | Ground truth for validation |
| `SYNTH_<tag>_TRAJ.nc` | `synth` | Surface GPS + pressure profile |
| `SYNTH_<tag>_AUX.nc` | `synth` | IMU counts (acc/gyro/mag) |
| `CYCLE_<tag>_W###.nc` | `preprocess` | Anchors + IMU in SI, ready for integration |
| `REC_<tag>_W###.nc` | `integrate` | Reconstructed ENU + lat/lon |

NOTE (IT): i file reali saranno TRAJ/AUX reali; TRUTH non esiste su dati reali.
