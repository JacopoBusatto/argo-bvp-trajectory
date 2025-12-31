## Variables Used for Synthetic TRAJ/AUX (Checkpoint 0)

### TRAJ variables (from raw_variables.txt)

| Variable | Dims | Dtype | Units | Description | Why needed |
| --- | --- | --- | --- | --- | --- |
| `JULD` | `(N_MEASUREMENT)` | `datetime64[ns]` | relative julian days (parts of day) | Julian day of each measurement relative to REFERENCE_DATE_TIME | Time axis for synthetic trajectory and sync with AUX |
| `LATITUDE` | `(N_MEASUREMENT)` | `float64` | degree_north | Latitude of each location | Surface GPS fixes (start/end) |
| `LONGITUDE` | `(N_MEASUREMENT)` | `float64` | degree_east | Longitude of each location | Surface GPS fixes (start/end) |
| `POSITION_QC` | `(N_MEASUREMENT)` | `object` | Argo reference table 2 | Quality on position | Provide minimal QC flagging for GPS fixes |
| `CYCLE_NUMBER` | `(N_MEASUREMENT)` | `float64` | cycle index (0..N) | Float cycle number of the measurement | Partition and label synthetic cycles |
| `MEASUREMENT_CODE` | `(N_MEASUREMENT)` | `float64` | Argo reference table 15 | Flag referring to a measurement event in the cycle | Encode phases (surface/descent/park/ascent) |
| `PRES` | `(N_MEASUREMENT)` | `float32` | decibar | Sea water pressure | Depth proxy for the profile |
| `TEMP` | `(N_MEASUREMENT)` | `float32` | degree_Celsius | Sea temperature in-situ ITS-90 scale | CTD-like signal along the profile |
| `PSAL` | `(N_MEASUREMENT)` | `float32` | psu | Practical salinity | CTD-like signal along the profile |

### AUX variables (from dsaux_variables.txt)

| Variable | Dims | Dtype | Units | Description | Why needed |
| --- | --- | --- | --- | --- | --- |
| `JULD` | `(N_MEASUREMENT)` | `datetime64[ns]` | relative julian days (parts of day) | Julian day of each measurement relative to REFERENCE_DATE_TIME | Time axis for IMU-like samples |
| `CYCLE_NUMBER` | `(N_MEASUREMENT)` | `float64` | cycle index (0..N) | Float cycle number of the measurement | Align AUX with TRAJ cycles |
| `MEASUREMENT_CODE` | `(N_MEASUREMENT)` | `float64` | Argo reference table 15 | Flag referring to a measurement event in the cycle | Share phase labeling with TRAJ |
| `PRES` | `(N_MEASUREMENT)` | `float32` | decibar | Sea water pressure | Depth proxy for IMU samples |
| `TEMP_COUNT_INERTIAL` | `(N_MEASUREMENT)` | `float32` | count | Temperature of the INERTIAL sensor | Simple temperature channel in AUX |
| `LINEAR_ACCELERATION_COUNT_X` | `(N_MEASUREMENT)` | `float32` | count | Linear acceleration along the X axis | Main IMU signal (counts) |
| `LINEAR_ACCELERATION_COUNT_Y` | `(N_MEASUREMENT)` | `float32` | count | Linear acceleration along the Y axis | Main IMU signal (counts) |
| `LINEAR_ACCELERATION_COUNT_Z` | `(N_MEASUREMENT)` | `float32` | count | Linear acceleration along the Z axis | Main IMU signal (counts) |
| `ANGULAR_RATE_COUNT_X` | `(N_MEASUREMENT)` | `float32` | count | Angular rate along the X axis | Gyro channel (counts) |
| `ANGULAR_RATE_COUNT_Y` | `(N_MEASUREMENT)` | `float32` | count | Angular rate along the Y axis | Gyro channel (counts) |
| `ANGULAR_RATE_COUNT_Z` | `(N_MEASUREMENT)` | `float32` | count | Angular rate along the Z axis | Gyro channel (counts) |
| `MAGNETIC_FIELD_COUNT_X` | `(N_MEASUREMENT)` | `float32` | count | Magnetic field along the X axis | Magnetometer channel (counts) |
| `MAGNETIC_FIELD_COUNT_Y` | `(N_MEASUREMENT)` | `float32` | count | Magnetic field along the Y axis | Magnetometer channel (counts) |
| `MAGNETIC_FIELD_COUNT_Z` | `(N_MEASUREMENT)` | `float32` | count | Magnetic field along the Z axis | Magnetometer channel (counts) |

### Notes

- "gyro e magnetometro presenti ma a zero" per mantenere lo schema completo senza modellare la dinamica rotazionale.
- GPS: lat/lon solo in superficie, NaN sott'acqua.
- Variabile `PHASE` non presente nel template TRAJ: le fasi sono codificate in `MEASUREMENT_CODE` con mappa `surface=1`, `descent=2`, `park=3`, `ascent=4`.

### Missing variables and alternatives

- Depth in meters is not present in TRAJ/AUX templates; use `PRES` as the depth proxy and derive meters with a standard conversion.
- Subsurface horizontal ground-truth positions are not present in TRAJ/AUX templates; keep them in the separate ground-truth output while TRAJ keeps `LATITUDE`/`LONGITUDE` only for surface fixes.
