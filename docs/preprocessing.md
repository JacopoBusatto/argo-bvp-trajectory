## Preprocessing: TRAJ + AUX -> Cycle File

### Overview

The preprocessing step builds a single cycle NetCDF file from:
- TRAJ (GPS fixes and pressure)
- AUX (IMU time series and pressure)

The output must conform to the `argo_bvp_cycle` schema v1.

### Surface window detection

Surface windows are defined on the TRAJ time series using:

- `pres <= p_surface` (default `5 dbar`)
- contiguous indices in time
- optional `max_gap_seconds` to split windows when time gaps are large

The anchor selection uses two consecutive surface windows:
- start anchor: last valid GPS fix in window `i`
- end anchor: first valid GPS fix in window `i+1`

### GPS QC filtering

GPS fixes are filtered using `qc_ok` (default `{"1","2","5"}`).
Only fixes with finite `juld`, `lat`, and `lon` and QC in `qc_ok` are eligible.

### AUX subsetting

The AUX time series is subset between the anchor start and end timestamps
(inclusive). The selected samples populate:
- `juld`, `pres`
- `lin_acc_count`, `ang_rate_count`, `mag_field_count`

`z`, `lin_acc`, `ang_rate`, and `mag_field` remain as placeholders for now.

### Phase classification (simple)

Phases are derived from pressure:
- `surface` if `pres <= p_surface`
- `descent` before the max-pressure sample
- `ascent` after the max-pressure sample
- `park` near the max-pressure sample (within `park_eps_dbar`, default `5 dbar`)

### Configuration keys

- `p_surface`: surface pressure threshold (dbar)
- `max_gap_seconds`: split surface windows if gaps exceed this value
- `qc_ok`: set of accepted QC codes
- `traj_vars`: mapping for TRAJ variable names
- `aux_vars`: mapping for AUX variable names
- `float_id`: override float identifier
- `cycle_number`: override cycle number
- `park_eps_dbar`: pressure tolerance for park phase tagging
