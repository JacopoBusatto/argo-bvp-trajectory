## Cycle File NetCDF Schema (v1)

Schema name: `argo_bvp_cycle`  
Schema version: `1.0`

### Dimensions

- `obs`: number of time samples in the IMU series
- `anchor`: 2 (start, end)
- `vec`: 3 (x, y, z)

### Coordinates

- `t (obs)`: seconds since cycle start (anchor start JULD)
- `anchor (anchor)`: `["start", "end"]`
- `vec (vec)`: `["x", "y", "z"]`

### Global attributes (minimum)

- `schema_name = "argo_bvp_cycle"`
- `schema_version = "1.0"`
- `cycle_number` (int)
- `float_id` (str or int)
- `time_origin_juld` (float, anchor start JULD)
- `time_units = "s since cycle start (anchor start JULD)"`
- `geodesy = "WGS84"`
- `lon_convention = "[-180,180)"`

### Required variables

- `anchor_juld (anchor)`: units `days since 1950-01-01 00:00:00 UTC`
- `anchor_lat (anchor)`: units `degree_north`
- `anchor_lon (anchor)`: units `degree_east`
- `anchor_position_qc (anchor)`: int8 or string
- `juld (obs)`: units `days since 1950-01-01 00:00:00 UTC`
- `pres (obs)`: units `dbar`
- `z (obs)`: units `m`, `positive="down"`
- `lin_acc_count (obs, vec)`: units `count`
- `lin_acc (obs, vec)`: units `m s-2`
- `ang_rate_count (obs, vec)`: units `count` (placeholder)
- `ang_rate (obs, vec)`: units `rad s-1` (placeholder)
- `mag_field_count (obs, vec)`: units `count` (placeholder)
- `mag_field (obs, vec)`: units `uT` (placeholder)
- `phase (obs)`: int8, attr `phase_meaning`

Phase meanings:
`0=unknown,1=surface,2=descent,3=park,4=ascent`

### Optional scalar variables (recommended)

- `lat0`: anchor start latitude, units `degree_north`
- `lon0`: anchor start longitude, units `degree_east`
- `g`: local gravity, units `m s-2`
