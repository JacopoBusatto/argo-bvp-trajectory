## Usage Examples

### Create an empty cycle dataset

```python
from argo_bvp.cycle_schema import make_empty_cycle_dataset, validate_cycle_dataset

ds = make_empty_cycle_dataset(n_obs=1000, float_id="6900001", cycle_number=12)
validate_cycle_dataset(ds)
```

### Fill anchors and time series (outline)

```python
import numpy as np

ds["anchor_juld"].values[:] = [25000.0, 25010.0]
ds["anchor_lat"].values[:] = [43.0, 43.5]
ds["anchor_lon"].values[:] = [9.0, 9.6]
ds["anchor_position_qc"].values[:] = [1, 1]
ds.attrs["time_origin_juld"] = ds["anchor_juld"].values[0]

t = np.arange(ds.sizes["obs"], dtype="float64")
ds = ds.assign_coords(t=("obs", t))
ds["juld"].values[:] = ds.attrs["time_origin_juld"] + t / 86400.0
```

### Write and read NetCDF

```python
from argo_bvp.io.cycle_io import write_cycle_netcdf, read_cycle_netcdf

write_cycle_netcdf(ds, "cycle_001.nc")
ds_read = read_cycle_netcdf("cycle_001.nc")
```
