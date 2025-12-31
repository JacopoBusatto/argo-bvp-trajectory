from pathlib import Path
import xarray as xr
from argo_bvp.preprocess.cycle_builder import build_cycle_from_traj_aux

traj = xr.load_dataset(Path("C:/Users/Jacopo/Documents/argo-bvp-trajectory/outputs/synthetic/SYNTH_CY24h_d5s_p10s_a5s_n0_TRAJ.nc"))
aux  = xr.load_dataset(Path("C:/Users/Jacopo/Documents/argo-bvp-trajectory/outputs/synthetic/SYNTH_CY24h_d5s_p10s_a5s_n0_AUX.nc"))

cycle = build_cycle_from_traj_aux(traj, aux, window_index=0, instrument_params=...)
