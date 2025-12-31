"""Integration test: build cycle file from synthetic TRAJ/AUX."""

from pathlib import Path

import numpy as np
import xarray as xr

from argo_bvp.cycle_schema import validate_cycle_dataset
from argo_bvp.io.cycle_io import read_cycle_netcdf
from argo_bvp.run_preprocess import build_cycle_file
from argo_bvp.instruments import INSTRUMENTS
from argo_bvp.synth.generate_aux import build_aux_from_truth
from argo_bvp.synth.generate_traj import build_traj_from_truth
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_preprocess_build_cycle_from_synth(tmp_path) -> None:
    truth = generate_truth_cycle()
    traj = build_traj_from_truth(truth)
    aux = build_aux_from_truth(truth, INSTRUMENTS["synth_v1"])

    traj_path = tmp_path / "synth_TRAJ.nc"
    aux_path = tmp_path / "synth_AUX.nc"
    out_path = tmp_path / "cycle.nc"

    traj.to_netcdf(traj_path, engine="h5netcdf")
    aux.to_netcdf(aux_path, engine="h5netcdf")

    build_cycle_file(
        traj_path=traj_path,
        aux_path=aux_path,
        out_path=out_path,
        window_index=0,
        instrument="synth_v1",
    )

    ds = read_cycle_netcdf(out_path)
    validate_cycle_dataset(ds, strict=True)

    phase = np.asarray(ds["phase"].values)
    mask = phase != 1
    if np.any(mask):
        lin_acc = ds["lin_acc"].values[mask]
        assert np.any(lin_acc != 0.0)
