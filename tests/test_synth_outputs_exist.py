"""Smoke test for synthetic raw outputs."""

from argo_bvp.instruments import INSTRUMENTS
from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_synthetic_raw import generate_synthetic_raw


def test_synth_outputs_exist(tmp_path) -> None:
    outputs = generate_synthetic_raw(tmp_path, DEFAULT_EXPERIMENT, INSTRUMENTS["synth_v1"])
    for key in ("truth_nc", "traj_nc", "aux_nc", "plan_png", "xyz_png", "acc_png", "depth_png"):
        assert outputs[key].exists()
