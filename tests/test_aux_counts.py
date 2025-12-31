"""Tests for synthetic AUX count conversion."""

import numpy as np

from argo_bvp.instruments import INSTRUMENTS
from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_aux import build_aux_from_truth
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_aux_counts() -> None:
    truth = generate_truth_cycle(DEFAULT_EXPERIMENT)
    aux = build_aux_from_truth(truth, INSTRUMENTS["synth_v1"])

    assert aux.sizes["N_MEASUREMENT"] == truth.sizes["obs"]

    acc_x = aux["LINEAR_ACCELERATION_COUNT_X"].values
    acc_y = aux["LINEAR_ACCELERATION_COUNT_Y"].values
    acc_z = aux["LINEAR_ACCELERATION_COUNT_Z"].values

    assert np.issubdtype(acc_x.dtype, np.integer)
    assert np.any(acc_x != 0) or np.any(acc_y != 0) or np.any(acc_z != 0)

    for name in (
        "ANGULAR_RATE_COUNT_X",
        "ANGULAR_RATE_COUNT_Y",
        "ANGULAR_RATE_COUNT_Z",
        "MAGNETIC_FIELD_COUNT_X",
        "MAGNETIC_FIELD_COUNT_Y",
        "MAGNETIC_FIELD_COUNT_Z",
    ):
        arr = aux[name].values
        assert np.all(arr == 0)
