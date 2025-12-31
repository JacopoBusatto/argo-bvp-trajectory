"""Tests for transition smoothing behavior."""

from dataclasses import replace

import numpy as np

from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_transition_smoothing_reduces_spikes() -> None:
    params_raw = replace(DEFAULT_EXPERIMENT, transition_seconds=0.0)
    params_smooth = replace(DEFAULT_EXPERIMENT, transition_seconds=60.0)

    truth_raw = generate_truth_cycle(params_raw)
    truth_smooth = generate_truth_cycle(params_smooth)

    acc_raw = _acc_magnitude(truth_raw)
    acc_smooth = _acc_magnitude(truth_smooth)

    assert acc_smooth <= 0.2 * acc_raw

    for coord in ("x", "y", "z"):
        raw_vals = truth_raw[coord].values
        smooth_vals = truth_smooth[coord].values
        assert np.isclose(raw_vals[0], smooth_vals[0], atol=1e-9)
        assert np.isclose(raw_vals[-1], smooth_vals[-1], atol=1e-6)


def _acc_magnitude(ds) -> float:
    ax = ds["ax"].values
    ay = ds["ay"].values
    az = ds["az"].values
    acc = np.sqrt(ax * ax + ay * ay + az * az)
    return float(np.nanmax(np.abs(acc)))
