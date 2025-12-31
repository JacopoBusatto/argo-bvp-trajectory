"""Basic checks for synthetic TRUTH generation."""

from dataclasses import replace

import numpy as np

from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_truth_basic() -> None:
    params = replace(DEFAULT_EXPERIMENT, lat0=43.0, lon0=10.0)
    ds = generate_truth_cycle(params)

    t = ds["t"].values
    assert t.size > 2
    assert np.all(np.diff(t) > 0)

    phase = ds["phase"].values
    is_transition = ds["is_transition"].values.astype(bool)
    park_mask = (phase == "park") & ~is_transition
    z = ds["z"].values[park_mask]
    expected_min = -params.park_depth_m - params.park_z_osc_amplitude_m
    assert np.isclose(np.nanmin(z), expected_min, atol=1.0)

    phases = set(ds["phase"].values.tolist())
    assert {"surface", "descent", "park", "ascent"}.issubset(phases)

    anchor_idx = ds["anchor_idx"].values.astype(int)
    for idx in anchor_idx:
        assert ds["phase"].values[int(idx)] == "surface"
