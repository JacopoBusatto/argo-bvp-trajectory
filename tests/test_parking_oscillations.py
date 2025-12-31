"""Tests for parking oscillations in synthetic truth."""

from dataclasses import replace

import numpy as np

from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_truth import generate_truth_cycle


def test_parking_z_oscillation_present() -> None:
    params = replace(
        DEFAULT_EXPERIMENT,
        park_z_osc_amplitude_m=5.0,
        park_z_osc_period_s=300.0,
    )
    ds = generate_truth_cycle(params)
    z_park = _parking_values(ds, "z", exclude_transition=True)
    z_detrended = z_park - np.nanmedian(z_park)
    ptp = np.nanmax(z_detrended) - np.nanmin(z_detrended)
    expected = 2.0 * params.park_z_osc_amplitude_m
    assert 0.8 * expected <= ptp <= 1.2 * expected


def test_parking_r_oscillation_present() -> None:
    params = replace(
        DEFAULT_EXPERIMENT,
        park_r_osc_amplitude_m=10.0,
        park_r_osc_period_s=1800.0,
    )
    ds = generate_truth_cycle(params)
    x_park = _parking_values(ds, "x", exclude_transition=True)
    y_park = _parking_values(ds, "y", exclude_transition=True)
    cx = float(ds.attrs["park_center_x"])
    cy = float(ds.attrs["park_center_y"])
    r = np.sqrt((x_park - cx) ** 2 + (y_park - cy) ** 2)
    r_detrended = r - np.nanmedian(r)
    ptp = np.nanmax(r_detrended) - np.nanmin(r_detrended)
    expected = 2.0 * params.park_r_osc_amplitude_m
    assert 0.8 * expected <= ptp <= 1.2 * expected


def test_parking_no_oscillations_when_zero() -> None:
    params = replace(
        DEFAULT_EXPERIMENT,
        park_z_osc_amplitude_m=0.0,
        park_r_osc_amplitude_m=0.0,
    )
    ds = generate_truth_cycle(params)
    z_park = _parking_values(ds, "z", exclude_transition=True)
    z_ptp = np.nanmax(z_park) - np.nanmin(z_park)
    assert z_ptp <= 1e-3

    x_park = _parking_values(ds, "x", exclude_transition=True)
    y_park = _parking_values(ds, "y", exclude_transition=True)
    cx = float(ds.attrs["park_center_x"])
    cy = float(ds.attrs["park_center_y"])
    r = np.sqrt((x_park - cx) ** 2 + (y_park - cy) ** 2)
    r_ptp = np.nanmax(r) - np.nanmin(r)
    assert r_ptp <= 1e-3


def _parking_values(ds, name: str, exclude_transition: bool = False) -> np.ndarray:
    phase = np.asarray(ds["phase"].values)
    mask = phase == "park"
    if exclude_transition and "is_transition" in ds:
        mask = mask & (np.asarray(ds["is_transition"].values) == 0)
    values = np.asarray(ds[name].values, dtype=float)
    return values[mask]
