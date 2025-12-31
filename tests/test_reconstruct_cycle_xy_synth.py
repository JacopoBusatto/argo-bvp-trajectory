"""Tests for ENU reconstruction from a synthetic cycle dataset."""

import numpy as np

from argo_bvp.integrate.park_xy import latlon_to_enu_m, reconstruct_cycle_xy_from_ds
from argo_bvp.preprocess.cycle_builder import build_cycle_from_traj_aux
from argo_bvp.synth.generate_aux import build_aux_from_truth
from argo_bvp.synth.generate_traj import build_traj_from_truth
from argo_bvp.synth.generate_truth import generate_truth_cycle


def _build_cycle_from_synth(instrument: object | str = "synth_v1"):
    truth = generate_truth_cycle()
    traj = build_traj_from_truth(truth)
    aux = build_aux_from_truth(truth, instrument)
    ds_cycle = build_cycle_from_traj_aux(traj, aux, window_index=0, instrument=instrument)
    return truth, ds_cycle


def test_reconstruct_cycle_hits_anchors() -> None:
    truth, ds_cycle = _build_cycle_from_synth()
    ds_rec = reconstruct_cycle_xy_from_ds(ds_cycle)

    mask = ds_rec["underwater_mask"].values.astype(bool)
    idx = np.where(mask)[0]
    assert idx.size > 2

    x0 = ds_rec["x_enu"].values[idx[0]]
    y0 = ds_rec["y_enu"].values[idx[0]]
    assert abs(x0) < 1e-6
    assert abs(y0) < 1e-6

    lat_end = ds_rec["lat_rec"].values[idx[-1]]
    lon_end = ds_rec["lon_rec"].values[idx[-1]]
    anchor_lat = float(ds_cycle["anchor_lat"].values[1])
    anchor_lon = float(ds_cycle["anchor_lon"].values[1])
    lat0 = float(ds_cycle["lat0"].values)
    lon0 = float(ds_cycle["lon0"].values)
    dx, dy = latlon_to_enu_m(lat_end, lon_end, lat0, lon0)
    dx_ref, dy_ref = latlon_to_enu_m(anchor_lat, anchor_lon, lat0, lon0)
    dist = np.sqrt((dx - dx_ref) ** 2 + (dy - dy_ref) ** 2)
    assert dist < 1.0


def test_reconstruct_cycle_underwater_finite() -> None:
    _, ds_cycle = _build_cycle_from_synth()
    ds_rec = reconstruct_cycle_xy_from_ds(ds_cycle)

    mask = ds_rec["underwater_mask"].values.astype(bool)
    x_rec = ds_rec["x_enu"].values
    y_rec = ds_rec["y_enu"].values
    lat_rec = ds_rec["lat_rec"].values
    lon_rec = ds_rec["lon_rec"].values

    assert np.all(np.isfinite(x_rec[mask]))
    assert np.all(np.isfinite(y_rec[mask]))
    assert np.all(np.isfinite(lat_rec[mask]))
    assert np.all(np.isfinite(lon_rec[mask]))
