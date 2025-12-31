"""Smoke test for sweep metrics analysis."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from argo_bvp.analysis.sweep_analysis import build_metrics_table
from argo_bvp.instruments import INSTRUMENTS
from argo_bvp.run_integrate import integrate_cycle_file
from argo_bvp.run_preprocess import build_cycle_file, derive_base_from_traj_path
from argo_bvp.synth.experiment_params import DEFAULT_EXPERIMENT
from argo_bvp.synth.generate_synthetic_raw import generate_synthetic_raw


def test_sweep_metrics_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "SYNTH_CY24h_d5s_p10s_a5s_n0_W000"
    params = replace(
        DEFAULT_EXPERIMENT,
        dt_descent_s=5.0,
        dt_park_s=10.0,
        dt_ascent_s=5.0,
        acc_sigma_ms2=0.0,
    )
    generate_synthetic_raw(run_dir, params, INSTRUMENTS["synth_v1"])

    traj_path = next(run_dir.glob("*_TRAJ.nc"))
    aux_path = next(run_dir.glob("*_AUX.nc"))
    base = derive_base_from_traj_path(traj_path)
    cycle_path = run_dir / f"CYCLE_{base}_W000.nc"

    build_cycle_file(
        traj_path=traj_path,
        aux_path=aux_path,
        out_path=cycle_path,
        window_index=0,
        instrument="synth_v1",
    )
    integrate_cycle_file(cycle_path, run_dir, method="trap")

    df = build_metrics_table(tmp_path)
    csv_path = tmp_path / "analysis" / "metrics.csv"
    assert csv_path.exists()

    loaded = pd.read_csv(csv_path)
    expected_cols = {
        "tag",
        "window_index",
        "cycle_hours",
        "dt_descent_s",
        "dt_park_s",
        "dt_ascent_s",
        "acc_sigma_ms2",
        "rms_underwater_m",
        "err_park_start_m",
        "err_park_end_m",
        "err_delta_dive_to_parkstart_m",
        "err_delta_parkend_to_emerge_m",
    }
    assert expected_cols.issubset(set(loaded.columns))
    assert len(df) == 1
