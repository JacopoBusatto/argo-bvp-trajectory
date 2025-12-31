"""Smoke test for the sweep CLI command."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_cli_sweep_smoke(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "argo_bvp.cli",
        "sweep",
        "--outdir",
        str(tmp_path),
        "--dt-descent-s-list",
        "5",
        "--dt-park-s-list",
        "10",
        "--dt-ascent-s-list",
        "5",
        "--acc-sigma-ms2-list",
        "0",
        "--park-hours-list",
        "8",
        "--seed",
        "0",
        "--instrument",
        "synth_v1",
        "--window-index",
        "0",
        "--method",
        "trap",
    ]
    subprocess.run(cmd, check=True, cwd=root)

    traj_files = list(tmp_path.rglob("*_TRAJ.nc"))
    aux_files = list(tmp_path.rglob("*_AUX.nc"))
    cycle_files = list(tmp_path.rglob("CYCLE_*_W000.nc"))

    assert traj_files
    assert aux_files
    assert cycle_files
