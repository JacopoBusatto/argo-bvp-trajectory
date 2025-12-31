"""Tests for preprocess naming helpers."""

from pathlib import Path

from argo_bvp.run_preprocess import derive_base_from_traj_path


def test_derive_base_strips_traj_suffix() -> None:
    path = Path("SYNTH_CY24h_d5s_p10s_a5s_n0_TRAJ.nc")
    assert derive_base_from_traj_path(path) == "SYNTH_CY24h_d5s_p10s_a5s_n0"


def test_derive_base_keeps_stem_without_suffix() -> None:
    path = Path("CUSTOM_INPUT.nc")
    assert derive_base_from_traj_path(path) == "CUSTOM_INPUT"
