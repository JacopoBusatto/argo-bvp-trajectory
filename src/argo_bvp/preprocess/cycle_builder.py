"""Build cycle files from TRAJ and AUX inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import xarray as xr

from ..cycle_schema import make_empty_cycle_dataset, validate_cycle_dataset
from ..instruments import INSTRUMENTS, InstrumentParams
from ..io.aux_reader import extract_imu_arrays, read_aux
from ..io.traj_reader import extract_traj_positions, read_traj
from .surface_windows import find_surface_windows, select_anchor_points


DEFAULT_CONFIG: dict[str, object] = {
    "p_surface": 5.0,
    "max_gap_seconds": None,
    "qc_ok": {"1", "2", "5"},
    "float_id": None,
    "cycle_number": None,
    "traj_vars": {
        "juld": "JULD",
        "pres": "PRES",
        "lat": "LATITUDE",
        "lon": "LONGITUDE",
        "position_qc": "POSITION_QC",
    },
    "aux_vars": {
        "juld": "JULD",
        "pres": "PRES",
        "lin_acc_count": "LIN_ACC_COUNT",
        "ang_rate_count": "ANG_RATE_COUNT",
        "mag_field_count": "MAG_FIELD_COUNT",
    },
    "park_eps_dbar": 5.0,
}


def build_cycle_from_traj_aux(
    traj_path: str | Path | xr.Dataset,
    aux_path: str | Path | xr.Dataset,
    window_index: int,
    instrument: InstrumentParams | str | None = None,
    config: Mapping[str, object] | None = None,
) -> xr.Dataset:
    """Build a cycle dataset from TRAJ and AUX inputs."""
    cfg = _merge_config(config)
    inst = _resolve_instrument(instrument)

    traj_ds = _load_dataset(traj_path, read_traj)
    aux_ds = _load_dataset(aux_path, read_aux)
    aux_vars = _resolve_aux_vars(aux_ds, cfg["aux_vars"])

    traj_data = extract_traj_positions(traj_ds, cfg["traj_vars"])
    windows = find_surface_windows(
        traj_data["juld"],
        traj_data["pres"],
        p_surface=float(cfg["p_surface"]),
        max_gap_seconds=cfg["max_gap_seconds"],
    )

    start_anchor, end_anchor = select_anchor_points(
        traj_data["juld"],
        traj_data["pres"],
        traj_data["lat"],
        traj_data["lon"],
        traj_data["position_qc"],
        windows=windows,
        window_index=int(window_index),
        qc_ok=cfg["qc_ok"],
    )

    aux_data = extract_imu_arrays(aux_ds, aux_vars)
    juld_aux = aux_data["juld"]
    pres_aux = aux_data["pres"]

    mask = _between(juld_aux, start_anchor["juld"], end_anchor["juld"])
    if not np.any(mask):
        raise ValueError("No AUX samples between anchor start and end")

    juld_sel = juld_aux[mask]
    pres_sel = pres_aux[mask]
    lin_acc_count = aux_data["lin_acc_count"][mask]
    ang_rate_count = aux_data["ang_rate_count"][mask]
    mag_field_count = aux_data["mag_field_count"][mask]

    float_id = _resolve_float_id(traj_ds, cfg.get("float_id"))
    cycle_number = _resolve_cycle_number(traj_ds, cfg.get("cycle_number"), window_index)

    ds = make_empty_cycle_dataset(
        n_obs=int(juld_sel.shape[0]),
        float_id=float_id,
        cycle_number=cycle_number,
    )

    ds.attrs["time_origin_juld"] = float(start_anchor["juld"])

    ds["anchor_juld"].values[:] = [start_anchor["juld"], end_anchor["juld"]]
    ds["anchor_lat"].values[:] = [start_anchor["lat"], end_anchor["lat"]]
    ds["anchor_lon"].values[:] = [start_anchor["lon"], end_anchor["lon"]]
    ds["anchor_position_qc"].values[:] = [
        _qc_to_int(start_anchor["position_qc"]),
        _qc_to_int(end_anchor["position_qc"]),
    ]

    ds["lat0"].values = float(start_anchor["lat"])
    ds["lon0"].values = float(start_anchor["lon"])

    ds["juld"].values[:] = juld_sel
    ds["pres"].values[:] = pres_sel
    ds["lin_acc_count"].values[:, :] = lin_acc_count
    ds["ang_rate_count"].values[:, :] = ang_rate_count
    ds["mag_field_count"].values[:, :] = mag_field_count
    ds["lin_acc"].values[:, :] = lin_acc_count * float(inst.lsb_to_ms2)
    ds["ang_rate"].values[:, :] = ang_rate_count * float(inst.gyro_lsb_to_rads)
    ds["mag_field"].values[:, :] = mag_field_count * float(inst.mag_lsb_to_uT)

    t_seconds = (juld_sel - float(start_anchor["juld"])) * 86400.0
    ds = ds.assign_coords(t=("obs", t_seconds))

    ds["phase"].values[:] = _classify_phase(
        pres_sel,
        p_surface=float(cfg["p_surface"]),
        park_eps=float(cfg["park_eps_dbar"]),
    )

    validate_cycle_dataset(ds, strict=True)
    return ds


def _merge_config(config: Mapping[str, object] | None) -> dict[str, object]:
    cfg: dict[str, object] = {
        "traj_vars": DEFAULT_CONFIG["traj_vars"].copy(),
        "aux_vars": DEFAULT_CONFIG["aux_vars"].copy(),
    }
    for key, value in DEFAULT_CONFIG.items():
        if key not in cfg:
            cfg[key] = value
    if config:
        for key, value in config.items():
            if key in {"traj_vars", "aux_vars"} and isinstance(value, Mapping):
                cfg[key].update(value)
            else:
                cfg[key] = value
    cfg["qc_ok"] = {str(code).strip() for code in cfg["qc_ok"]}
    return cfg


def _resolve_instrument(instrument: InstrumentParams | str | None) -> InstrumentParams:
    if instrument is None:
        instrument = "synth_v1"
    if isinstance(instrument, InstrumentParams):
        return instrument
    if isinstance(instrument, str):
        key = instrument.strip()
        if key in INSTRUMENTS:
            return INSTRUMENTS[key]
        raise KeyError(f"Unknown instrument: {instrument}")
    raise TypeError("instrument must be InstrumentParams, str, or None")


def _resolve_aux_vars(aux_ds: xr.Dataset, aux_vars: Mapping[str, object]) -> dict[str, object]:
    mapping = dict(aux_vars)

    component_map = {
        "lin_acc_count": (
            "LINEAR_ACCELERATION_COUNT_X",
            "LINEAR_ACCELERATION_COUNT_Y",
            "LINEAR_ACCELERATION_COUNT_Z",
        ),
        "ang_rate_count": (
            "ANGULAR_RATE_COUNT_X",
            "ANGULAR_RATE_COUNT_Y",
            "ANGULAR_RATE_COUNT_Z",
        ),
        "mag_field_count": (
            "MAGNETIC_FIELD_COUNT_X",
            "MAGNETIC_FIELD_COUNT_Y",
            "MAGNETIC_FIELD_COUNT_Z",
        ),
    }
    fallback_map = {
        "lin_acc_count": "LIN_ACC_COUNT",
        "ang_rate_count": "ANG_RATE_COUNT",
        "mag_field_count": "MAG_FIELD_COUNT",
    }

    for key, fallback_name in fallback_map.items():
        spec = mapping.get(key)
        component_names = component_map[key]
        if isinstance(spec, str):
            if spec not in aux_ds.variables and all(name in aux_ds.variables for name in component_names):
                mapping[key] = component_names
        elif isinstance(spec, (list, tuple)):
            missing = any(name not in aux_ds.variables for name in spec)
            if missing and fallback_name in aux_ds.variables:
                mapping[key] = fallback_name
    return mapping


def _load_dataset(
    obj: str | Path | xr.Dataset,
    loader: Callable[[str | Path], xr.Dataset],
) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    return loader(obj)


def _between(values: np.ndarray, start: float, end: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr) & (arr >= float(start)) & (arr <= float(end))


def _resolve_float_id(ds: xr.Dataset, override: object | None) -> object:
    if override is not None:
        return override
    for key in ("PLATFORM_NUMBER", "FLOAT_ID", "float_id"):
        if key in ds.attrs:
            return ds.attrs[key]
    return "UNKNOWN"


def _resolve_cycle_number(
    ds: xr.Dataset,
    override: object | None,
    window_index: int,
) -> int:
    if override is not None:
        return int(override)
    for key in ("CYCLE_NUMBER", "cycle_number"):
        if key in ds.attrs:
            return int(ds.attrs[key])
    return int(window_index)


def _qc_to_int(qc_value: object) -> int:
    try:
        return int(str(qc_value).strip())
    except (TypeError, ValueError):
        return 0


def _classify_phase(pres: np.ndarray, p_surface: float, park_eps: float) -> np.ndarray:
    pres_arr = np.asarray(pres, dtype=float)
    n = pres_arr.shape[0]
    phase = np.zeros((n,), dtype="int8")
    if n == 0:
        return phase

    surface_mask = pres_arr <= float(p_surface)
    phase[surface_mask] = 1

    finite_mask = np.isfinite(pres_arr)
    if not np.any(finite_mask):
        return phase

    max_idx = int(np.nanargmax(pres_arr))
    max_pres = pres_arr[max_idx]
    park_mask = np.zeros_like(surface_mask)
    if np.isfinite(max_pres):
        park_mask = (
            (pres_arr >= (max_pres - park_eps))
            & (pres_arr <= (max_pres + park_eps))
            & ~surface_mask
        )

    descent_mask = np.arange(n) < max_idx
    ascent_mask = np.arange(n) > max_idx

    phase[descent_mask & ~surface_mask & ~park_mask] = 2
    phase[park_mask] = 3
    phase[ascent_mask & ~surface_mask & ~park_mask] = 4
    return phase


__all__ = ["build_cycle_from_traj_aux"]
