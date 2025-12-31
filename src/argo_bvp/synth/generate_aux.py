"""Build synthetic AUX datasets from TRUTH."""

from __future__ import annotations

import numpy as np
import xarray as xr

from ..instruments import INSTRUMENTS, InstrumentParams


def build_aux_from_truth(
    truth: xr.Dataset,
    instrument: InstrumentParams | str = "synth_v1",
) -> xr.Dataset:
    """Create a Coriolis-like AUX dataset using TRUTH inputs."""
    if not isinstance(truth, xr.Dataset):
        raise TypeError("truth must be an xarray.Dataset")
    inst = _resolve_instrument(instrument)
    if inst.lsb_to_ms2 <= 0:
        raise ValueError("lsb_to_ms2 must be positive")

    t = _as_1d_float(truth, "t")
    n = t.size
    if n == 0:
        raise ValueError("truth has no observations")

    start_juld = _resolve_start_juld(truth)
    juld = start_juld + t / 86400.0
    juld_qc = np.full((n,), "1", dtype="U1")

    ax = _as_1d_float(truth, "ax")
    ay = _as_1d_float(truth, "ay")
    az = _as_1d_float(truth, "az")
    _require_same_length(t, ax, ay, az)

    pres = _get_pres(truth, n)

    acc_x = _acc_to_counts(ax, inst.lsb_to_ms2)
    acc_y = _acc_to_counts(ay, inst.lsb_to_ms2)
    acc_z = _acc_to_counts(az, inst.lsb_to_ms2)

    zeros = np.zeros((n,), dtype="int32")

    ds = xr.Dataset(
        data_vars={
            "JULD": ("N_MEASUREMENT", juld, {"units": "days"}),
            "JULD_QC": ("N_MEASUREMENT", juld_qc),
            "PRES": ("N_MEASUREMENT", pres, {"units": "decibar"}),
            "LINEAR_ACCELERATION_COUNT_X": ("N_MEASUREMENT", acc_x),
            "LINEAR_ACCELERATION_COUNT_Y": ("N_MEASUREMENT", acc_y),
            "LINEAR_ACCELERATION_COUNT_Z": ("N_MEASUREMENT", acc_z),
            "ANGULAR_RATE_COUNT_X": ("N_MEASUREMENT", zeros.copy()),
            "ANGULAR_RATE_COUNT_Y": ("N_MEASUREMENT", zeros.copy()),
            "ANGULAR_RATE_COUNT_Z": ("N_MEASUREMENT", zeros.copy()),
            "MAGNETIC_FIELD_COUNT_X": ("N_MEASUREMENT", zeros.copy()),
            "MAGNETIC_FIELD_COUNT_Y": ("N_MEASUREMENT", zeros.copy()),
            "MAGNETIC_FIELD_COUNT_Z": ("N_MEASUREMENT", zeros.copy()),
        },
        coords={
            "N_MEASUREMENT": np.arange(n, dtype=int),
        },
        attrs={
            "lsb_to_ms2": float(inst.lsb_to_ms2),
            "gyro_lsb_to_rads": float(inst.gyro_lsb_to_rads),
            "mag_lsb_to_uT": float(inst.mag_lsb_to_uT),
            "note": "gyro count rate unknown: placeholder",
            "source": "synthetic TRUTH",
        },
    )
    return ds


def _acc_to_counts(acc_ms2: np.ndarray, lsb_to_ms2: float) -> np.ndarray:
    counts = np.rint(np.asarray(acc_ms2, dtype=float) / float(lsb_to_ms2))
    return counts.astype("int32")


def _as_1d_float(ds: xr.Dataset, name: str) -> np.ndarray:
    if name not in ds:
        raise KeyError(f"Missing {name} in truth dataset")
    arr = np.asarray(ds[name].values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    return arr


def _resolve_start_juld(ds: xr.Dataset) -> float:
    if "start_juld" in ds.attrs:
        return float(ds.attrs["start_juld"])
    if "anchor_juld" in ds:
        anchor = np.asarray(ds["anchor_juld"].values, dtype=float)
        if anchor.size:
            return float(anchor.flat[0])
    raise KeyError("start_juld not found in TRUTH dataset")


def _resolve_instrument(instrument: InstrumentParams | str) -> InstrumentParams:
    if isinstance(instrument, InstrumentParams):
        return instrument
    if isinstance(instrument, str):
        key = instrument.strip()
        if key in INSTRUMENTS:
            return INSTRUMENTS[key]
        raise KeyError(f"Unknown instrument: {instrument}")
    raise TypeError("instrument must be InstrumentParams or str")


def _get_pres(ds: xr.Dataset, n: int) -> np.ndarray:
    if "pres" in ds:
        pres = np.asarray(ds["pres"].values, dtype=float)
    elif "z" in ds:
        z = np.asarray(ds["z"].values, dtype=float)
        pres = np.maximum(-z, 0.0)
    else:
        raise KeyError("TRUTH dataset missing pres or z")
    if pres.shape[0] != n:
        raise ValueError("pres must match t length")
    return pres.astype("float32", copy=False)


def _require_same_length(*arrays: np.ndarray) -> None:
    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("Input arrays must have the same length")


__all__ = ["build_aux_from_truth"]
