from __future__ import annotations

from typing import Dict, Iterable

MACRO_MAP: Dict[str, str] = {
    "park_drift": "parking",
    "parking": "parking",
    "ascent": "ascent",
    "descent": "descent",
    "descent_to_profile": "descent",
    "profile_drift": "profile",
    "surface": "surface",
    "in_air": "surface",
    "grounded": "grounded",
    "other": "other",
}


def macro_phase(phase_name: str) -> str:
    return MACRO_MAP.get(str(phase_name), "other")


def phase_stats(phases: Iterable[str], times_s) -> Dict[str, Dict[str, float]]:
    """
    Compute per-macro-phase n_obs and median dt (seconds).
    """
    import numpy as np

    phases = [macro_phase(p) for p in phases]
    phases_arr = np.asarray(phases, dtype=object)
    times_s = np.asarray(times_s, dtype=float)
    dt = np.diff(times_s)

    stats: Dict[str, Dict[str, float]] = {}
    for ph in ["descent", "parking", "ascent", "surface", "profile", "grounded", "other"]:
        m = phases_arr == ph
        n = int(m.sum())
        stats[ph] = {"n_obs": n, "median_dt_s": float(np.median(dt[m[:-1]])) if n > 1 else float("nan")}
    return stats
