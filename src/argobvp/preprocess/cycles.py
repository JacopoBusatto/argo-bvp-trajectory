from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import xarray as xr

from .config import PreprocessConfig


# -----------------------------
# Phase names (strings)
# -----------------------------
PHASE_OTHER = "other"
PHASE_SURFACE = "surface"
PHASE_PARK_DRIFT = "park_drift"
PHASE_DESCENT_TO_PROFILE = "descent_to_profile"
PHASE_PROFILE_DRIFT = "profile_drift"
PHASE_ASCENT = "ascent"
PHASE_IN_AIR = "in_air"
PHASE_GROUNDED = "grounded"


@dataclass(frozen=True)
class PhaseRules:
    """
    Mapping MEASUREMENT_CODE -> phase.

    IMPORTANT:
    - This mapping is intended to be float/dataset-aware.
    - For your current AUX, the observed codes are: 590, 711, 290, 503.
      So we MUST map 290 (park) and 590/503 (ascent), 711 (in_air).

    If later you find additional codes (e.g., 389/489/689), you can add them here
    in the appropriate tuples.
    """

    # Park drift / park actions (include both 289 and 290 variants)
    park_codes: Tuple[int, ...] = (289, 290, 297, 298, 301)

    # Descent from park to deeper profile (if present)
    descent_to_profile_codes: Tuple[int, ...] = (389, 398)

    # Drift at deep/profile pressure (if present)
    profile_drift_codes: Tuple[int, ...] = (489, 497, 498)

    # Ascent/profile up actions
    ascent_codes: Tuple[int, ...] = (503, 589, 590, 599)

    # Surface sequence marker (often 689, but your file may not have it)
    surface_codes: Tuple[int, ...] = (689,)

    # In-air samples
    in_air_codes: Tuple[int, ...] = (711, 799)

    # Grounded information
    grounded_codes: Tuple[int, ...] = (901,)

    # Cycle start marker (optional, used only for a keypoint if present)
    cycle_start_codes: Tuple[int, ...] = (89,)

    def phase_for_code(self, code: int) -> str:
        if code in self.grounded_codes:
            return PHASE_GROUNDED
        if code in self.in_air_codes:
            return PHASE_IN_AIR
        if code in self.surface_codes:
            return PHASE_SURFACE
        if code in self.park_codes:
            return PHASE_PARK_DRIFT
        if code in self.profile_drift_codes:
            return PHASE_PROFILE_DRIFT
        if code in self.ascent_codes:
            return PHASE_ASCENT
        if code in self.descent_to_profile_codes:
            return PHASE_DESCENT_TO_PROFILE
        return PHASE_OTHER


DEFAULT_RULES = PhaseRules()


def assign_phase(
    measurement_code: np.ndarray,
    pres: np.ndarray,
    pres_surface_max: float,
    rules: PhaseRules = DEFAULT_RULES,
) -> np.ndarray:
    """
    Assign a phase name per observation.

    Rules:
      1) If MEASUREMENT_CODE matches known sets -> phase.
      2) Otherwise, fallback to surface if pres <= pres_surface_max.
      3) Otherwise -> other.

    Notes:
    - This is deliberately conservative.
    - It DOES NOT create 'descent_to_park' without explicit codes.
    """
    mc = np.asarray(measurement_code).reshape(-1).astype(int)
    p = np.asarray(pres).reshape(-1).astype(float)

    out = np.full(mc.shape, PHASE_OTHER, dtype=object)

    for i, c in enumerate(mc):
        ph = rules.phase_for_code(int(c))
        if ph != PHASE_OTHER:
            out[i] = ph

    # fallback surface (only if not already tagged)
    shallow = (out == PHASE_OTHER) & (p <= float(pres_surface_max))
    out[shallow] = PHASE_SURFACE

    return out


def _contiguous_segments(phase: np.ndarray) -> List[Tuple[int, int, str]]:
    """
    Build contiguous segments based on constant phase.
    Returns list of (i0, i1, phase_name) with i1 exclusive.
    """
    phase = np.asarray(phase, dtype=object).reshape(-1)
    if phase.size == 0:
        return []

    segs: List[Tuple[int, int, str]] = []
    i0 = 0
    cur = phase[0]
    for i in range(1, phase.size):
        if phase[i] != cur:
            segs.append((i0, i, str(cur)))
            i0 = i
            cur = phase[i]
    segs.append((i0, phase.size, str(cur)))
    return segs


def _first_time_of_code(t: np.ndarray, mc: np.ndarray, code: int) -> Optional[np.datetime64]:
    m = (mc == code)
    if not np.any(m):
        return None
    return t[int(np.argmax(m))]


def _first_time_of_any_code(t: np.ndarray, mc: np.ndarray, codes: Tuple[int, ...]) -> Optional[np.datetime64]:
    for c in codes:
        tt = _first_time_of_code(t, mc, c)
        if tt is not None:
            return tt
    return None


def _pres_at_first_code(p: np.ndarray, mc: np.ndarray, code: int) -> Optional[float]:
    m = (mc == code)
    if not np.any(m):
        return None
    idx = int(np.argmax(m))
    return float(p[idx])


def build_cycle_products(ds_continuous: xr.Dataset, cfg: PreprocessConfig) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    Build:
      - ds_cycles: one row per cycle with keypoint times/pressures
      - ds_segments: one row per (cycle, contiguous phase segment) with idx0/idx1 and t0/t1

    Input ds_continuous must contain:
      time[obs], pres[obs], cycle_number[obs], measurement_code[obs]
    """
    for r in ["time", "pres", "cycle_number", "measurement_code"]:
        if r not in ds_continuous:
            raise KeyError(f"ds_continuous missing required variable: {r}")

    t = np.asarray(ds_continuous["time"].values).reshape(-1).astype("datetime64[ns]")
    p = np.asarray(ds_continuous["pres"].values).reshape(-1).astype(float)
    cyc = np.asarray(ds_continuous["cycle_number"].values).reshape(-1).astype(int)
    mc = np.asarray(ds_continuous["measurement_code"].values).reshape(-1).astype(int)

    # Phase per obs
    phase = assign_phase(mc, p, cfg.pres_surface_max, rules=DEFAULT_RULES)

    # Unique cycles
    cycles = np.unique(cyc)

    # ------------------------
    # Segments (per cycle)
    # ------------------------
    seg_cycle: List[int] = []
    seg_name: List[str] = []
    seg_i0: List[int] = []
    seg_i1: List[int] = []
    seg_t0: List[np.datetime64] = []
    seg_t1: List[np.datetime64] = []

    # ------------------------
    # Cycles keypoints
    # ------------------------
    cycle_number_out: List[int] = []

    t_cycle_start: List[np.datetime64] = []
    t_park_start: List[np.datetime64] = []
    t_descent_to_profile_start: List[np.datetime64] = []
    t_profile_deepest: List[np.datetime64] = []
    t_ascent_start: List[np.datetime64] = []
    t_surface_start: List[np.datetime64] = []
    t_surface_end: List[np.datetime64] = []
    t_park_end: List[np.datetime64] = []

    pres_park_rep: List[float] = []
    pres_profile_deepest: List[float] = []

    parking_n_obs: List[int] = []
    parking_attendible: List[bool] = []
    park_sampled: List[bool] = []
    valid_for_bvp: List[bool] = []

    T_NAT = np.datetime64("NaT", "ns")
    V_NAN = np.nan

    for c in cycles:
        m = (cyc == int(c))
        if not np.any(m):
            continue

        idx = np.where(m)[0]
        i0c, i1c = int(idx[0]), int(idx[-1]) + 1

        tc = t[i0c:i1c]
        pc = p[i0c:i1c]
        mcc = mc[i0c:i1c]
        phc = phase[i0c:i1c]

        # segments inside this cycle
        segs = _contiguous_segments(phc)
        for (a, b, name) in segs:
            seg_cycle.append(int(c))
            seg_name.append(name)
            seg_i0.append(i0c + a)
            seg_i1.append(i0c + b)
            seg_t0.append(tc[a])
            seg_t1.append(tc[b - 1])

        # ---- keypoints (explicit codes first, then fallbacks) ----

        # cycle start: code 89 if present, else first time in cycle
        tcs = _first_time_of_any_code(tc, mcc, DEFAULT_RULES.cycle_start_codes)
        if tcs is None:
            tcs = tc[0]

        # park start: first park code (289/290/301/...)
        tps = _first_time_of_any_code(tc, mcc, DEFAULT_RULES.park_codes)

        # park end: last sample labeled as park drift (or NaT if none)
        park_idx = np.where(phc == PHASE_PARK_DRIFT)[0]
        tpe = tc[int(park_idx[-1])] if park_idx.size > 0 else None

        # Parking sampling counts/flags
        n_park = int(park_idx.size)
        parking_n_obs.append(n_park)
        park_has_samples = n_park > 0
        park_sampled.append(bool(park_has_samples))
        attendible = n_park >= int(cfg.min_parking_samples_for_bvp)
        parking_attendible.append(bool(attendible))
        valid_for_bvp.append(bool(attendible))

        # descent to profile start: first descent-to-profile code
        tdps = _first_time_of_any_code(tc, mcc, DEFAULT_RULES.descent_to_profile_codes)

        # deepest point: prefer code 503 time, else time of max pressure
        tdeep = _first_time_of_code(tc, mcc, 503)
        if tdeep is None:
            tdeep = tc[int(np.argmax(pc))]

        # ascent start: prefer first ascent code time, else first sample labeled ascent
        tas = _first_time_of_any_code(tc, mcc, DEFAULT_RULES.ascent_codes)
        if tas is None:
            ascent_idx = np.where(phc == PHASE_ASCENT)[0]
            tas = tc[int(ascent_idx[0])] if ascent_idx.size > 0 else None

        # surface start: prefer surface code, else first shallow sample (not already tagged)
        tss = _first_time_of_any_code(tc, mcc, DEFAULT_RULES.surface_codes)
        if tss is None:
            shallow = np.where(pc <= float(cfg.pres_surface_max))[0]
            tss = tc[int(shallow[0])] if shallow.size > 0 else None

        # surface end: prefer last in-air sample time if present, else last shallow, else last time
        inair = np.where((mcc == 711) | (mcc == 799))[0]
        if inair.size > 0:
            tse = tc[int(inair[-1])]
        else:
            shallow = np.where(pc <= float(cfg.pres_surface_max))[0]
            tse = tc[int(shallow[-1])] if shallow.size > 0 else tc[-1]

        # representative park pressure: prefer code 301 pressure, else median pressure during park_drift
        ppark = _pres_at_first_code(pc, mcc, 301)
        if ppark is None:
            park_idx = np.where(phc == PHASE_PARK_DRIFT)[0]
            ppark = float(np.median(pc[park_idx])) if park_idx.size > 0 else V_NAN

        # deepest pressure: prefer pressure at code 503, else max pressure
        pdeep = _pres_at_first_code(pc, mcc, 503)
        if pdeep is None:
            pdeep = float(np.max(pc))

        # store
        cycle_number_out.append(int(c))
        t_cycle_start.append(tcs if tcs is not None else T_NAT)
        t_park_start.append(tps if tps is not None else T_NAT)
        t_park_end.append(tpe if tpe is not None else T_NAT)
        t_descent_to_profile_start.append(tdps if tdps is not None else T_NAT)
        t_profile_deepest.append(tdeep if tdeep is not None else T_NAT)
        t_ascent_start.append(tas if tas is not None else T_NAT)
        t_surface_start.append(tss if tss is not None else T_NAT)
        t_surface_end.append(tse if tse is not None else T_NAT)

        pres_park_rep.append(ppark)
        pres_profile_deepest.append(pdeep)

    ds_cycles = xr.Dataset(
        coords=dict(cycle=("cycle", np.asarray(cycle_number_out, dtype=int))),
        data_vars=dict(
            cycle_number=("cycle", np.asarray(cycle_number_out, dtype=int)),

            t_cycle_start=("cycle", np.asarray(t_cycle_start, dtype="datetime64[ns]")),
            t_park_start=("cycle", np.asarray(t_park_start, dtype="datetime64[ns]")),
            t_park_end=("cycle", np.asarray(t_park_end, dtype="datetime64[ns]")),
            t_descent_to_profile_start=("cycle", np.asarray(t_descent_to_profile_start, dtype="datetime64[ns]")),
            t_profile_deepest=("cycle", np.asarray(t_profile_deepest, dtype="datetime64[ns]")),
            t_ascent_start=("cycle", np.asarray(t_ascent_start, dtype="datetime64[ns]")),
            t_surface_start=("cycle", np.asarray(t_surface_start, dtype="datetime64[ns]")),
            t_surface_end=("cycle", np.asarray(t_surface_end, dtype="datetime64[ns]")),

            pres_park_rep=("cycle", np.asarray(pres_park_rep, dtype=float)),
            pres_profile_deepest=("cycle", np.asarray(pres_profile_deepest, dtype=float)),

            parking_n_obs=("cycle", np.asarray(parking_n_obs, dtype=int)),
            parking_attendible=("cycle", np.asarray(parking_attendible, dtype=bool)),
            park_sampled=("cycle", np.asarray(park_sampled, dtype=bool)),
            valid_for_bvp=("cycle", np.asarray(valid_for_bvp, dtype=bool)),
        ),
        attrs=dict(
            platform=str(ds_continuous.attrs.get("platform", "")),
            pres_surface_max=float(cfg.pres_surface_max),
            min_parking_samples_for_bvp=int(cfg.min_parking_samples_for_bvp),
            notes="Keypoints derived from MEASUREMENT_CODE when available; fallbacks use pressure thresholds/extrema.",
        ),
    )
    ds_cycles["pres_park_rep"].attrs["units"] = "dbar"
    ds_cycles["pres_profile_deepest"].attrs["units"] = "dbar"

    ds_segments = xr.Dataset(
        coords=dict(segment=("segment", np.arange(len(seg_cycle), dtype=int))),
        data_vars=dict(
            cycle_number=("segment", np.asarray(seg_cycle, dtype=int)),
            segment_name=("segment", np.asarray(seg_name, dtype=object)),
            idx0=("segment", np.asarray(seg_i0, dtype=int)),
            idx1=("segment", np.asarray(seg_i1, dtype=int)),
            t0=("segment", np.asarray(seg_t0, dtype="datetime64[ns]")),
            t1=("segment", np.asarray(seg_t1, dtype="datetime64[ns]")),
            is_parking_phase=("segment", np.asarray([name == PHASE_PARK_DRIFT for name in seg_name], dtype=bool)),
        ),
        attrs=dict(
            platform=str(ds_continuous.attrs.get("platform", "")),
            pres_surface_max=float(cfg.pres_surface_max),
            notes="Segments are contiguous runs of constant phase within each cycle. idx0/idx1 refer to obs indices in the continuous dataset.",
        ),
    )

    return ds_cycles, ds_segments
