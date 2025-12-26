from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

from .writers import write_netcdf


PHASE_VAR_BASE: Dict[str, str] = {
    "park_drift": "parking",
    "ascent": "ascent",
    "descent_to_profile": "descent_to_profile",
    "profile_drift": "profile_drift",
    "surface": "surface",
    "in_air": "in_air",
    "grounded": "grounded",
    "other": "other",
}


@dataclass(frozen=True)
class BVPReadyConfig:
    """
    Minimal config for selecting which accelerations to expose.

    acc_source:
        "acc_lin" -> use gravity-removed components (acc_lin_ned_n/e) if present (default)
        "acc"     -> use total accelerations (acc_ned_n/e) if present
    acc_n_name / acc_e_name:
        Optional explicit variable names to override discovery.
    min_parking_samples_for_bvp / min_phase_samples_for_bvp:
        Optional overrides; fall back to ds_cycles attrs or default=10.
    """

    acc_source: str = "acc_lin"
    acc_n_name: str | None = None
    acc_e_name: str | None = None
    min_parking_samples_for_bvp: int | None = None
    min_phase_samples_for_bvp: int | None = None

    def resolve_acc_vars(self, ds_continuous: xr.Dataset) -> Tuple[str, str]:
        if self.acc_n_name and self.acc_e_name:
            return self.acc_n_name, self.acc_e_name

        source = self.acc_source.lower()
        preferred: List[Tuple[str, str]] = []
        if source in ("acc_lin", "lin", "acc_lin_ned"):
            preferred.append(("acc_lin_ned_n", "acc_lin_ned_e"))
            preferred.append(("acc_ned_n", "acc_ned_e"))
        elif source in ("acc", "raw", "acc_ned", "total"):
            preferred.append(("acc_ned_n", "acc_ned_e"))
            preferred.append(("acc_lin_ned_n", "acc_lin_ned_e"))
        else:
            raise ValueError(f"Unknown acc_source '{self.acc_source}'. Expected acc_lin | acc.")

        for names in preferred:
            if names[0] in ds_continuous and names[1] in ds_continuous:
                return names

        raise KeyError(
            f"Could not find acceleration components for acc_source='{self.acc_source}'. "
            f"Tried: {preferred}."
        )


def _require_vars(ds: xr.Dataset, required: Iterable[str], label: str) -> None:
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"{label} missing required variables: {missing}")


def _resolve_thresholds(cfg: BVPReadyConfig, ds_cycles: xr.Dataset) -> Tuple[int, int]:
    mp = cfg.min_parking_samples_for_bvp
    mp = int(mp) if mp is not None else int(ds_cycles.attrs.get("min_parking_samples_for_bvp", 10))
    mo = cfg.min_phase_samples_for_bvp
    mo = int(mo) if mo is not None else int(ds_cycles.attrs.get("min_phase_samples_for_bvp", 10))
    return mp, mo


def _segments_for_cycle_phase(
    ds_segments: xr.Dataset, cycle_number: int, phase_name: str
) -> List[Tuple[int, int]]:
    cyc = np.asarray(ds_segments["cycle_number"].values).astype(int)
    seg_name = np.asarray(ds_segments["segment_name"].values).astype(str)
    idx0 = np.asarray(ds_segments["idx0"].values).astype(int)
    idx1 = np.asarray(ds_segments["idx1"].values).astype(int)

    m = (cyc == int(cycle_number)) & (seg_name == phase_name)
    if not np.any(m):
        return []

    s0 = idx0[m]
    s1 = idx1[m]
    order = np.argsort(s0)
    return [(int(s0[i]), int(s1[i])) for i in order]


def _gather_phase_vars(required_phase_bases: Iterable[str]) -> Tuple[List[str], List[str]]:
    n_vars = []
    att_vars = []
    for base in required_phase_bases:
        n_vars.append(f"{base}_n_obs")
        att_vars.append(f"{base}_attendible")
    return n_vars, att_vars


def build_bvp_ready_dataset(
    ds_continuous: xr.Dataset,
    ds_cycles: xr.Dataset,
    ds_segments: xr.Dataset,
    *,
    cfg: BVPReadyConfig | None = None,
) -> xr.Dataset:
    """
    Extract a BVP-ready view including all attendible phases (parking mandatory).
    Includes only cycles with valid_for_bvp == True.
    """
    cfg = cfg or BVPReadyConfig()

    acc_n_var, acc_e_var = cfg.resolve_acc_vars(ds_continuous)

    phase_bases = list(PHASE_VAR_BASE.values())
    n_vars, att_vars = _gather_phase_vars(phase_bases)

    required_cycles = [
        "cycle_number",
        "t_park_start",
        "t_park_end",
        "valid_for_bvp",
        "parking_n_obs",
        "lat_surface_start",
        "lon_surface_start",
        "lat_surface_end",
        "lon_surface_end",
        "pos_age_s",
        "pos_source",
        "t_surface_end",
        "t_surface_start_fix",
        "t_surface_end_fix",
    ] + n_vars + att_vars
    _require_vars(ds_continuous, ["time", "pres", "cycle_number", acc_n_var, acc_e_var], "ds_continuous")
    _require_vars(ds_cycles, required_cycles, "ds_cycles")
    _require_vars(ds_segments, ["cycle_number", "idx0", "idx1", "segment_name"], "ds_segments")

    min_parking_samples, min_phase_samples = _resolve_thresholds(cfg, ds_cycles)
    parking_counts = np.asarray(ds_cycles["parking_n_obs"].values).astype(int)
    n_total = parking_counts.size
    n_too_few = int(np.sum(parking_counts < min_parking_samples))
    if n_too_few > 0:
        print(
            f"[bvp_ready] cycles skipped by parking sample threshold "
            f"(parking_n_obs < {min_parking_samples}): {n_too_few} / {n_total}"
        )

    valid_mask = ds_cycles["valid_for_bvp"].astype(bool)
    anchors_ok = (
        np.isfinite(ds_cycles["lat_surface_start"].values)
        & np.isfinite(ds_cycles["lon_surface_start"].values)
        & np.isfinite(ds_cycles["lat_surface_end"].values)
        & np.isfinite(ds_cycles["lon_surface_end"].values)
    )
    n_missing_anchors = int(np.sum(valid_mask & ~anchors_ok))
    if n_missing_anchors > 0:
        print(f"[bvp_ready] cycles skipped by anchor availability: {n_missing_anchors} / {int(valid_mask.size)}")
    valid_mask = valid_mask & anchors_ok
    ds_cyc_valid = ds_cycles.where(valid_mask, drop=True)
    cycle_numbers_all = np.asarray(ds_cyc_valid["cycle_number"].values).astype(int)

    time_all = np.asarray(ds_continuous["time"].values).astype("datetime64[ns]")
    pres_all = np.asarray(ds_continuous["pres"].values).astype(float)
    acc_n_all = np.asarray(ds_continuous[acc_n_var].values).astype(float)
    acc_e_all = np.asarray(ds_continuous[acc_e_var].values).astype(float)
    obs_time: List[np.datetime64] = []
    obs_acc_n: List[float] = []
    obs_acc_e: List[float] = []
    obs_pres: List[float] = []
    obs_phase: List[str] = []
    obs_cycle_number: List[int] = []
    obs_cycle_index: List[int] = []
    obs_index: List[int] = []

    row_start: List[int] = []
    row_size: List[int] = []
    row_t0: List[np.datetime64] = []
    row_t1: List[np.datetime64] = []

    cyc_meta_times: Dict[str, List[np.datetime64]] = {
        "t_cycle_start": [],
        "t_descent_to_profile_start": [],
        "t_profile_deepest": [],
        "t_ascent_start": [],
        "t_surface_start": [],
        "t_surface_end": [],
        "t_surface_start_fix": [],
        "t_surface_end_fix": [],
        "t_park_start": [],
        "t_park_end": [],
    }
    cyc_lat_surface_start: List[float] = []
    cyc_lon_surface_start: List[float] = []
    cyc_lat_surface_end: List[float] = []
    cyc_lon_surface_end: List[float] = []
    cyc_pos_age_s: List[float] = []
    cyc_pos_source: List[str] = []

    phase_att_out: Dict[str, List[bool]] = {base: [] for base in phase_bases}
    phase_n_out: Dict[str, List[int]] = {base: [] for base in phase_bases}

    cycles_out: List[int] = []
    offset = 0

    for cyc in cycle_numbers_all:
        row = ds_cyc_valid.sel(cycle=cyc)

        attendible_phases: List[str] = []
        for phase_name, base in PHASE_VAR_BASE.items():
            att_var = f"{base}_attendible"
            if att_var not in row:
                continue
            if bool(row[att_var].values):
                attendible_phases.append(phase_name)

        if "park_drift" not in attendible_phases:
            print(f"[bvp_ready] skip cycle {cyc}: parking not attendible")
            continue

        local_time: List[np.datetime64] = []
        local_acc_n: List[float] = []
        local_acc_e: List[float] = []
        local_pres: List[float] = []
        local_phase: List[str] = []
        local_obs_idx: List[int] = []

        for ph in attendible_phases:
            segs = _segments_for_cycle_phase(ds_segments, cyc, ph)
            if not segs:
                continue
            for (a, b) in segs:
                if b <= a:
                    continue
                idx = np.arange(a, b, dtype=int)
                local_time.extend(time_all[a:b])
                local_acc_n.extend(acc_n_all[a:b])
                local_acc_e.extend(acc_e_all[a:b])
                local_pres.extend(pres_all[a:b])
                local_phase.extend([ph] * (b - a))
                local_obs_idx.extend(idx.tolist())

        n_this = len(local_time)
        if n_this == 0:
            print(f"[bvp_ready] skip cycle {cyc}: no attendible phase samples found")
            continue

        idx_out = len(cycles_out)
        cycles_out.append(int(cyc))
        row_start.append(int(offset))
        row_size.append(int(n_this))
        row_t0.append(np.asarray(local_time[0]).astype("datetime64[ns]"))
        row_t1.append(np.asarray(local_time[-1]).astype("datetime64[ns]"))
        offset += n_this

        obs_time.extend(local_time)
        obs_acc_n.extend(local_acc_n)
        obs_acc_e.extend(local_acc_e)
        obs_pres.extend(local_pres)
        obs_phase.extend(local_phase)
        obs_cycle_number.extend([int(cyc)] * n_this)
        obs_cycle_index.extend([int(idx_out)] * n_this)
        obs_index.extend(local_obs_idx)

        for k in cyc_meta_times:
            cyc_meta_times[k].append(
                np.asarray(row[k].values).astype("datetime64[ns]") if k in row else np.datetime64("NaT")
            )
        cyc_lat_surface_start.append(float(row["lat_surface_start"].values))
        cyc_lon_surface_start.append(float(row["lon_surface_start"].values))
        cyc_lat_surface_end.append(float(row["lat_surface_end"].values))
        cyc_lon_surface_end.append(float(row["lon_surface_end"].values))
        cyc_pos_age_s.append(float(row["pos_age_s"].values))
        cyc_pos_source.append(str(row["pos_source"].values))

        for phase_name, base in PHASE_VAR_BASE.items():
            att_var = f"{base}_attendible"
            n_var = f"{base}_n_obs"
            att_val = bool(row[att_var].values) if att_var in row else False
            n_val = int(row[n_var].values) if n_var in row else 0
            phase_att_out[base].append(att_val)
            phase_n_out[base].append(n_val)

    n_obs = len(obs_time)
    cycles_out_arr = np.asarray(cycles_out, dtype=int)

    ds_out = xr.Dataset(
        coords=dict(
            obs=("obs", np.arange(n_obs, dtype=int)),
            cycle=("cycle", cycles_out_arr),
        ),
        data_vars=dict(
            cycle_number=("cycle", cycles_out_arr),
            row_start=("cycle", np.asarray(row_start, dtype=int)),
            row_size=("cycle", np.asarray(row_size, dtype=int)),
            t0=("cycle", np.asarray(row_t0, dtype="datetime64[ns]")),
            t1=("cycle", np.asarray(row_t1, dtype="datetime64[ns]")),
            lat_surface_start=("cycle", np.asarray(cyc_lat_surface_start, dtype=float)),
            lon_surface_start=("cycle", np.asarray(cyc_lon_surface_start, dtype=float)),
            lat_surface_end=("cycle", np.asarray(cyc_lat_surface_end, dtype=float)),
            lon_surface_end=("cycle", np.asarray(cyc_lon_surface_end, dtype=float)),
            pos_age_s=("cycle", np.asarray(cyc_pos_age_s, dtype=float)),
            pos_source=("cycle", np.asarray(cyc_pos_source, dtype=object)),
        ),
        attrs=dict(
            platform=str(
                ds_continuous.attrs.get(
                    "platform",
                    ds_cycles.attrs.get("platform", ""),
                )
            ),
            min_parking_samples_for_bvp=int(min_parking_samples),
            min_phase_samples_for_bvp=int(min_phase_samples),
            acc_source=f"{acc_n_var},{acc_e_var}",
            notes="BVP-ready view including all attendible phases; parking attendible is mandatory.",
            phase_name_map=str(PHASE_VAR_BASE),
        ),
    )

    for k, vals in cyc_meta_times.items():
        ds_out[k] = ("cycle", np.asarray(vals, dtype="datetime64[ns]"))
    for base, vals in phase_att_out.items():
        ds_out[f"{base}_attendible"] = ("cycle", np.asarray(vals, dtype=bool))
    for base, vals in phase_n_out.items():
        ds_out[f"{base}_n_obs"] = ("cycle", np.asarray(vals, dtype=int))

    ds_out["time"] = ("obs", np.asarray(obs_time, dtype="datetime64[ns]"))
    ds_out["acc_n"] = ("obs", np.asarray(obs_acc_n, dtype=float))
    ds_out["acc_e"] = ("obs", np.asarray(obs_acc_e, dtype=float))
    ds_out["z_from_pres"] = ("obs", np.asarray(obs_pres, dtype=float))
    ds_out["phase_name"] = ("obs", np.asarray(obs_phase, dtype=object))
    ds_out["cycle_number_for_obs"] = ("obs", np.asarray(obs_cycle_number, dtype=int))
    ds_out["cycle_index"] = ("obs", np.asarray(obs_cycle_index, dtype=int))
    ds_out["obs_index"] = ("obs", np.asarray(obs_index, dtype=int))

    if "units" in ds_continuous[acc_n_var].attrs:
        ds_out["acc_n"].attrs["units"] = ds_continuous[acc_n_var].attrs["units"]
        ds_out["acc_e"].attrs["units"] = ds_continuous[acc_n_var].attrs["units"]
    ds_out["acc_n"].attrs["source_var"] = acc_n_var
    ds_out["acc_e"].attrs["source_var"] = acc_e_var
    if "units" in ds_continuous["pres"].attrs:
        ds_out["z_from_pres"].attrs["units"] = ds_continuous["pres"].attrs["units"]
    ds_out["z_from_pres"].attrs["comment"] = "Pressure used as depth proxy (no conversion)."

    return ds_out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a BVP-ready NetCDF from preprocess outputs (all attendible phases; parking mandatory).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cont", required=True, help="Path to *_preprocessed_imu.nc")
    p.add_argument("--cycles", required=True, help="Path to *_cycles.nc (with valid_for_bvp + phase attendibility)")
    p.add_argument("--segments", required=True, help="Path to *_segments.nc")
    p.add_argument(
        "--acc-source",
        default="acc_lin",
        choices=["acc_lin", "acc", "lin", "raw"],
        help="Which acceleration components to expose (linear/gravity-removed vs total).",
    )
    p.add_argument("--acc-n-var", default=None, help="Optional override for the northward acceleration variable name.")
    p.add_argument("--acc-e-var", default=None, help="Optional override for the eastward acceleration variable name.")
    p.add_argument(
        "--out",
        required=True,
        help="Output path or directory. If a directory is given, writes <platform>_bvp_ready.nc inside it.",
    )
    return p


def _derive_out_path(out_arg: str, platform: str) -> Path:
    out_path = Path(out_arg)
    if out_path.suffix.lower() != ".nc":
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path / f"{platform}_bvp_ready.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _print_selection_summary(ds_bvp: xr.Dataset) -> None:
    n_cycles = int(ds_bvp.sizes.get("cycle", 0))
    n_obs = int(ds_bvp.sizes.get("obs", 0))
    print(f"[bvp_ready] selected cycles: {n_cycles}, observations: {n_obs}")
    if n_obs == 0:
        return
    phases = np.asarray(ds_bvp["phase_name"].values).astype(str)
    uniq, cnt = np.unique(phases, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"  phase={u:>18s} : {int(c)} samples")


def _print_anchor_summary(ds_bvp: xr.Dataset, n_show: int = 3) -> None:
    n_cycles = int(ds_bvp.sizes.get("cycle", 0))
    if n_cycles == 0:
        return

    lat_start = np.asarray(ds_bvp["lat_surface_start"].values, dtype=float)
    lon_start = np.asarray(ds_bvp["lon_surface_start"].values, dtype=float)
    lat_end = np.asarray(ds_bvp["lat_surface_end"].values, dtype=float)
    lon_end = np.asarray(ds_bvp["lon_surface_end"].values, dtype=float)
    anchors_ok = np.isfinite(lat_start) & np.isfinite(lon_start) & np.isfinite(lat_end) & np.isfinite(lon_end)
    print(f"[bvp_ready] cycles with both anchors: {int(np.sum(anchors_ok))} / {n_cycles}")

    if n_show <= 0:
        return

    n_show = min(n_show, n_cycles)
    cycles = np.asarray(ds_bvp["cycle_number"].values).astype(int)
    has_t_start = "t_surface_start_fix" in ds_bvp.variables
    has_t_end = "t_surface_end_fix" in ds_bvp.variables

    print(f"[bvp_ready] first {n_show} cycles anchors:")
    for i in range(n_show):
        row = ds_bvp.isel(cycle=i)
        t_start = row["t_surface_start_fix"].values if has_t_start else np.datetime64("NaT")
        t_end = row["t_surface_end_fix"].values if has_t_end else np.datetime64("NaT")
        lat_s = float(row["lat_surface_start"].values) if np.isfinite(row["lat_surface_start"].values) else np.nan
        lon_s = float(row["lon_surface_start"].values) if np.isfinite(row["lon_surface_start"].values) else np.nan
        lat_e = float(row["lat_surface_end"].values) if np.isfinite(row["lat_surface_end"].values) else np.nan
        lon_e = float(row["lon_surface_end"].values) if np.isfinite(row["lon_surface_end"].values) else np.nan
        print(
            f"  cyc={int(cycles[i])} "
            f"start=({lat_s:.6f},{lon_s:.6f}) t_start_fix={str(t_start)} "
            f"end=({lat_e:.6f},{lon_e:.6f}) t_end_fix={str(t_end)}"
        )


def main() -> None:
    args = _build_parser().parse_args()
    cfg = BVPReadyConfig(
        acc_source=args.acc_source,
        acc_n_name=args.acc_n_var,
        acc_e_name=args.acc_e_var,
    )

    ds_cont = xr.open_dataset(args.cont)
    ds_cyc = xr.open_dataset(args.cycles)
    ds_seg = xr.open_dataset(args.segments)

    ds_bvp = build_bvp_ready_dataset(ds_cont, ds_cyc, ds_seg, cfg=cfg)

    if ds_bvp.sizes.get("cycle", 0) == 0:
        total = int(ds_cyc.sizes.get("cycle", 0))
        print(f"No valid cycles for BVP-ready output (0/{total}).")
        ds_cont.close()
        ds_cyc.close()
        ds_seg.close()
        ds_bvp.close()
        raise SystemExit(2)

    _print_selection_summary(ds_bvp)

    platform = str(
        ds_bvp.attrs.get(
            "platform",
            ds_cont.attrs.get("platform", ds_cyc.attrs.get("platform", "platform")),
        )
    )
    out_path = _derive_out_path(args.out, platform)
    write_netcdf(ds_bvp, out_path)
    print(f"OK: wrote BVP-ready file to {out_path}")
    _print_anchor_summary(ds_bvp, n_show=3)

    ds_cont.close()
    ds_cyc.close()
    ds_seg.close()
    ds_bvp.close()


if __name__ == "__main__":
    main()
