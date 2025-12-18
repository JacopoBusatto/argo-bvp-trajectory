from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Iterable, List, Tuple

import numpy as np
import xarray as xr

from .writers import write_netcdf


@dataclass(frozen=True)
class BVPReadyConfig:
    """
    Minimal config for selecting which accelerations to expose.

    acc_source:
        "acc_lin" -> use gravity-removed components (acc_lin_ned_n/e) if present (default)
        "acc"     -> use total accelerations (acc_ned_n/e) if present
    acc_n_name / acc_e_name:
        Optional explicit variable names to override discovery.
    """

    acc_source: str = "acc_lin"
    acc_n_name: str | None = None
    acc_e_name: str | None = None

    def resolve_acc_vars(self, ds_continuous: xr.Dataset) -> Tuple[str, str]:
        """
        Decide which acceleration components to use (n/e).
        Preference order:
          1) explicit overrides
          2) requested source
          3) fallback to the other pair if requested not found
        """
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


def _parking_segments_for_cycle(ds_segments: xr.Dataset, cycle_number: int) -> List[Tuple[int, int]]:
    cyc = np.asarray(ds_segments["cycle_number"].values).astype(int)
    is_park = np.asarray(ds_segments["is_parking_phase"].values).astype(bool)
    idx0 = np.asarray(ds_segments["idx0"].values).astype(int)
    idx1 = np.asarray(ds_segments["idx1"].values).astype(int)

    m = (cyc == int(cycle_number)) & is_park
    if not np.any(m):
        return []

    s0 = idx0[m]
    s1 = idx1[m]
    order = np.argsort(s0)
    return [(int(s0[i]), int(s1[i])) for i in order]


def build_bvp_ready_dataset(
    ds_continuous: xr.Dataset,
    ds_cycles: xr.Dataset,
    ds_segments: xr.Dataset,
    *,
    cfg: BVPReadyConfig | None = None,
) -> xr.Dataset:
    """
    Extract a minimal, parking-phase-only view for BVP.
    Includes only cycles with valid_for_bvp == True.
    """
    cfg = cfg or BVPReadyConfig()

    acc_n_var, acc_e_var = cfg.resolve_acc_vars(ds_continuous)

    _require_vars(ds_continuous, ["time", "pres", "cycle_number", acc_n_var, acc_e_var], "ds_continuous")
    _require_vars(
        ds_cycles,
        [
            "cycle_number",
            "t_park_start",
            "t_park_end",
            "valid_for_bvp",
            "lat_surface_end",
            "lon_surface_end",
            "pos_age_s",
            "pos_source",
        ],
        "ds_cycles",
    )
    _require_vars(ds_segments, ["cycle_number", "idx0", "idx1", "is_parking_phase", "t0", "t1"], "ds_segments")

    # Filter valid cycles (keep xarray mask to preserve coords)
    valid_mask = ds_cycles["valid_for_bvp"].astype(bool)
    ds_cyc_valid = ds_cycles.where(valid_mask, drop=True)
    cycle_numbers = np.asarray(ds_cyc_valid["cycle_number"].values).astype(int)

    time_all = np.asarray(ds_continuous["time"].values).astype("datetime64[ns]")
    pres_all = np.asarray(ds_continuous["pres"].values).astype(float)
    acc_n_all = np.asarray(ds_continuous[acc_n_var].values).astype(float)
    acc_e_all = np.asarray(ds_continuous[acc_e_var].values).astype(float)

    sample_time: List[np.datetime64] = []
    sample_acc_n: List[float] = []
    sample_acc_e: List[float] = []
    sample_pres: List[float] = []
    sample_cycle_number: List[int] = []
    sample_cycle_index: List[int] = []
    sample_obs_index: List[int] = []

    row_start: List[int] = []
    row_size: List[int] = []
    row_idx0: List[int] = []
    row_idx1: List[int] = []
    row_t0: List[np.datetime64] = []
    row_t1: List[np.datetime64] = []

    offset = 0

    for i_cyc, cyc in enumerate(cycle_numbers):
        segs = _parking_segments_for_cycle(ds_segments, cyc)
        if not segs:
            raise RuntimeError(f"Cycle {cyc} is marked valid_for_bvp but has no parking segment.")

        row_idx0.append(int(segs[0][0]))
        row_idx1.append(int(segs[-1][1]))

        row_start.append(int(offset))

        n_this = 0
        for (a, b) in segs:
            if b <= a:
                continue
            obs_idx = np.arange(a, b, dtype=int)

            sample_time.extend(time_all[a:b])
            sample_acc_n.extend(acc_n_all[a:b])
            sample_acc_e.extend(acc_e_all[a:b])
            sample_pres.extend(pres_all[a:b])
            sample_cycle_number.extend([int(cyc)] * (b - a))
            sample_cycle_index.extend([int(i_cyc)] * (b - a))
            sample_obs_index.extend(obs_idx.tolist())

            n_this += (b - a)

        if n_this == 0:
            raise RuntimeError(f"Cycle {cyc} is marked valid_for_bvp but parking slice has zero samples.")
        row_size.append(int(n_this))
        offset += n_this

        if n_this > 0:
            row_t0.append(np.asarray(time_all[segs[0][0]]).astype("datetime64[ns]"))
            row_t1.append(np.asarray(time_all[segs[-1][1] - 1]).astype("datetime64[ns]"))
        else:
            row_t0.append(np.datetime64("NaT"))
            row_t1.append(np.datetime64("NaT"))

    n_samples = len(sample_time)

    ds_out = xr.Dataset(
        coords=dict(
            sample=("sample", np.arange(n_samples, dtype=int)),
            cycle=("cycle", cycle_numbers),
        ),
        data_vars=dict(
            cycle_number=("cycle", cycle_numbers),
            idx0=("cycle", np.asarray(row_idx0, dtype=int)),
            idx1=("cycle", np.asarray(row_idx1, dtype=int)),
            row_start=("cycle", np.asarray(row_start, dtype=int)),
            row_size=("cycle", np.asarray(row_size, dtype=int)),
            t0=("cycle", np.asarray(row_t0, dtype="datetime64[ns]")),
            t1=("cycle", np.asarray(row_t1, dtype="datetime64[ns]")),
            t_park_start=("cycle", np.asarray(ds_cyc_valid["t_park_start"].values, dtype="datetime64[ns]")),
            t_park_end=("cycle", np.asarray(ds_cyc_valid["t_park_end"].values, dtype="datetime64[ns]")),
            lat_surface_end=("cycle", np.asarray(ds_cyc_valid["lat_surface_end"].values, dtype=float)),
            lon_surface_end=("cycle", np.asarray(ds_cyc_valid["lon_surface_end"].values, dtype=float)),
            pos_age_s=("cycle", np.asarray(ds_cyc_valid["pos_age_s"].values, dtype=float)),
            pos_source=("cycle", np.asarray(ds_cyc_valid["pos_source"].values).astype(str)),
        ),
        attrs=dict(
            platform=str(
                ds_continuous.attrs.get(
                    "platform",
                    ds_cycles.attrs.get("platform", ""),
                )
            ),
            acc_source=f"{acc_n_var},{acc_e_var}",
            notes="Parking-phase-only view for BVP; samples come from ds_continuous slices of ds_segments.is_parking_phase.",
        ),
    )

    ds_out["time"] = ("sample", np.asarray(sample_time, dtype="datetime64[ns]"))
    ds_out["acc_n"] = ("sample", np.asarray(sample_acc_n, dtype=float))
    ds_out["acc_e"] = ("sample", np.asarray(sample_acc_e, dtype=float))
    ds_out["z_from_pres"] = ("sample", np.asarray(sample_pres, dtype=float))
    ds_out["cycle_number_for_sample"] = ("sample", np.asarray(sample_cycle_number, dtype=int))
    ds_out["cycle_index"] = ("sample", np.asarray(sample_cycle_index, dtype=int))
    ds_out["obs_index"] = ("sample", np.asarray(sample_obs_index, dtype=int))

    # Units/metadata
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
        description="Build a parking-phase BVP-ready NetCDF from preprocess outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cont", required=True, help="Path to *_preprocessed_imu.nc")
    p.add_argument("--cycles", required=True, help="Path to *_cycles.nc (with valid_for_bvp + surface fixes)")
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


def main() -> None:
    args = _build_parser().parse_args()
    cfg = BVPReadyConfig(acc_source=args.acc_source, acc_n_name=args.acc_n_var, acc_e_name=args.acc_e_var)

    ds_cont = xr.open_dataset(args.cont)
    ds_cyc = xr.open_dataset(args.cycles)
    ds_seg = xr.open_dataset(args.segments)

    ds_bvp = build_bvp_ready_dataset(ds_cont, ds_cyc, ds_seg, cfg=cfg)

    platform = str(
        ds_bvp.attrs.get(
            "platform",
            ds_cont.attrs.get("platform", ds_cyc.attrs.get("platform", "platform")),
        )
    )
    out_path = _derive_out_path(args.out, platform)
    write_netcdf(ds_bvp, out_path)
    print(f"OK: wrote BVP-ready file to {out_path}")

    ds_cont.close()
    ds_cyc.close()
    ds_seg.close()
    ds_bvp.close()


if __name__ == "__main__":
    main()
