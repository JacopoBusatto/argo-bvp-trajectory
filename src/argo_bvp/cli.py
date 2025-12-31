"""Command-line interface for argo-bvp utilities."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .instruments import INSTRUMENTS
from .synth.experiment_params import DEFAULT_EXPERIMENT
from .synth.generate_synthetic_raw import generate_synthetic_raw


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="argo-bvp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="Generate synthetic TRUTH/TRAJ/AUX data")
    synth.add_argument("--outdir", type=Path, default=Path("outputs/synthetic"))
    synth.add_argument("--cycle-hours", type=float, default=DEFAULT_EXPERIMENT.cycle_hours)
    synth.add_argument("--start-juld", type=float, default=DEFAULT_EXPERIMENT.start_juld)
    synth.add_argument("--lat0", type=float, default=DEFAULT_EXPERIMENT.lat0)
    synth.add_argument("--lon0", type=float, default=DEFAULT_EXPERIMENT.lon0)
    synth.add_argument("--dt-surface-s", type=float, default=DEFAULT_EXPERIMENT.dt_surface_s)
    synth.add_argument("--dt-descent-s", type=float, default=DEFAULT_EXPERIMENT.dt_descent_s)
    synth.add_argument("--dt-park-s", type=float, default=DEFAULT_EXPERIMENT.dt_park_s)
    synth.add_argument("--dt-ascent-s", type=float, default=DEFAULT_EXPERIMENT.dt_ascent_s)
    synth.add_argument("--transition-seconds", type=float, default=DEFAULT_EXPERIMENT.transition_seconds)
    synth.add_argument("--surface1-minutes", type=float, default=DEFAULT_EXPERIMENT.surface1_minutes)
    synth.add_argument("--descent-hours", type=float, default=DEFAULT_EXPERIMENT.descent_hours)
    synth.add_argument("--park-depth-m", type=float, default=DEFAULT_EXPERIMENT.park_depth_m)
    synth.add_argument("--park-hours", type=float, default=DEFAULT_EXPERIMENT.park_hours)
    synth.add_argument("--ascent-hours", type=float, default=DEFAULT_EXPERIMENT.ascent_hours)
    synth.add_argument("--surface2-minutes", type=float, default=DEFAULT_EXPERIMENT.surface2_minutes)
    synth.add_argument("--spiral-radius-m", type=float, default=DEFAULT_EXPERIMENT.spiral_radius_m)
    synth.add_argument("--spiral-period-s", type=float, default=DEFAULT_EXPERIMENT.spiral_period_s)
    synth.add_argument("--park-arc-fraction", type=float, default=DEFAULT_EXPERIMENT.park_arc_fraction)
    synth.add_argument("--park-radius-m", type=float, default=DEFAULT_EXPERIMENT.park_radius_m)
    synth.add_argument(
        "--park-z-osc-amplitude-m",
        type=float,
        default=DEFAULT_EXPERIMENT.park_z_osc_amplitude_m,
    )
    synth.add_argument(
        "--park-z-osc-period-s",
        type=float,
        default=DEFAULT_EXPERIMENT.park_z_osc_period_s,
    )
    synth.add_argument(
        "--park-r-osc-amplitude-m",
        type=float,
        default=DEFAULT_EXPERIMENT.park_r_osc_amplitude_m,
    )
    synth.add_argument(
        "--park-r-osc-period-s",
        type=float,
        default=DEFAULT_EXPERIMENT.park_r_osc_period_s,
    )
    synth.add_argument(
        "--park-z-osc-phase-rad",
        type=float,
        default=DEFAULT_EXPERIMENT.park_z_osc_phase_rad,
    )
    synth.add_argument(
        "--park-r-osc-phase-rad",
        type=float,
        default=DEFAULT_EXPERIMENT.park_r_osc_phase_rad,
    )
    synth.add_argument("--acc-sigma-ms2", type=float, default=DEFAULT_EXPERIMENT.acc_sigma_ms2)
    synth.add_argument("--seed", type=int, default=DEFAULT_EXPERIMENT.seed)
    synth_default_instrument = INSTRUMENTS["synth_v1"]
    synth.add_argument("--lsb-to-ms2", type=float, default=synth_default_instrument.lsb_to_ms2)
    synth.add_argument("--gyro-lsb-to-rads", type=float, default=synth_default_instrument.gyro_lsb_to_rads)
    synth.add_argument("--mag-lsb-to-ut", type=float, default=synth_default_instrument.mag_lsb_to_uT)

    preprocess = subparsers.add_parser("preprocess", help="Build a cycle file from TRAJ/AUX")
    preprocess.add_argument("--traj", type=Path, required=True)
    preprocess.add_argument("--aux", type=Path, required=True)
    preprocess.add_argument("--window-index", type=int, default=0)
    preprocess.add_argument("--instrument", type=str, default="synth_v1")
    preprocess.add_argument("--outdir", type=Path, required=True)
    preprocess.add_argument("--out", type=Path, default=None)

    integrate = subparsers.add_parser("integrate", help="Reconstruct ENU positions from a cycle file")
    integrate.add_argument("--cycle", type=Path, required=True)
    integrate.add_argument("--outdir", type=Path, required=True)
    integrate.add_argument("--method", type=str, choices=["trap", "rect"], default="trap")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "synth":
        params = replace(
            DEFAULT_EXPERIMENT,
            cycle_hours=args.cycle_hours,
            start_juld=args.start_juld,
            lat0=args.lat0,
            lon0=args.lon0,
            dt_surface_s=args.dt_surface_s,
            dt_descent_s=args.dt_descent_s,
            dt_park_s=args.dt_park_s,
            dt_ascent_s=args.dt_ascent_s,
            transition_seconds=args.transition_seconds,
            surface1_minutes=args.surface1_minutes,
            descent_hours=args.descent_hours,
            park_depth_m=args.park_depth_m,
            park_hours=args.park_hours,
            ascent_hours=args.ascent_hours,
            surface2_minutes=args.surface2_minutes,
            spiral_radius_m=args.spiral_radius_m,
            spiral_period_s=args.spiral_period_s,
            park_arc_fraction=args.park_arc_fraction,
            park_radius_m=args.park_radius_m,
            park_z_osc_amplitude_m=args.park_z_osc_amplitude_m,
            park_z_osc_period_s=args.park_z_osc_period_s,
            park_r_osc_amplitude_m=args.park_r_osc_amplitude_m,
            park_r_osc_period_s=args.park_r_osc_period_s,
            park_z_osc_phase_rad=args.park_z_osc_phase_rad,
            park_r_osc_phase_rad=args.park_r_osc_phase_rad,
            acc_sigma_ms2=args.acc_sigma_ms2,
            seed=args.seed,
        )
        instrument = INSTRUMENTS["synth_v1"]
        if (
            args.lsb_to_ms2 != instrument.lsb_to_ms2
            or args.gyro_lsb_to_rads != instrument.gyro_lsb_to_rads
            or args.mag_lsb_to_ut != instrument.mag_lsb_to_uT
        ):
            from dataclasses import replace as dc_replace

            instrument = dc_replace(
                instrument,
                lsb_to_ms2=args.lsb_to_ms2,
                gyro_lsb_to_rads=args.gyro_lsb_to_rads,
                mag_lsb_to_uT=args.mag_lsb_to_ut,
            )
        generate_synthetic_raw(args.outdir, params, instrument)
        return 0

    if args.command == "preprocess":
        from .run_preprocess import build_cycle_file, derive_base_from_traj_path

        if args.out is None:
            base = derive_base_from_traj_path(args.traj)
            out_path = args.outdir / f"CYCLE_{base}_W{args.window_index:03d}.nc"
        else:
            out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

        build_cycle_file(
            traj_path=args.traj,
            aux_path=args.aux,
            out_path=out_path,
            window_index=args.window_index,
            instrument=args.instrument,
        )
        return 0

    if args.command == "integrate":
        from .run_integrate import integrate_cycle_file

        integrate_cycle_file(
            cycle_path=args.cycle,
            outdir=args.outdir,
            method=args.method,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
