"""Command-line interface for argo-bvp utilities."""

from __future__ import annotations

import argparse
from itertools import product
from dataclasses import replace
from pathlib import Path

import numpy as np

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

    sweep = subparsers.add_parser("sweep", help="Run sensitivity sweep: synth -> preprocess -> integrate")
    sweep.add_argument("--outdir", type=Path, default=Path("outputs/sweep"))
    sweep.add_argument("--dt-descent-s-list", type=str, required=True)
    sweep.add_argument("--dt-park-s-list", type=str, required=True)
    sweep.add_argument("--dt-ascent-s-list", type=str, required=True)
    sweep.add_argument("--acc-sigma-ms2-list", type=str, required=True)
    sweep.add_argument("--park-hours-list", type=str, required=True)
    sweep.add_argument("--seed", type=int, default=DEFAULT_EXPERIMENT.seed)
    sweep.add_argument("--instrument", type=str, default="synth_v1")
    sweep.add_argument("--window-index", type=int, default=0)
    sweep.add_argument("--method", type=str, choices=["trap", "rect"], default="trap")

    analyze = subparsers.add_parser("analyze-sweep", help="Analyze sweep outputs")
    analyze.add_argument("--outdir", type=Path, default=Path("outputs/sweep"))

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

    if args.command == "sweep":
        from .run_integrate import integrate_cycle_file
        from .run_preprocess import build_cycle_file, derive_base_from_traj_path

        if args.instrument not in INSTRUMENTS:
            raise KeyError(f"Unknown instrument: {args.instrument}")
        instrument = INSTRUMENTS[args.instrument]

        dt_descent_list = _parse_float_list(args.dt_descent_s_list)
        dt_park_list = _parse_float_list(args.dt_park_s_list)
        dt_ascent_list = _parse_float_list(args.dt_ascent_s_list)
        acc_sigma_list = _parse_float_list(args.acc_sigma_ms2_list)
        park_hours_list = _parse_float_list(args.park_hours_list)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        completed = 0
        for run_index, (dt_desc, dt_park, dt_asc, acc_sigma, park_hours) in enumerate(
            product(
                dt_descent_list,
                dt_park_list,
                dt_ascent_list,
                acc_sigma_list,
                park_hours_list,
            )
        ):
            cycle_hours = _compute_cycle_hours(park_hours)
            params = replace(
                DEFAULT_EXPERIMENT,
                cycle_hours=cycle_hours,
                dt_descent_s=dt_desc,
                dt_park_s=dt_park,
                dt_ascent_s=dt_asc,
                acc_sigma_ms2=acc_sigma,
                park_hours=park_hours,
                seed=args.seed + run_index,
            )
            tag = _build_synth_tag(params)
            exp_outdir = outdir / tag
            exp_outdir.mkdir(parents=True, exist_ok=True)

            generate_synthetic_raw(exp_outdir, params, instrument)

            traj_paths = sorted(exp_outdir.glob("*_TRAJ.nc"))
            aux_paths = sorted(exp_outdir.glob("*_AUX.nc"))
            if len(traj_paths) != 1 or len(aux_paths) != 1:
                raise FileNotFoundError("Expected exactly one *_TRAJ.nc and one *_AUX.nc")
            traj_path = traj_paths[0]
            aux_path = aux_paths[0]

            base = derive_base_from_traj_path(traj_path)
            cycle_path = exp_outdir / f"CYCLE_{base}_W{args.window_index:03d}.nc"

            build_cycle_file(
                traj_path=traj_path,
                aux_path=aux_path,
                out_path=cycle_path,
                window_index=args.window_index,
                instrument=args.instrument,
            )
            integrate_cycle_file(cycle_path, exp_outdir, method=args.method)
            completed += 1

        print(f"Sweep completed: {completed} experiment(s) in {outdir}")
        return 0

    if args.command == "analyze-sweep":
        from .analysis.sweep_analysis import (
            build_metrics_table,
            discover_sweep_runs,
            plot_heatmaps,
            plot_trajectories_by_freq,
        )

        runs = discover_sweep_runs(args.outdir)
        df = build_metrics_table(args.outdir)
        plot_heatmaps(df, args.outdir)
        plot_trajectories_by_freq(df, runs, args.outdir)
        _print_metric_summary(df)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 1


def _parse_float_list(values: str) -> list[float]:
    items = [item.strip() for item in values.split(",") if item.strip()]
    if not items:
        raise ValueError("List must contain at least one value")
    return [float(item) for item in items]


def _compute_cycle_hours(park_hours: float) -> float:
    surface_hours = (DEFAULT_EXPERIMENT.surface1_minutes + DEFAULT_EXPERIMENT.surface2_minutes) / 60.0
    return surface_hours + DEFAULT_EXPERIMENT.descent_hours + float(park_hours) + DEFAULT_EXPERIMENT.ascent_hours


def _build_synth_tag(params: object) -> str:
    cycle = _format_tag(params.cycle_hours, decimals=2)
    dt_descent = _format_tag(params.dt_descent_s, decimals=2)
    dt_park = _format_tag(params.dt_park_s, decimals=2)
    dt_ascent = _format_tag(params.dt_ascent_s, decimals=2)
    noise = _format_tag(params.acc_sigma_ms2, decimals=6)
    return f"SYNTH_CY{cycle}h_d{dt_descent}s_p{dt_park}s_a{dt_ascent}s_n{noise}"


def _format_tag(value: float, decimals: int) -> str:
    if value == round(value):
        return str(int(round(value)))
    fmt = f"{{:.{decimals}f}}"
    text = fmt.format(value).rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def _print_metric_summary(df) -> None:
    if df.empty:
        print("No sweep runs found for summary.")
        return
    df_valid = df[np.isclose(df["dt_descent_s"], df["dt_ascent_s"])]
    if df_valid.empty:
        df_valid = df

    metrics = ["rms_underwater_m", "err_park_start_m"]
    for metric in metrics:
        df_metric = df_valid[np.isfinite(df_valid[metric])]
        if df_metric.empty:
            continue
        best = df_metric.nsmallest(5, metric)
        worst = df_metric.nlargest(5, metric)
        print(f"Top 5 best by {metric}:")
        for _, row in best.iterrows():
            print(f"  {row['tag']} W{int(row['window_index']):03d} -> {row[metric]:.3f} m")
        print(f"Top 5 worst by {metric}:")
        for _, row in worst.iterrows():
            print(f"  {row['tag']} W{int(row['window_index']):03d} -> {row[metric]:.3f} m")


if __name__ == "__main__":
    raise SystemExit(main())
