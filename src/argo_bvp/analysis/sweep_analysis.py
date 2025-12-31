"""Analysis utilities for sweep outputs.

Metrics are computed in the local ENU frame defined by cycle lat0/lon0.
Key points:
  - dive: start anchor (last GPS in surface1).
  - emerge: end anchor (first GPS in surface2).
  - park_start/park_end: first/last sample where phase == 3 (park).

Metrics:
  - rms_underwater_m: RMS distance between REC and TRUTH on underwater samples.
  - err_park_start_m / err_park_end_m: distance at park start/end.
  - err_delta_dive_to_parkstart_m: |dist(rec_park_start - dive) - dist(truth_park_start - dive)|.
  - err_delta_parkend_to_emerge_m: |dist(emerge - rec_park_end) - dist(emerge - truth_park_end)|.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from ..integrate.park_xy import latlon_to_enu_m
from ..io.cycle_io import read_cycle_netcdf


@dataclass(frozen=True)
class Run:
    path: Path
    cycle_path: Path
    rec_path: Path
    truth_path: Path
    cycle_hours: float
    dt_descent_s: float
    dt_park_s: float
    dt_ascent_s: float
    acc_sigma_ms2: float
    window_index: int
    tag: str


_RUN_RE = re.compile(
    r"^(?P<base>SYNTH_CY(?P<cy>[0-9p]+)h_d(?P<d>[0-9p]+)s_p(?P<p>[0-9p]+)s_a(?P<a>[0-9p]+)s_n(?P<n>[0-9p]+))(?:_W(?P<w>\d{3}))?$"
)


def discover_sweep_runs(outdir: Path) -> list[Run]:
    """Scan sweep directory and return runs with cycle/rec/truth files."""
    outdir = Path(outdir)
    candidates = [outdir] + [p for p in outdir.iterdir() if p.is_dir()]
    runs: list[Run] = []

    for folder in candidates:
        cycle_path = _pick_single(folder.glob("CYCLE_*.nc"))
        rec_path = _pick_single(folder.glob("REC_*.nc"))
        truth_path = _pick_single(folder.glob("*_TRUTH.nc"))
        if cycle_path is None or rec_path is None or truth_path is None:
            if folder != outdir:
                warnings.warn(f"Skipping {folder}: missing CYCLE/REC/TRUTH", stacklevel=2)
            continue

        params = _parse_run_name(folder.name)
        if params is None:
            params = _parse_run_name(cycle_path.stem)
        if params is None:
            params = _parse_run_name(truth_path.stem)
        if params is None:
            warnings.warn(f"Skipping {folder}: cannot parse run name", stacklevel=2)
            continue

        run = Run(
            path=folder,
            cycle_path=cycle_path,
            rec_path=rec_path,
            truth_path=truth_path,
            cycle_hours=params["cycle_hours"],
            dt_descent_s=params["dt_descent_s"],
            dt_park_s=params["dt_park_s"],
            dt_ascent_s=params["dt_ascent_s"],
            acc_sigma_ms2=params["acc_sigma_ms2"],
            window_index=params["window_index"],
            tag=params["tag"],
        )
        runs.append(run)

    return runs


def compute_metrics_for_run(run: Run) -> dict[str, float | str | int]:
    """Compute accuracy metrics between REC and TRUTH for one run."""
    ds_cycle = read_cycle_netcdf(run.cycle_path)
    ds_rec = xr.open_dataset(run.rec_path, engine="h5netcdf").load()
    ds_truth = xr.open_dataset(run.truth_path, engine="h5netcdf").load()

    lat0 = float(ds_cycle["lat0"].values)
    lon0 = float(ds_cycle["lon0"].values)

    x_rec = np.asarray(ds_rec["x_enu"].values, dtype=float)
    y_rec = np.asarray(ds_rec["y_enu"].values, dtype=float)

    if "t" in ds_rec.coords:
        t_rec = np.asarray(ds_rec.coords["t"].values, dtype=float)
    else:
        t_rec = np.asarray(ds_rec["t"].values, dtype=float)

    if "underwater_mask" in ds_rec:
        underwater_mask = np.asarray(ds_rec["underwater_mask"].values, dtype=bool)
    else:
        phase = np.asarray(ds_cycle["phase"].values)
        underwater_mask = phase != 1

    x_true, y_true, t_truth_shifted = _truth_enu_on_cycle_time(ds_truth, ds_cycle, lat0, lon0)

    x_true_i = np.interp(t_rec, t_truth_shifted, x_true)
    y_true_i = np.interp(t_rec, t_truth_shifted, y_true)

    mask = underwater_mask & np.isfinite(x_rec) & np.isfinite(y_rec)
    if not np.any(mask):
        raise ValueError("No underwater samples available for RMS computation")

    dist2 = (x_rec - x_true_i) ** 2 + (y_rec - y_true_i) ** 2
    rms_underwater_m = float(np.sqrt(np.nanmean(dist2[mask])))

    phase = np.asarray(ds_cycle["phase"].values)
    park_mask = phase == 3
    err_park_start_m = np.nan
    err_park_end_m = np.nan
    err_delta_dive_to_parkstart_m = np.nan
    err_delta_parkend_to_emerge_m = np.nan

    anchor_lat = np.asarray(ds_cycle["anchor_lat"].values, dtype=float)
    anchor_lon = np.asarray(ds_cycle["anchor_lon"].values, dtype=float)
    dive_x, dive_y = latlon_to_enu_m(anchor_lat[0], anchor_lon[0], lat0, lon0)
    emerge_x, emerge_y = latlon_to_enu_m(anchor_lat[1], anchor_lon[1], lat0, lon0)

    if np.any(park_mask):
        idx = np.where(park_mask)[0]
        park_start_idx = int(idx[0])
        park_end_idx = int(idx[-1])

        park_start_time = t_rec[park_start_idx]
        park_end_time = t_rec[park_end_idx]

        x_true_park_start = float(np.interp(park_start_time, t_truth_shifted, x_true))
        y_true_park_start = float(np.interp(park_start_time, t_truth_shifted, y_true))
        x_true_park_end = float(np.interp(park_end_time, t_truth_shifted, x_true))
        y_true_park_end = float(np.interp(park_end_time, t_truth_shifted, y_true))

        err_park_start_m = float(
            _distance(x_rec[park_start_idx], y_rec[park_start_idx], x_true_park_start, y_true_park_start)
        )
        err_park_end_m = float(
            _distance(x_rec[park_end_idx], y_rec[park_end_idx], x_true_park_end, y_true_park_end)
        )

        delta_rec_dive_to_parkstart = _distance(x_rec[park_start_idx], y_rec[park_start_idx], dive_x, dive_y)
        delta_rec_parkend_to_emerge = _distance(emerge_x, emerge_y, x_rec[park_end_idx], y_rec[park_end_idx])
        delta_truth_dive_to_parkstart = _distance(x_true_park_start, y_true_park_start, dive_x, dive_y)
        delta_truth_parkend_to_emerge = _distance(emerge_x, emerge_y, x_true_park_end, y_true_park_end)

        err_delta_dive_to_parkstart_m = float(
            abs(delta_rec_dive_to_parkstart - delta_truth_dive_to_parkstart)
        )
        err_delta_parkend_to_emerge_m = float(
            abs(delta_rec_parkend_to_emerge - delta_truth_parkend_to_emerge)
        )

    return {
        "tag": run.tag,
        "window_index": run.window_index,
        "cycle_hours": run.cycle_hours,
        "dt_descent_s": run.dt_descent_s,
        "dt_park_s": run.dt_park_s,
        "dt_ascent_s": run.dt_ascent_s,
        "acc_sigma_ms2": run.acc_sigma_ms2,
        "rms_underwater_m": rms_underwater_m,
        "err_park_start_m": err_park_start_m,
        "err_park_end_m": err_park_end_m,
        "err_delta_dive_to_parkstart_m": err_delta_dive_to_parkstart_m,
        "err_delta_parkend_to_emerge_m": err_delta_parkend_to_emerge_m,
    }


def build_metrics_table(outdir: Path) -> pd.DataFrame:
    """Compute metrics for all runs and write metrics.csv."""
    runs = discover_sweep_runs(Path(outdir))
    rows = [compute_metrics_for_run(run) for run in runs]
    df = pd.DataFrame(rows)

    analysis_dir = Path(outdir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    return df


def plot_heatmaps(df: pd.DataFrame, outdir: Path) -> None:
    """Plot heatmaps for each noise sigma with shared color scales.

    Axes:
      - x: dt_descent_s (== dt_ascent_s)
      - y: dt_park_s
    """
    plt = _maybe_import_matplotlib()
    outdir = Path(outdir) / "analysis" / "heatmaps"
    outdir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        _write_placeholder_png(outdir / "heatmaps_unavailable.png")
        return
    from matplotlib.colors import LogNorm

    df_filtered = df[np.isclose(df["dt_descent_s"], df["dt_ascent_s"])]
    if df_filtered.empty:
        warnings.warn("No runs with dt_descent_s == dt_ascent_s for heatmaps", stacklevel=2)
        return

    metrics = [
        "rms_underwater_m",
        "err_park_start_m",
        "err_park_end_m",
        "err_delta_dive_to_parkstart_m",
        "err_delta_parkend_to_emerge_m",
    ]

    sigma_values = sorted(df_filtered["acc_sigma_ms2"].unique())
    dt_descent_values = sorted(df_filtered["dt_descent_s"].unique())
    dt_park_values = sorted(df_filtered["dt_park_s"].unique())

    floor = 1e-2
    for metric in metrics:
        metric_values = df_filtered[metric].to_numpy(dtype=float)
        finite_mask = np.isfinite(metric_values)
        if not np.any(finite_mask):
            continue
        positive_mask = finite_mask & (metric_values > 0.0)
        if not np.any(positive_mask):
            warnings.warn(f"No positive values for {metric}; skipping heatmaps.", stacklevel=2)
            continue
        vmin = float(np.nanmin(metric_values[positive_mask]))
        vmax = float(np.nanmax(metric_values))
        vmin = max(vmin, floor)
        if vmax <= vmin:
            warnings.warn(f"Invalid LogNorm bounds for {metric}; skipping heatmaps.", stacklevel=2)
            continue

        for sigma in sigma_values:
            df_sigma = df_filtered[df_filtered["acc_sigma_ms2"] == sigma]
            grid = _build_metric_grid(df_sigma, dt_descent_values, dt_park_values, metric)
            grid = np.ma.masked_less_equal(grid, 0.0)
            fig, ax = plt.subplots(figsize=(7.2, 5.6))
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                cmap="viridis",
            )
            ax.set_xticks(range(len(dt_descent_values)))
            ax.set_xticklabels([_format_tag(x, decimals=2) for x in dt_descent_values])
            ax.set_yticks(range(len(dt_park_values)))
            ax.set_yticklabels([_format_tag(y, decimals=2) for y in dt_park_values])
            ax.set_xlabel("dt_descent_s (== dt_ascent_s)")
            ax.set_ylabel("dt_park_s")
            ax.set_title(f"{metric} (sigma={_format_tag(sigma, decimals=6)})")
            fig.colorbar(im, ax=ax, label="Error [m] (log scale)")
            fig.tight_layout()

            sigma_tag = _format_tag(sigma, decimals=6)
            outpath = outdir / f"heatmap_{metric}_n{sigma_tag}.png"
            fig.savefig(outpath, dpi=150)
            plt.close(fig)


def plot_trajectories_by_freq(
    df: pd.DataFrame,
    runs: list[Run],
    outdir: Path,
) -> None:
    """Plot REC vs TRUTH trajectories by sampling frequencies.

    Filters:
      - dt_descent_s == dt_ascent_s
      - dt_park_s >= dt_ascent_s
    """
    plt = _maybe_import_matplotlib()
    outdir = Path(outdir) / "analysis" / "trajectories"
    outdir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        _write_placeholder_png(outdir / "trajectories_unavailable.png")
        return

    df_filtered = df[np.isclose(df["dt_descent_s"], df["dt_ascent_s"])]
    df_filtered = df_filtered[df_filtered["dt_park_s"] >= df_filtered["dt_ascent_s"]]
    if df_filtered.empty:
        warnings.warn("No runs available for trajectory plots after filters", stacklevel=2)
        return

    combos = _select_frequency_combos(df_filtered)
    if not combos:
        warnings.warn("No frequency combinations selected for trajectory plots", stacklevel=2)
        return

    run_lookup = _build_run_lookup(runs)
    sigma_values = sorted(df_filtered["acc_sigma_ms2"].unique())

    for dt_descent, dt_park in combos:
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        truth_drawn = False
        marker_done = False

        for sigma in sigma_values:
            run = _find_run(
                run_lookup,
                dt_descent_s=dt_descent,
                dt_park_s=dt_park,
                dt_ascent_s=dt_descent,
                acc_sigma_ms2=sigma,
            )
            if run is None:
                continue

            ds_cycle = read_cycle_netcdf(run.cycle_path)
            ds_rec = xr.open_dataset(run.rec_path, engine="h5netcdf").load()
            ds_truth = xr.open_dataset(run.truth_path, engine="h5netcdf").load()

            lat0 = float(ds_cycle["lat0"].values)
            lon0 = float(ds_cycle["lon0"].values)

            if not truth_drawn:
                x_true, y_true, _ = _truth_enu_on_cycle_time(ds_truth, ds_cycle, lat0, lon0)
                ax.plot(x_true, y_true, color="black", linewidth=1.3, label="truth")
                truth_drawn = True

            x_rec = np.asarray(ds_rec["x_enu"].values, dtype=float)
            y_rec = np.asarray(ds_rec["y_enu"].values, dtype=float)
            label = f"n{_format_tag(sigma, decimals=6)}"
            ax.plot(x_rec, y_rec, linewidth=1.1, label=label)

            if not marker_done:
                phase = np.asarray(ds_cycle["phase"].values)
                park_idx = np.where(phase == 3)[0]
                if park_idx.size:
                    if "t" in ds_rec.coords:
                        t_rec = np.asarray(ds_rec.coords["t"].values, dtype=float)
                    else:
                        t_rec = np.asarray(ds_rec["t"].values, dtype=float)
                    t_start = t_rec[int(park_idx[0])]
                    t_end = t_rec[int(park_idx[-1])]
                    x_true, y_true, t_truth_shifted = _truth_enu_on_cycle_time(
                        ds_truth, ds_cycle, lat0, lon0
                    )
                    x_start = float(np.interp(t_start, t_truth_shifted, x_true))
                    y_start = float(np.interp(t_start, t_truth_shifted, y_true))
                    x_end = float(np.interp(t_end, t_truth_shifted, x_true))
                    y_end = float(np.interp(t_end, t_truth_shifted, y_true))
                    ax.scatter(
                        [x_start, x_end],
                        [y_start, y_end],
                        s=28,
                        facecolors="none",
                        edgecolors="0.2",
                        label="park start/end",
                    )
                    marker_done = True

        if not truth_drawn:
            plt.close(fig)
            continue

        ax.set_xlabel("x_enu (m)")
        ax.set_ylabel("y_enu (m)")
        ax.set_title(f"Trajectories d={_format_tag(dt_descent, 2)}s p={_format_tag(dt_park, 2)}s")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        fig.tight_layout()

        outpath = outdir / f"traj_d{_format_tag(dt_descent, 2)}s_p{_format_tag(dt_park, 2)}s.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)


def _truth_enu_on_cycle_time(
    truth: xr.Dataset,
    cycle: xr.Dataset,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "t" not in truth:
        raise KeyError("TRUTH dataset missing t")
    if "lat" not in truth or "lon" not in truth:
        raise KeyError("TRUTH dataset missing lat/lon")
    if "start_juld" not in truth.attrs:
        raise KeyError("TRUTH dataset missing start_juld attribute")

    t_truth = np.asarray(truth["t"].values, dtype=float)
    x_true, y_true = latlon_to_enu_m(
        np.asarray(truth["lat"].values, dtype=float),
        np.asarray(truth["lon"].values, dtype=float),
        lat0,
        lon0,
    )

    start_juld = float(truth.attrs["start_juld"])
    anchor_start_juld = float(np.asarray(cycle["anchor_juld"].values, dtype=float)[0])
    shift_seconds = (anchor_start_juld - start_juld) * 86400.0
    t_truth_shifted = t_truth - shift_seconds

    return x_true, y_true, t_truth_shifted


def _parse_run_name(name: str) -> dict[str, float | int | str] | None:
    stem = name
    for prefix in ("CYCLE_", "REC_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    match = _RUN_RE.match(stem)
    if not match:
        return None

    cycle_hours = _parse_tag_number(match.group("cy"))
    dt_descent_s = _parse_tag_number(match.group("d"))
    dt_park_s = _parse_tag_number(match.group("p"))
    dt_ascent_s = _parse_tag_number(match.group("a"))
    acc_sigma_ms2 = _parse_tag_number(match.group("n"))
    window_index = int(match.group("w") or 0)
    tag = match.group("base")
    return {
        "cycle_hours": cycle_hours,
        "dt_descent_s": dt_descent_s,
        "dt_park_s": dt_park_s,
        "dt_ascent_s": dt_ascent_s,
        "acc_sigma_ms2": acc_sigma_ms2,
        "window_index": window_index,
        "tag": tag,
    }


def _parse_tag_number(text: str) -> float:
    return float(text.replace("p", "."))


def _pick_single(items: Iterable[Path]) -> Path | None:
    paths = list(items)
    if len(paths) != 1:
        return None
    return paths[0]


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _build_metric_grid(
    df: pd.DataFrame,
    dt_descent_values: list[float],
    dt_park_values: list[float],
    metric: str,
) -> np.ndarray:
    grid = np.full((len(dt_park_values), len(dt_descent_values)), np.nan, dtype=float)
    for i, dt_park in enumerate(dt_park_values):
        for j, dt_descent in enumerate(dt_descent_values):
            mask = (df["dt_park_s"] == dt_park) & (df["dt_descent_s"] == dt_descent)
            if np.any(mask):
                grid[i, j] = float(df.loc[mask, metric].iloc[0])
    return grid


def _select_frequency_combos(df: pd.DataFrame) -> list[tuple[float, float]]:
    dt_descent_values = sorted(df["dt_descent_s"].unique())
    dt_park_values = sorted(df["dt_park_s"].unique())
    preferred_descent = [5.0, 10.0, 60.0, 120.0]
    preferred_park = [10.0, 60.0, 120.0, 300.0]

    combos: list[tuple[float, float]] = []
    for d in preferred_descent:
        if not any(np.isclose(d, val) for val in dt_descent_values):
            continue
        for p in preferred_park:
            if not any(np.isclose(p, val) for val in dt_park_values):
                continue
            if p >= d:
                combos.append((d, p))

    if combos:
        return combos[:12]

    candidates = [(d, p) for d in dt_descent_values for p in dt_park_values if p >= d]
    if not candidates:
        return []

    descent_sel = _select_representative(dt_descent_values)
    park_sel = _select_representative(dt_park_values)
    for d in descent_sel:
        for p in park_sel:
            if p >= d:
                combos.append((d, p))
    return combos[:12]


def _select_representative(values: list[float]) -> list[float]:
    if not values:
        return []
    values_sorted = sorted(values)
    if len(values_sorted) <= 3:
        return values_sorted
    return [values_sorted[0], values_sorted[len(values_sorted) // 2], values_sorted[-1]]


def _build_run_lookup(runs: list[Run]) -> dict[tuple[float, float, float, float], Run]:
    lookup: dict[tuple[float, float, float, float], Run] = {}
    for run in runs:
        key = (run.dt_descent_s, run.dt_park_s, run.dt_ascent_s, run.acc_sigma_ms2)
        lookup[key] = run
    return lookup


def _find_run(
    lookup: dict[tuple[float, float, float, float], Run],
    dt_descent_s: float,
    dt_park_s: float,
    dt_ascent_s: float,
    acc_sigma_ms2: float,
) -> Run | None:
    for key, run in lookup.items():
        if (
            np.isclose(key[0], dt_descent_s)
            and np.isclose(key[1], dt_park_s)
            and np.isclose(key[2], dt_ascent_s)
            and np.isclose(key[3], acc_sigma_ms2)
        ):
            return run
    return None


def _format_tag(value: float, decimals: int) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    fmt = f"{{:.{decimals}f}}"
    text = fmt.format(value).rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _write_placeholder_png(outpath: Path, width: int = 640, height: int = 480) -> None:
    import struct
    import zlib

    rgb = bytes((255, 255, 255))
    row = rgb * width
    raw = b"".join(b"\x00" + row for _ in range(height))
    compressed = zlib.compress(raw, level=6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    outpath.write_bytes(png)


__all__ = [
    "Run",
    "discover_sweep_runs",
    "compute_metrics_for_run",
    "build_metrics_table",
    "plot_heatmaps",
    "plot_trajectories_by_freq",
]
