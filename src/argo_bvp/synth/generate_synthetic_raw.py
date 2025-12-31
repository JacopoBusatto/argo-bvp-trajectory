"""Pipeline for synthetic TRUTH/TRAJ/AUX generation and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .experiment_params import DEFAULT_EXPERIMENT, ExperimentParams
from .generate_aux import build_aux_from_truth
from .generate_traj import build_traj_from_truth
from .generate_truth import generate_truth_cycle
from ..instruments import INSTRUMENTS, InstrumentParams


def generate_synthetic_raw(
    outdir: str | Path,
    params: ExperimentParams = DEFAULT_EXPERIMENT,
    instrument: InstrumentParams | str = "synth_v1",
) -> dict[str, Path]:
    """Generate TRUTH, TRAJ, AUX and diagnostic plots."""
    if not isinstance(params, ExperimentParams):
        raise TypeError("params must be an ExperimentParams instance")
    inst = _resolve_instrument(instrument)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    truth = generate_truth_cycle(params)
    traj = build_traj_from_truth(truth)
    aux = build_aux_from_truth(truth, inst)

    base = _build_basename(params)

    truth_path = outdir / f"{base}_TRUTH.nc"
    traj_path = outdir / f"{base}_TRAJ.nc"
    aux_path = outdir / f"{base}_AUX.nc"

    truth.to_netcdf(truth_path, engine="h5netcdf")
    traj.to_netcdf(traj_path, engine="h5netcdf")
    aux.to_netcdf(aux_path, engine="h5netcdf")

    plan_path = outdir / f"TRUTH_{base}_plan.png"
    xyz_path = outdir / f"TRUTH_{base}_3d.png"
    acc_path = outdir / f"TRUTH_{base}_acc.png"
    depth_path = outdir / f"TRUTH_{base}_depth.png"

    _plot_plan_view(truth, plan_path)
    _plot_3d_view(truth, xyz_path)
    _plot_acc_series(truth, acc_path)
    _plot_depth_series(truth, depth_path)

    return {
        "truth_nc": truth_path,
        "traj_nc": traj_path,
        "aux_nc": aux_path,
        "plan_png": plan_path,
        "xyz_png": xyz_path,
        "acc_png": acc_path,
        "depth_png": depth_path,
    }


def _resolve_instrument(instrument: InstrumentParams | str) -> InstrumentParams:
    if isinstance(instrument, InstrumentParams):
        return instrument
    if isinstance(instrument, str):
        key = instrument.strip()
        if key in INSTRUMENTS:
            return INSTRUMENTS[key]
        raise KeyError(f"Unknown instrument: {instrument}")
    raise TypeError("instrument must be InstrumentParams or str")


def _build_basename(params: ExperimentParams) -> str:
    cycle = _format_tag(params.cycle_hours, decimals=2)
    dt_descent = _format_tag(params.dt_descent_s, decimals=2)
    dt_park = _format_tag(params.dt_park_s, decimals=2)
    dt_ascent = _format_tag(params.dt_ascent_s, decimals=2)
    noise = _format_tag(params.acc_sigma_ms2, decimals=6)
    return f"SYNTH_CY{cycle}h_d{dt_descent}s_p{dt_park}s_a{dt_ascent}s_n{noise}"


def _format_tag(value: float, decimals: int) -> str:
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    fmt = f"{{:.{decimals}f}}"
    text = fmt.format(value).rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def _plot_plan_view(truth: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    x = np.asarray(truth["x"].values, dtype=float)
    y = np.asarray(truth["y"].values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Plan view (x-y)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_3d_view(truth: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.asarray(truth["x"].values, dtype=float)
    y = np.asarray(truth["y"].values, dtype=float)
    z = np.asarray(truth["z"].values, dtype=float)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="tab:green", linewidth=1.2)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D trajectory (x-y-z)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_acc_series(truth: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    t = np.asarray(truth["t"].values, dtype=float)
    ax_vals = np.asarray(truth["ax"].values, dtype=float)
    ay_vals = np.asarray(truth["ay"].values, dtype=float)
    az_vals = np.asarray(truth["az"].values, dtype=float)

    fig, axes = plt.subplots(figsize=(7.2, 6.0), nrows=2, sharex=False)
    ax = axes[0]
    ax.plot(t, ax_vals, label="ax", linewidth=1.0)
    ax.plot(t, ay_vals, label="ay", linewidth=1.0)
    ax.plot(t, az_vals, label="az", linewidth=1.0)
    _plot_transition_markers(truth, t, ax, ax_vals)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("acc (m/s^2)")
    ax.set_title("Acceleration time series")
    ax.legend(loc="best")

    ax_zoom = axes[1]
    park_mask = np.asarray(truth["phase"].values) == "park"
    if np.any(park_mask):
        t_park = t[park_mask]
        ax_zoom.plot(t_park, ax_vals[park_mask], label="ax", linewidth=1.0)
        ax_zoom.plot(t_park, ay_vals[park_mask], label="ay", linewidth=1.0)
        ax_zoom.plot(t_park, az_vals[park_mask], label="az", linewidth=1.0)
        _plot_transition_markers(truth, t, ax_zoom, ax_vals, mask=park_mask)
    ax_zoom.set_xlabel("t (s)")
    ax_zoom.set_ylabel("acc (m/s^2)")
    ax_zoom.set_title("Parking zoom")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_depth_series(truth: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    t = np.asarray(truth["t"].values, dtype=float)
    z = np.asarray(truth["z"].values, dtype=float)

    fig, axes = plt.subplots(figsize=(7.2, 5.2), nrows=2, sharex=False)
    ax = axes[0]
    ax.plot(t, z, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z (m)")
    ax.set_title("Depth time series")

    ax_zoom = axes[1]
    park_mask = np.asarray(truth["phase"].values) == "park"
    if np.any(park_mask):
        ax_zoom.plot(t[park_mask], z[park_mask], color="tab:blue", linewidth=1.2)
    ax_zoom.set_xlabel("t (s)")
    ax_zoom.set_ylabel("z (m)")
    ax_zoom.set_title("Parking zoom")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_transition_markers(
    truth: xr.Dataset,
    t: np.ndarray,
    ax,
    series: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    if "is_transition" not in truth:
        return
    transition = np.asarray(truth["is_transition"].values, dtype=int) == 1
    if mask is not None:
        transition = transition & mask
    if np.any(transition):
        ax.scatter(
            t[transition],
            series[transition],
            s=14,
            facecolors="none",
            edgecolors="0.2",
            label="transition",
        )


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
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", compressed) + _chunk(b"IEND", b"")
    outpath.write_bytes(png)


__all__ = ["generate_synthetic_raw"]
