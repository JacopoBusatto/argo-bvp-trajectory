"""Convenience entrypoint for integration outputs from a cycle file."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import xarray as xr

from .integrate.park_xy import reconstruct_cycle_xy_from_ds
from .io.cycle_io import read_cycle_netcdf


_CYCLE_RE = re.compile(r"^CYCLE_(?P<base>.+)_W(?P<window>\d{3})$")


def parse_cycle_base_window(cycle_path: Path) -> tuple[str, int]:
    """Parse BASE and window index from a cycle filename."""
    stem = Path(cycle_path).stem
    match = _CYCLE_RE.match(stem)
    if match:
        base = match.group("base")
        window = int(match.group("window"))
        return base, window
    return stem, 0


def integrate_cycle_file(
    cycle_path: str | Path,
    outdir: str | Path,
    method: str = "trap",
) -> dict[str, Path]:
    """Integrate ENU positions for a cycle and write outputs."""
    cycle_path = Path(cycle_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base, window_index = parse_cycle_base_window(cycle_path)

    ds_cycle = read_cycle_netcdf(cycle_path)
    ds_rec = reconstruct_cycle_xy_from_ds(ds_cycle, method=method)

    rec_path = outdir / f"REC_{base}_W{window_index:03d}.nc"
    ds_rec.to_netcdf(rec_path, engine="h5netcdf")

    plan_path = outdir / f"{base}_W{window_index:03d}_plan.png"
    xyz_path = outdir / f"{base}_W{window_index:03d}_3d.png"
    acc_path = outdir / f"{base}_W{window_index:03d}_acc.png"

    _plot_plan_view(ds_rec, plan_path)
    _plot_3d_view(ds_rec, xyz_path)
    _plot_acc_series(ds_rec, acc_path)

    return {
        "rec_nc": rec_path,
        "plan_png": plan_path,
        "xyz_png": xyz_path,
        "acc_png": acc_path,
    }


def _plot_plan_view(ds: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return

    x = np.asarray(ds["x_enu"].values, dtype=float)
    y = np.asarray(ds["y_enu"].values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, color="tab:blue", linewidth=1.4)
    ax.set_xlabel("x_enu (m)")
    ax.set_ylabel("y_enu (m)")
    ax.set_title("Reconstructed plan view (ENU)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_3d_view(ds: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.asarray(ds["x_enu"].values, dtype=float)
    y = np.asarray(ds["y_enu"].values, dtype=float)
    if "pres" in ds:
        z = -np.asarray(ds["pres"].values, dtype=float)
    else:
        z = np.zeros_like(x, dtype=float)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="tab:green", linewidth=1.1)
    ax.set_xlabel("x_enu (m)")
    ax.set_ylabel("y_enu (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Reconstructed 3D (ENU + depth)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _plot_acc_series(ds: xr.Dataset, outpath: Path) -> None:
    plt = _maybe_import_matplotlib()
    if plt is None:
        _write_placeholder_png(outpath)
        return
    if "t" in ds.coords:
        t = np.asarray(ds.coords["t"].values, dtype=float)
    else:
        t = np.asarray(ds["t"].values, dtype=float)
    lin_acc = np.asarray(ds["lin_acc"].values, dtype=float)
    ax_vals = lin_acc[:, 0]
    ay_vals = lin_acc[:, 1]
    az_vals = lin_acc[:, 2]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(t, ax_vals, label="ax", linewidth=1.0)
    ax.plot(t, ay_vals, label="ay", linewidth=1.0)
    ax.plot(t, az_vals, label="az", linewidth=1.0)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("acc (m/s^2)")
    ax.set_title("Acceleration time series")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


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
    raw = b"".join(b"\\x00" + row for _ in range(height))
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
        b"\\x89PNG\\r\\n\\x1a\\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    outpath.write_bytes(png)


__all__ = ["parse_cycle_base_window", "integrate_cycle_file"]
