"""Generate synthetic ground-truth trajectories (TRUTH only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .experiment_params import DEFAULT_EXPERIMENT, ExperimentParams


def generate_truth_cycle(
    params: ExperimentParams = DEFAULT_EXPERIMENT,
) -> xr.Dataset:
    """Generate a synthetic TRUTH dataset for a single cycle."""
    if not isinstance(params, ExperimentParams):
        raise TypeError("params must be an ExperimentParams instance")

    lat0 = float(params.lat0)
    lon0 = float(params.lon0)
    start_juld = float(params.start_juld)
    park_depth_m = float(params.park_depth_m)
    surface1_duration_s = float(params.surface1_minutes) * 60.0
    surface2_duration_s = float(params.surface2_minutes) * 60.0
    park_duration_s = float(params.park_hours) * 3600.0
    dt_surface_s = float(params.dt_surface_s)
    dt_descent_s = float(params.dt_descent_s)
    dt_park_s = float(params.dt_park_s)
    dt_ascent_s = float(params.dt_ascent_s)
    spiral_radius_m = float(params.spiral_radius_m)
    spiral_period_s = float(params.spiral_period_s)
    park_arc_radius_m = float(params.park_radius_m)
    park_arc_fraction = float(params.park_arc_fraction)
    park_z_osc_amplitude_m = float(params.park_z_osc_amplitude_m)
    park_z_osc_period_s = float(params.park_z_osc_period_s)
    park_r_osc_amplitude_m = float(params.park_r_osc_amplitude_m)
    park_r_osc_period_s = float(params.park_r_osc_period_s)
    park_z_osc_phase_rad = float(params.park_z_osc_phase_rad)
    park_r_osc_phase_rad = float(params.park_r_osc_phase_rad)
    noise_accel_sigma_ms2 = float(params.acc_sigma_ms2)
    noise_seed = int(params.seed)
    transition_seconds = float(params.transition_seconds)

    if park_depth_m <= 0:
        raise ValueError("park_depth_m must be positive")
    if params.descent_hours <= 0 or params.ascent_hours <= 0:
        raise ValueError("descent_hours and ascent_hours must be positive")

    descent_duration_s = float(params.descent_hours) * 3600.0
    ascent_duration_s = float(params.ascent_hours) * 3600.0
    descent_rate_m_s = park_depth_m / descent_duration_s
    ascent_rate_m_s = park_depth_m / ascent_duration_s

    t_surface1 = _make_time_vector(surface1_duration_s, dt_surface_s)
    t_descent = _make_time_vector(descent_duration_s, dt_descent_s)
    t_park = _make_time_vector(park_duration_s, dt_park_s)
    t_ascent = _make_time_vector(ascent_duration_s, dt_ascent_s)
    t_surface2 = _make_time_vector(surface2_duration_s, dt_surface_s)

    t_segments: list[np.ndarray] = []
    x_segments: list[np.ndarray] = []
    y_segments: list[np.ndarray] = []
    z_segments: list[np.ndarray] = []
    vx_segments: list[np.ndarray] = []
    vy_segments: list[np.ndarray] = []
    vz_segments: list[np.ndarray] = []
    phase_segments: list[np.ndarray] = []

    current_time = 0.0

    def _append_phase(
        local_t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        phase_label: str,
    ) -> None:
        nonlocal current_time
        if t_segments:
            if local_t.size <= 1:
                return
            local_t = local_t[1:]
            x = x[1:]
            y = y[1:]
            z = z[1:]
            vx = vx[1:]
            vy = vy[1:]
            vz = vz[1:]
        if local_t.size == 0:
            return
        t_global = current_time + local_t
        current_time = float(t_global[-1])
        t_segments.append(t_global)
        x_segments.append(x)
        y_segments.append(y)
        z_segments.append(z)
        vx_segments.append(vx)
        vy_segments.append(vy)
        vz_segments.append(vz)
        phase_segments.append(np.full(local_t.shape, phase_label, dtype=object))

    x0 = spiral_radius_m
    y0 = 0.0
    z0 = 0.0

    x_s1 = np.full(t_surface1.shape, x0, dtype=float)
    y_s1 = np.full(t_surface1.shape, y0, dtype=float)
    z_s1 = np.full(t_surface1.shape, z0, dtype=float)
    v_s1 = np.zeros(t_surface1.shape, dtype=float)
    _append_phase(t_surface1, x_s1, y_s1, z_s1, v_s1, v_s1, v_s1, "surface")

    x_desc, y_desc, vx_desc, vy_desc = _spiral_xy(
        t_descent,
        start_pos=(x_s1[-1], y_s1[-1]),
        radius=spiral_radius_m,
        period=spiral_period_s,
    )
    z_desc = z0 - descent_rate_m_s * t_descent
    vz_desc = np.full(t_descent.shape, -descent_rate_m_s, dtype=float)
    _append_phase(t_descent, x_desc, y_desc, z_desc, vx_desc, vy_desc, vz_desc, "descent")

    arc_angle = 2.0 * np.pi * park_arc_fraction
    park_start_x = x_desc[-1]
    park_start_y = y_desc[-1]
    park_center_x, park_center_y = _arc_center(park_start_x, park_start_y, park_arc_radius_m)
    (
        x_park,
        y_park,
        vx_park,
        vy_park,
    ) = _arc_xy_with_radius_osc(
        t_park,
        center=(park_center_x, park_center_y),
        base_radius=park_arc_radius_m,
        arc_angle=arc_angle,
        r_osc_amp=park_r_osc_amplitude_m,
        r_osc_period=park_r_osc_period_s,
        r_osc_phase=park_r_osc_phase_rad,
    )
    z_park, vz_park = _park_z_with_osc(
        t_park,
        base_depth=-park_depth_m,
        osc_amp=park_z_osc_amplitude_m,
        osc_period=park_z_osc_period_s,
        osc_phase=park_z_osc_phase_rad,
    )
    _append_phase(t_park, x_park, y_park, z_park, vx_park, vy_park, vz_park, "park")

    x_asc, y_asc, vx_asc, vy_asc = _spiral_xy(
        t_ascent,
        start_pos=(x_park[-1], y_park[-1]),
        radius=spiral_radius_m,
        period=spiral_period_s,
    )
    z_asc = -park_depth_m + ascent_rate_m_s * t_ascent
    vz_asc = np.full(t_ascent.shape, ascent_rate_m_s, dtype=float)
    _append_phase(t_ascent, x_asc, y_asc, z_asc, vx_asc, vy_asc, vz_asc, "ascent")

    x_s2 = np.full(t_surface2.shape, x_asc[-1], dtype=float)
    y_s2 = np.full(t_surface2.shape, y_asc[-1], dtype=float)
    z_s2 = np.zeros(t_surface2.shape, dtype=float)
    v_s2 = np.zeros(t_surface2.shape, dtype=float)
    _append_phase(t_surface2, x_s2, y_s2, z_s2, v_s2, v_s2, v_s2, "surface")

    t = np.concatenate(t_segments)
    x = np.concatenate(x_segments)
    y = np.concatenate(y_segments)
    z = np.concatenate(z_segments)
    vx = np.concatenate(vx_segments)
    vy = np.concatenate(vy_segments)
    vz = np.concatenate(vz_segments)
    phase = np.concatenate(phase_segments)

    x, y, z, vx, vy, vz, is_transition = _apply_transition_smoothing(
        t,
        x,
        y,
        z,
        vx,
        vy,
        vz,
        phase,
        transition_seconds,
    )

    ax = _compute_acceleration_from_velocity(t, vx)
    ay = _compute_acceleration_from_velocity(t, vy)
    az = _compute_acceleration_from_velocity(t, vz)

    if noise_accel_sigma_ms2 > 0.0:
        rng = np.random.default_rng(noise_seed)
        ax = ax + rng.normal(0.0, noise_accel_sigma_ms2, size=ax.shape)
        ay = ay + rng.normal(0.0, noise_accel_sigma_ms2, size=ay.shape)
        az = az + rng.normal(0.0, noise_accel_sigma_ms2, size=az.shape)

    pres = np.maximum(-z, 0.0)
    lat, lon = _enu_to_latlon(x, y, lat0, lon0)

    anchor_labels = np.array(["start_gps", "end_gps"], dtype="U9")
    anchor_idx = np.array([0, int(t.size - 1)], dtype=int)
    anchor_lat = np.array([lat[0], lat[-1]], dtype=float)
    anchor_lon = np.array([lon[0], lon[-1]], dtype=float)
    anchor_juld = start_juld + np.array([t[0], t[-1]]) / 86400.0

    ds = xr.Dataset(
        data_vars={
            "t": ("obs", t, {"units": "s"}),
            "x": ("obs", x, {"units": "m"}),
            "y": ("obs", y, {"units": "m"}),
            "z": ("obs", z, {"units": "m"}),
            "ax": ("obs", ax, {"units": "m s-2"}),
            "ay": ("obs", ay, {"units": "m s-2"}),
            "az": ("obs", az, {"units": "m s-2"}),
            "pres": ("obs", pres, {"units": "dbar"}),
            "phase": ("obs", phase),
            "lat": ("obs", lat, {"units": "degree_north"}),
            "lon": ("obs", lon, {"units": "degree_east"}),
            "is_transition": ("obs", is_transition),
            "anchor_idx": ("anchor", anchor_idx),
            "anchor_lat": ("anchor", anchor_lat, {"units": "degree_north"}),
            "anchor_lon": ("anchor", anchor_lon, {"units": "degree_east"}),
            "anchor_juld": ("anchor", anchor_juld, {"units": "days"}),
        },
        coords={
            "obs": np.arange(t.size, dtype=int),
            "anchor": anchor_labels,
        },
        attrs={
            "lat0": lat0,
            "lon0": lon0,
            "start_juld": start_juld,
            "park_depth_m": park_depth_m,
            "park_center_x": park_center_x,
            "park_center_y": park_center_y,
        },
    )
    return ds


def save_truth(truth_ds: xr.Dataset, outpath: str | Path) -> None:
    """Save a TRUTH dataset to NetCDF."""
    path = Path(outpath)
    truth_ds.to_netcdf(path)


def _make_time_vector(duration_s: float, dt_s: float) -> np.ndarray:
    if duration_s <= 0:
        return np.array([0.0], dtype=float)
    if dt_s <= 0:
        raise ValueError("dt must be positive")
    times = np.arange(0.0, duration_s + 1e-9, dt_s, dtype=float)
    if times.size == 0:
        times = np.array([0.0], dtype=float)
    if times.size:
        if np.isclose(times[-1], duration_s, rtol=0.0, atol=1e-6):
            times[-1] = duration_s
        elif times[-1] < duration_s:
            times = np.append(times, duration_s)
    return times


def _spiral_xy(
    t: np.ndarray,
    start_pos: tuple[float, float],
    radius: float,
    period: float,
    start_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if period <= 0:
        raise ValueError("period must be positive")
    omega = 2.0 * np.pi / period
    start_x, start_y = start_pos
    center_x = start_x - radius * np.cos(start_angle)
    center_y = start_y - radius * np.sin(start_angle)
    angle = start_angle + omega * t
    x = center_x + radius * np.cos(angle)
    y = center_y + radius * np.sin(angle)
    vx = -radius * omega * np.sin(angle)
    vy = radius * omega * np.cos(angle)
    return x, y, vx, vy


def _arc_center(start_x: float, start_y: float, radius: float) -> tuple[float, float]:
    return start_x - radius, start_y


def _arc_xy_with_radius_osc(
    t: np.ndarray,
    center: tuple[float, float],
    base_radius: float,
    arc_angle: float,
    r_osc_amp: float,
    r_osc_period: float,
    r_osc_phase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_x, center_y = center
    if t.size <= 1 or t[-1] <= 0:
        theta = np.zeros_like(t, dtype=float)
        theta_dot = 0.0
    else:
        theta = (t / t[-1]) * arc_angle
        theta_dot = arc_angle / t[-1]

    r_osc = _sin_oscillation(t, r_osc_amp, r_osc_period, r_osc_phase)
    if r_osc.size:
        r_osc = r_osc - r_osc[0]
    radius = base_radius + r_osc

    rdot = _sin_oscillation_derivative(t, r_osc_amp, r_osc_period, r_osc_phase)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    vx = rdot * np.cos(theta) - radius * theta_dot * np.sin(theta)
    vy = rdot * np.sin(theta) + radius * theta_dot * np.cos(theta)
    return x, y, vx, vy


def _park_z_with_osc(
    t: np.ndarray,
    base_depth: float,
    osc_amp: float,
    osc_period: float,
    osc_phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_osc = _sin_oscillation(t, osc_amp, osc_period, osc_phase)
    if z_osc.size:
        z_osc = z_osc - z_osc[0]
    z = np.full(t.shape, base_depth, dtype=float) + z_osc
    vz = _sin_oscillation_derivative(t, osc_amp, osc_period, osc_phase)
    return z, vz


def _apply_transition_smoothing(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    phase: np.ndarray,
    transition_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = t.size
    is_transition = np.zeros((n,), dtype="int8")
    if transition_seconds <= 0 or n < 3:
        return x, y, z, vx, vy, vz, is_transition

    if not (x.size == y.size == z.size == vx.size == vy.size == vz.size == n):
        raise ValueError("Position/velocity arrays must match t length")

    segments = _phase_segments(phase)
    if len(segments) < 2:
        return x, y, z, vx, vy, vz, is_transition

    v_x = np.asarray(vx, dtype=float)
    v_y = np.asarray(vy, dtype=float)
    v_z = np.asarray(vz, dtype=float)

    half = transition_seconds * 3.0
    for idx in range(len(segments) - 1):
        _, old_start, old_end = segments[idx]
        _, new_start, new_end = segments[idx + 1]

        if old_end - old_start < 2 or new_end - new_start < 2:
            continue

        t_boundary = t[new_start]
        start_t = max(t[old_start], t_boundary - half)
        end_t = min(t[new_end - 1], t_boundary + half)
        if end_t <= start_t:
            continue

        window_idx = np.where((t >= start_t) & (t <= end_t))[0]
        if window_idx.size < 3:
            continue
        if np.any(is_transition[window_idx]):
            continue

        is_transition[window_idx] = 1
        t_window = t[window_idx]
        u = (t_window - start_t) / (end_t - start_t)
        s = _smoothstep(u)
        s_prime = _smoothstep_derivative(u, end_t - start_t)

        x, v_x = _blend_window(
            x,
            v_x,
            t,
            t_window,
            window_idx,
            old_start,
            old_end,
            new_start,
            new_end,
            s,
            s_prime,
        )
        y, v_y = _blend_window(
            y,
            v_y,
            t,
            t_window,
            window_idx,
            old_start,
            old_end,
            new_start,
            new_end,
            s,
            s_prime,
        )
        z, v_z = _blend_window(
            z,
            v_z,
            t,
            t_window,
            window_idx,
            old_start,
            old_end,
            new_start,
            new_end,
            s,
            s_prime,
        )

    return x, y, z, v_x, v_y, v_z, is_transition


def _blend_window(
    pos: np.ndarray,
    vel: np.ndarray,
    t: np.ndarray,
    t_window: np.ndarray,
    window_idx: np.ndarray,
    old_start: int,
    old_end: int,
    new_start: int,
    new_end: int,
    s: np.ndarray,
    s_prime: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v_old = np.interp(t_window, t[old_start:old_end], vel[old_start:old_end])
    v_new = np.interp(t_window, t[new_start:new_end], vel[new_start:new_end])
    v_blend = (1.0 - s) * v_old + s * v_new

    x0 = pos[window_idx[0]]
    x_window = _integrate_velocity(t_window, v_blend, x0)
    diff_end = x_window[-1] - pos[window_idx[-1]]
    x_window = x_window - diff_end * s
    v_corr = v_blend - diff_end * s_prime

    pos[window_idx] = x_window
    vel[window_idx] = v_corr
    return pos, vel


def _integrate_velocity(t: np.ndarray, v: np.ndarray, x0: float) -> np.ndarray:
    if t.size <= 1:
        return np.array([x0], dtype=float)
    dt = np.diff(t)
    increments = 0.5 * (v[:-1] + v[1:]) * dt
    return x0 + np.concatenate(([0.0], np.cumsum(increments)))


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smoothstep_derivative(u: np.ndarray, duration: float) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if duration <= 0:
        return np.zeros_like(u)
    return (6.0 * u * (1.0 - u)) / duration


def _compute_acceleration_from_velocity(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    if t.size < 3:
        return np.zeros_like(v)
    return np.gradient(v, t, edge_order=2)


def _sin_oscillation(
    t: np.ndarray,
    amplitude: float,
    period: float,
    phase: float,
) -> np.ndarray:
    if amplitude == 0.0 or period <= 0.0:
        return np.zeros_like(t, dtype=float)
    omega = 2.0 * np.pi / period
    return amplitude * np.sin(omega * t + phase)


def _sin_oscillation_derivative(
    t: np.ndarray,
    amplitude: float,
    period: float,
    phase: float,
) -> np.ndarray:
    if amplitude == 0.0 or period <= 0.0:
        return np.zeros_like(t, dtype=float)
    omega = 2.0 * np.pi / period
    return amplitude * omega * np.cos(omega * t + phase)


def _phase_segments(phase: np.ndarray) -> list[tuple[str, int, int]]:
    labels = []
    start = 0
    for idx in range(1, phase.size):
        if phase[idx] != phase[idx - 1]:
            labels.append((str(phase[start]), start, idx))
            start = idx
    labels.append((str(phase[start]), start, phase.size))
    return labels


def _enu_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    lat0: float,
    lon0: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = 6371000.0
    lat0_rad = np.deg2rad(lat0)
    cos_lat = np.cos(lat0_rad)
    if np.isclose(cos_lat, 0.0):
        cos_lat = 1e-12
    lat = lat0 + np.rad2deg(y / radius)
    lon = lon0 + np.rad2deg(x / (radius * cos_lat))
    return lat, lon


__all__ = ["generate_truth_cycle", "save_truth"]
