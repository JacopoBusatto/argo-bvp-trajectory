from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Any

import yaml


Frame = Literal["NED", "ENU"]


@dataclass(frozen=True)
class AccelCalib:
    scale_g: float = 4.0
    denom: float = 65536.0
    bias_counts: Dict[str, float] = None  # {"x":..., "y":..., "z":...}
    gain: Dict[str, float] = None         # {"x":..., "y":..., "z":...}
    axis_map: Dict[str, str] = None       # {"x":"X","y":"Y","z":"Z"} meaning output axis <- source axis
    sign: Dict[str, int] = None           # {"x":+1,"y":+1,"z":+1}

    def __post_init__(self):
        object.__setattr__(self, "bias_counts", self.bias_counts or {"x": 0.0, "y": 0.0, "z": 0.0})
        object.__setattr__(self, "gain", self.gain or {"x": 1.0, "y": 1.0, "z": 1.0})
        object.__setattr__(self, "axis_map", self.axis_map or {"x": "X", "y": "Y", "z": "Z"})
        object.__setattr__(self, "sign", self.sign or {"x": +1, "y": +1, "z": +1})


@dataclass(frozen=True)
class GyroCalib:
    scale: float = 1.0
    units: Literal["rad/s", "deg/s"] = "rad/s"
    bias_counts: Dict[str, float] = None

    def __post_init__(self):
        object.__setattr__(self, "bias_counts", self.bias_counts or {"x": 0.0, "y": 0.0, "z": 0.0})


@dataclass(frozen=True)
class MagCalib:
    hard_iron: Dict[str, float] = None  # {"x":..., "y":..., "z":...}
    soft_iron_xy: Dict[str, float] = None  # {"xx":..., "xy":..., "yx":..., "yy":...}

    def __post_init__(self):
        object.__setattr__(self, "hard_iron", self.hard_iron or {"x": 0.0, "y": 0.0, "z": 0.0})
        object.__setattr__(
            self,
            "soft_iron_xy",
            self.soft_iron_xy or {"xx": 1.0, "xy": 0.0, "yx": 0.0, "yy": 1.0},
        )


@dataclass(frozen=True)
class IMUConfig:
    frame: Frame = "NED"
    g: float = 9.80665
    accel: AccelCalib = AccelCalib()
    gyro: GyroCalib = GyroCalib()
    mag: MagCalib = MagCalib()


@dataclass(frozen=True)
class PathsConfig:
    traj: str
    aux: str

@dataclass(frozen=True)
class AttitudeConfig:
    mode: Literal["safe_tilt_only", "complementary"] = "safe_tilt_only"
    dt_max: float = 300.0
    alpha: float = 0.98

@dataclass(frozen=True)
class PreprocessConfig:
    platform: str
    paths: PathsConfig
    imu: IMUConfig = IMUConfig()

    pres_surface_max: float = 5.0
    min_parking_samples_for_bvp: int = 10
    min_phase_samples_for_bvp: int = 10
    time_reference: str = "1950-01-01T00:00:00Z"

    attitude: AttitudeConfig = AttitudeConfig()



def load_config(path: str | Path) -> PreprocessConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    def lower_keys(d: Any) -> Any:
        if isinstance(d, dict):
            return {str(k).lower(): lower_keys(v) for k, v in d.items()}
        return d

    raw_l = lower_keys(raw)

    # paths
    if "paths" not in raw_l:
        raise ValueError("Missing required key: paths")
    paths = PathsConfig(**raw_l["paths"])

    # imu block (case-insensitive)
    imu_raw = raw_l.get("imu", {})

    # gyro block: allow both imu.gyro and top-level gyro
    gyro_raw = imu_raw.get("gyro", raw_l.get("gyro", {}))
    accel_raw = imu_raw.get("accel", raw_l.get("accel", {}))
    mag_raw = imu_raw.get("mag", raw_l.get("mag", {}))

    # Validate gyro.scale presence (avoid silent default=1.0)
    if "scale" not in gyro_raw:
        raise ValueError(
            "gyro.scale missing in YAML. "
            "Expected location: imu: gyro: scale: <float> (or top-level gyro: scale: <float>)."
        )

    accel = AccelCalib(**accel_raw)
    gyro = GyroCalib(
        scale=float(gyro_raw.get("scale")),
        units=str(gyro_raw.get("units", "rad/s")),
        bias_counts=gyro_raw.get("bias_counts", None),
    )
    mag = MagCalib(**mag_raw)

    imu = IMUConfig(
        frame=str(imu_raw.get("frame", "NED")).upper(),
        g=float(imu_raw.get("g", 9.80665)),
        accel=accel,
        gyro=gyro,
        mag=mag,
    )

    att_raw = raw_l.get("attitude", {})
    att = AttitudeConfig(
        mode=str(att_raw.get("mode", "safe_tilt_only")),
        dt_max=float(att_raw.get("dt_max", 300.0)),
        alpha=float(att_raw.get("alpha", 0.98)),
    )

    return PreprocessConfig(
        platform=str(raw_l["platform"]),
        paths=paths,
        imu=imu,
        pres_surface_max=float(raw_l.get("pres_surface_max", 5.0)),
        min_parking_samples_for_bvp=int(raw_l.get("min_parking_samples_for_bvp", 10)),
        min_phase_samples_for_bvp=int(raw_l.get("min_phase_samples_for_bvp", 10)),
        time_reference=str(raw_l.get("time_reference", "1950-01-01T00:00:00Z")),
        attitude=att,
    )
