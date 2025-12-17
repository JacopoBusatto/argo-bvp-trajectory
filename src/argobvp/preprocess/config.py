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
class PreprocessConfig:
    platform: str
    paths: PathsConfig
    imu: IMUConfig = IMUConfig()

    # segmentation / helpers
    pres_surface_max: float = 5.0  # for optional diagnostics
    time_reference: str = "1950-01-01T00:00:00Z"  # Argo JULD epoch


def load_config(path: str | Path) -> PreprocessConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # very small “manual” parsing to keep dependencies minimal
    paths = PathsConfig(**raw["paths"])
    imu_raw = raw.get("imu", {})

    accel = AccelCalib(**imu_raw.get("accel", {}))
    gyro = GyroCalib(**imu_raw.get("gyro", {}))
    mag = MagCalib(**imu_raw.get("mag", {}))
    imu = IMUConfig(
        frame=imu_raw.get("frame", "NED"),
        g=float(imu_raw.get("g", 9.80665)),
        accel=accel,
        gyro=gyro,
        mag=mag,
    )

    return PreprocessConfig(
        platform=str(raw["platform"]),
        paths=paths,
        imu=imu,
        pres_surface_max=float(raw.get("pres_surface_max", 5.0)),
        time_reference=str(raw.get("time_reference", "1950-01-01T00:00:00Z")),
    )
