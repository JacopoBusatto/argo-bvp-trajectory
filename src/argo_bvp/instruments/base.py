"""Instrument parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentParams:
    lsb_to_ms2: float
    gyro_lsb_to_rads: float
    mag_lsb_to_uT: float


__all__ = ["InstrumentParams"]
