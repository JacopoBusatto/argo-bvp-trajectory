"""Instrument registry."""

from __future__ import annotations

from .base import InstrumentParams


INSTRUMENTS: dict[str, InstrumentParams] = {
    "synth_v1": InstrumentParams(
        lsb_to_ms2=1e-7,
        gyro_lsb_to_rads=1.0,
        mag_lsb_to_uT=1.0,
    ),
    "synth_v2": InstrumentParams(
        lsb_to_ms2=6e-5,
        gyro_lsb_to_rads=1.0,
        mag_lsb_to_uT=1.0,
    )
}


__all__ = ["INSTRUMENTS"]
