"""
Lightweight solver utilities to integrate IMU-derived accelerations over attendible phases.
"""

from .solve import solve_bvp_ready, SolveConfig
from .runner import main as solve_main

__all__ = ["solve_bvp_ready", "SolveConfig", "solve_main"]
