from __future__ import annotations
import numpy as np

Array = np.ndarray


def nearest_index(t: Array, t_query: float) -> int:
    t = np.asarray(t, dtype=float)
    return int(np.argmin(np.abs(t - float(t_query))))


def endpoint_error(r: Array, r_target: Array, ord: int = 2) -> float:
    r = np.asarray(r, dtype=float)
    r_target = np.asarray(r_target, dtype=float).reshape(-1)
    return float(np.linalg.norm(r[-1] - r_target, ord=ord))


def point_error_at_time(t: Array, r: Array, t_query: float, r_target: Array, ord: int = 2) -> float:
    idx = nearest_index(t, t_query)
    r_target = np.asarray(r_target, dtype=float).reshape(-1)
    return float(np.linalg.norm(r[idx] - r_target, ord=ord))
