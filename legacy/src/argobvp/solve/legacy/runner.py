from __future__ import annotations

import argparse
from pathlib import Path

from .solve import solve_bvp_ready, SolveConfig
from ..preprocess.writers import write_netcdf


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="First-pass solver for BVP-ready outputs (horizontal integration).")
    p.add_argument("--bvp-ready", required=True, help="Path to *_bvp_ready.nc")
    p.add_argument("--cycles", default=None, help="Optional path to *_cycles.nc (for surface fixes if missing).")
    p.add_argument("--out", required=True, help="Output NetCDF path (e.g., outputs/solve/<platform>_solved.nc)")
    p.add_argument("--acc-n-var", default="acc_n", help="Acceleration north variable in bvp_ready (default: acc_n).")
    p.add_argument("--acc-e-var", default="acc_e", help="Acceleration east variable in bvp_ready (default: acc_e).")
    return p


def main():
    args = _build_parser().parse_args()
    cfg = SolveConfig(acc_n_var=args.acc_n_var, acc_e_var=args.acc_e_var)
    ds_out = solve_bvp_ready(Path(args.bvp_ready), path_cycles=Path(args.cycles) if args.cycles else None, cfg=cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_netcdf(ds_out, out_path)
    print(f"OK: wrote solved file to {out_path}")
    ds_out.close()


if __name__ == "__main__":
    main()
