from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main():
    bvp_path = Path("outputs/synthetic/preprocess/SYNTHETIC01_bvp_ready.nc")
    cycles_path = Path("outputs/synthetic/preprocess/SYNTHETIC01_cycles.nc")
    out_path = Path("outputs/solve/SYNTHETIC01_solved.nc")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "argobvp.solve.runner",
        "--bvp-ready",
        str(bvp_path),
        "--cycles",
        str(cycles_path),
        "--out",
        str(out_path),
    ]
    print("[cmd]", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    if res.returncode != 0:
        raise SystemExit(res.returncode)


if __name__ == "__main__":
    main()
