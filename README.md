## argo-bvp-trajectory

Reconstruct submerged trajectories of ARGO floats by integrating IMU accelerations
and solving a boundary value problem (BVP) constrained by GPS fixes at the start
and end of each cycle.

### Status

Milestone 1: cycle file schema v1, writer/validator, and tests.

### Quickstart

```bash
python -m pip install -e ".[dev]"
pytest -q
```

### Repository layout

- `src/argo_bvp/`: core package (schema, I/O, future readers)
- `tests/`: pytest suite
- `docs/`: project documentation
