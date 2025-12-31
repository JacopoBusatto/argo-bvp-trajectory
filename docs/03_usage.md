# CLI Usage

This project provides a single CLI entrypoint:

```bash
python -m argo_bvp.cli --help
```

Below are the current subcommands and their real options.

## 1) `synth`

**Purpose**  
Generate synthetic TRUTH/TRAJ/AUX NetCDF files plus diagnostic plots.

**Example command**
```bash
python -m argo_bvp.cli synth \
  --outdir outputs/synthetic \
  --cycle-hours 24 \
  --dt-descent-s 5 \
  --dt-park-s 10 \
  --dt-ascent-s 5 \
  --acc-sigma-ms2 0.002 \
  --seed 0
```

**Inputs**  
No input files. Parameters are provided via CLI.

**Outputs**
- `SYNTH_<tag>_TRUTH.nc`
- `SYNTH_<tag>_TRAJ.nc`
- `SYNTH_<tag>_AUX.nc`
- `SYNTH_<tag>_{plan,3d,acc,depth}.png`

**Options**
```text
--outdir PATH
--cycle-hours FLOAT
--start-juld FLOAT
--lat0 FLOAT
--lon0 FLOAT
--dt-surface-s FLOAT
--dt-descent-s FLOAT
--dt-park-s FLOAT
--dt-ascent-s FLOAT
--transition-seconds FLOAT
--surface1-minutes FLOAT
--descent-hours FLOAT
--park-depth-m FLOAT
--park-hours FLOAT
--ascent-hours FLOAT
--surface2-minutes FLOAT
--spiral-radius-m FLOAT
--spiral-period-s FLOAT
--park-arc-fraction FLOAT
--park-radius-m FLOAT
--park-z-osc-amplitude-m FLOAT
--park-z-osc-period-s FLOAT
--park-r-osc-amplitude-m FLOAT
--park-r-osc-period-s FLOAT
--park-z-osc-phase-rad FLOAT
--park-r-osc-phase-rad FLOAT
--acc-sigma-ms2 FLOAT
--seed INT
--lsb-to-ms2 FLOAT
--gyro-lsb-to-rads FLOAT
--mag-lsb-to-ut FLOAT
```

## 2) `preprocess`

**Purpose**  
Build a cycle NetCDF file from TRAJ/AUX inputs and choose GPS anchors.

**Example command**
```bash
python -m argo_bvp.cli preprocess \
  --traj outputs/synthetic/SYNTH_CY24h_d5s_p10s_a5s_n0_TRAJ.nc \
  --aux  outputs/synthetic/SYNTH_CY24h_d5s_p10s_a5s_n0_AUX.nc \
  --window-index 0 \
  --instrument synth_v1 \
  --outdir outputs/preprocess
```

**Inputs**
- TRAJ NetCDF (Coriolis-like)
- AUX NetCDF (Coriolis-like)

**Outputs**
- `CYCLE_<tag>_W{window:03d}.nc`

**Options**
```text
--traj PATH
--aux PATH
--window-index INT
--instrument NAME
--outdir PATH
--out PATH
```

NOTE (IT): se `--out` non è specificato, il nome è derivato dal TRAJ come
`CYCLE_<tag>_W{window:03d}.nc`.

## 3) `integrate`

**Purpose**  
Reconstruct ENU positions from a cycle file using the Fubini BVP integrator.

**Example command**
```bash
python -m argo_bvp.cli integrate \
  --cycle outputs/preprocess/CYCLE_SYNTH_CY24h_d5s_p10s_a5s_n0_W000.nc \
  --outdir outputs/integrate
```

**Inputs**
- Cycle NetCDF (`CYCLE_*.nc`)

**Outputs**
- `REC_<tag>_W{window:03d}.nc`
- `<tag>_W{window:03d}_{plan,3d,acc}.png`

**Options**
```text
--cycle PATH
--outdir PATH
--method trap|rect
```

## 4) `sweep`

**Purpose**  
Run a grid of synthetic experiments and, for each case, execute:
synth → preprocess → integrate.

**Example command**
```bash
python -m argo_bvp.cli sweep \
  --outdir outputs/sweep \
  --dt-descent-s-list 5,10,20 \
  --dt-park-s-list 10,30,60 \
  --dt-ascent-s-list 5,10,20 \
  --acc-sigma-ms2-list 0,0.002,0.01 \
  --park-hours-list 8,12,20 \
  --seed 0 \
  --instrument synth_v1 \
  --window-index 0 \
  --method trap
```

**Outputs**
Each experiment goes to its own subfolder:

```
outputs/sweep/<tag>/
  SYNTH_<tag>_TRUTH.nc
  SYNTH_<tag>_TRAJ.nc
  SYNTH_<tag>_AUX.nc
  CYCLE_<tag>_W000.nc
  REC_<tag>_W000.nc
  <tag>_W000_{plan,3d,acc}.png
```

**Options**
```text
--outdir PATH
--dt-descent-s-list LIST
--dt-park-s-list LIST
--dt-ascent-s-list LIST
--acc-sigma-ms2-list LIST
--park-hours-list LIST
--seed INT
--instrument NAME
--window-index INT
--method trap|rect
```

## 5) `analyze-sweep`

**Purpose**  
Compute metrics vs TRUTH, generate heatmaps and trajectory plots.

**Example command**
```bash
python -m argo_bvp.cli analyze-sweep --outdir outputs/sweep
```

**Outputs**
```
outputs/sweep/analysis/metrics.csv
outputs/sweep/analysis/heatmaps/*.png
outputs/sweep/analysis/trajectories/*.png
```

**Options**
```text
--outdir PATH
```
