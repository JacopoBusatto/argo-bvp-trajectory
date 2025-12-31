## Synthetic Data Model (Step A)

This synthetic dataset mimics Coriolis TRAJ + AUX content while keeping the
physics simple and configurable. The underwater path follows a spiral + arc
pattern across phases of the cycle.

### Physical model (spiral + arc)

- Surface: short drift at the sea surface with GPS fixes.
- Descent: spiral around a center point down to the park depth.
- Park: constant-depth arc (circular drift) at park depth.
- Ascent: spiral back to the surface, ending near the final surface fix.

### Configurable parameters

- `cycle_hours`: total cycle duration in hours.
- `dt_surface_s`, `dt_descent_s`, `dt_park_s`, `dt_ascent_s`: time step per phase.
- `start_juld`: initial time (relative to `REFERENCE_DATE_TIME`).
- `lat0`, `lon0`: starting GPS fix at the surface.
- `park_depth_m`: target depth for the park phase.
- `descent_rate_m_s`, `ascent_rate_m_s`: vertical speed per phase.
- `spiral_radius_m`, `spiral_turns`: spiral geometry for descent/ascent.
- `arc_radius_m`, `arc_angle_deg`: park arc geometry.
- `transition_seconds`: base half-width of the velocity-blend window at phase
  boundaries (the implementation uses +/- 3*transition_seconds around each switch).
- `park_z_osc_amplitude_m`, `park_z_osc_period_s`, `park_z_osc_phase_rad`: vertical
  oscillation during parking.
- `park_r_osc_amplitude_m`, `park_r_osc_period_s`, `park_r_osc_phase_rad`: radial
  oscillation during parking.
- `noise_gps_m`, `noise_pressure_dbar`, `noise_temp_C`, `noise_psal_psu`: sensor noise levels.
- `noise_acc_count`: IMU acceleration noise in counts.
- `lsb_to_ms2`: conversion factor from acceleration counts to m/s^2.

Notes:
- GPS fixes are only emitted at the surface; underwater `LATITUDE`/`LONGITUDE`
  are NaN by design.

### Parametri: Esperimento vs Strumento

- I parametri di esperimento (scenario sintetico) vivono in
  `src/argo_bvp/synth/experiment_params.py` e sono usati direttamente durante lo
  sviluppo per iterare rapidamente sul modello fisico.
- I parametri di strumento (conversioni IMU) vivono in
  `src/argo_bvp/synth/instrument_params.py` e sono separati per facilitare la
  transizione sintetico → dato reale.
- In futuro si potra' aggiungere un loader YAML/JSON sopra queste dataclass,
  senza toccare il core di generazione.

### Smoothing tra fasi (Level 1)

Per evitare spike non fisici nelle accelerazioni, le velocita' sono raccordate
tra una fase e la successiva usando un blending su una finestra temporale che
si estende per 3*`transition_seconds` prima e dopo lo switch. Il raccordo usa una
smoothstep:

smoothstep(u) = 3u^2 - 2u^3, u=(t-t0)/T

La posizione non viene "spostata" fuori dalla finestra: dentro la finestra
la velocita' viene blendata e poi la posizione viene reintegrata mantenendo
gli stessi punti di inizio/fine della finestra. Non viene introdotta una
nuova fase nel dataset; se serve debug, usare `is_transition`.

Limiti: e' un raccordo cinematico, non un modello fisico; T troppo lungo
smussa anche dinamiche reali, T troppo corto lascia residui.

### Oscillazioni di piccolo scala in parking

Solo durante il parcheggio, la profondita' e il raggio vengono modulati con
oscillazioni sinusoidali senza introdurre nuove fasi. Con u = t - t_park_start:

z(t) = z_base(t) + Az * sin(2*pi*u/Tz + phi_z)

R(t) = R0 + dR * sin(2*pi*u/Tr + phi_r)

La legge di theta(t) sull'arco rimane invariata; la modulazione agisce solo su
R(t) e z(t). Il raccordo di smoothing tra fasi resta attivo ai confini e non
richiede una fase aggiuntiva.

### Esecuzione da terminale

Esempi:

```bash
python -m argo_bvp.cli synth --outdir outputs/synthetic
```

Con parametri espliciti:

```bash
python -m argo_bvp.cli synth --outdir outputs/synthetic --cycle-hours 24 --dt-descent-s 5 --dt-park-s 30 --dt-ascent-s 5 --acc-sigma-ms2 0.002
```
