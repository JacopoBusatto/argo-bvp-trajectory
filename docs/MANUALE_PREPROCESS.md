MANUALE D’USO — PREPROCESS ARGO-BVP
==================================

Questo documento descrive **come usare** e **come interpretare**
i moduli di preprocess presenti in `src/argobvp/preprocess/`.

L’obiettivo del preprocess è trasformare i file NetCDF Coriolis
(AUX + TRAJ) in prodotti pronti per:
- integrazione cinematica delle traiettorie
- imposizione di vincoli di posizione (BVP)
- analisi per ciclo e per fase


------------------------------------------------------------
1. OBIETTIVO GENERALE
------------------------------------------------------------

Dato un float Argo dotato di IMU:

- leggere i dati IMU (accelerometri, giroscopi, magnetometro) da AUX
- calibrare e ruotare le accelerazioni in un frame cartesiano (NED)
- organizzare i dati in:
  - un dataset continuo (time series)
  - prodotti per ciclo
  - prodotti per segmento/fase
- associare a ogni ciclo un vincolo di posizione in superficie
  usando i fix disponibili nel file TRAJ

Il preprocess **NON integra traiettorie**:
produce solo i dati e i keypoints necessari
per l’integrazione e il BVP a valle.


------------------------------------------------------------
2. STRUTTURA DEI MODULI
------------------------------------------------------------

Tutti i moduli descritti qui si trovano in:

    src/argobvp/preprocess/


------------------------------------------------------------
2.1 config.py
------------------------------------------------------------

Responsabilità:
- leggere il file YAML di configurazione (`configs/*.yml`)
- validare e rendere disponibili:
  - path ai file AUX e TRAJ
  - parametri di calibrazione IMU
  - opzioni di attitude estimation
  - soglia minima di campioni di parking per considerare un ciclo utilizzabile dal BVP (`min_parking_samples_for_bvp`, default=10)

Uso:
- usato automaticamente dal runner
- normalmente NON si importa a mano


------------------------------------------------------------
2.2 io_coriolis.py
------------------------------------------------------------

Responsabilità:
- apertura dei NetCDF Coriolis via xarray
- funzioni:
  - open_aux(path)
  - open_traj(path)

Note:
- i dataset vengono aperti con `decode_times=True`
- nessuna trasformazione scientifica qui, solo I/O


------------------------------------------------------------
2.3 imu_calib.py
------------------------------------------------------------

Responsabilità:
- conversione COUNTS → unità fisiche

Implementa:
- accelerometri:
    counts → g → m/s²
    usando bias, gain, scale
- giroscopi:
    counts → rad/s (scala attualmente provvisoria)
- magnetometro:
    hard-iron e soft-iron (se configurati)

Output:
- canali fisici, ancora nel frame sensore


------------------------------------------------------------
2.4 attitude.py
------------------------------------------------------------

Responsabilità:
- stima dell’orientazione del sensore

Modalità attuale:
- "safe_tilt_only"
  - roll/pitch da accelerometro (low-pass)
  - yaw da magnetometro tilt-compensato
  - robusto anche con gyro non affidabile

Output:
- angoli di assetto
- matrici di rotazione
- accelerazioni ruotate nel frame NED

Nota:
- la qualità dello yaw è limitata finché il gyro non è calibrato


------------------------------------------------------------
2.5 products.py
------------------------------------------------------------

Responsabilità:
- orchestrare la costruzione del dataset continuo

Input:
- ds_aux
- configurazione

Output:
- ds_cont (dataset continuo preprocessato)

Contiene:
- time (datetime64)
- cycle_number
- pressione
- accelerazioni calibrate
- accelerazioni ruotate (NED)
- assetto stimato


------------------------------------------------------------
2.6 cycles.py
------------------------------------------------------------

Responsabilità:
- derivare prodotti discreti dal dataset continuo

Produce:
- ds_cycles:
    una riga per ciclo, con keypoints temporali:
    - t_park_start
    - t_profile_deepest
    - t_ascent_start
    - t_surface_start
    - t_surface_end
    - pressioni rappresentative
- ds_segments:
    segmentazione temporale per ciclo:
    - park_drift
    - ascent
    - (altri se riconosciuti)

Nota importante:
- le fasi riconosciute dipendono da ciò che è presente
  nel dataset (es. MEASUREMENT_CODE)
- l’assenza di una fase NON implica che fisicamente non esista


------------------------------------------------------------
2.7 surface_fixes.py
------------------------------------------------------------

Responsabilità:
- associare a ogni ciclo una posizione di superficie
  usando i fix del file TRAJ

Strategia:
- target temporale: t_surface_end (da AUX)
- si cercano fix TRAJ in lat/lon nel tempo
- regole:
    - interpolazione solo se i fix prima e dopo sono vicini nel tempo
    - altrimenti si usa il fix più vicino
    - altrimenti il vincolo è mancante

Output aggiunto a ds_cycles:
- lat_surface_end
- lon_surface_end
- pos_source: nearest | interp | missing
- t_pos_used: tempo del fix usato
- pos_age_s: differenza temporale rispetto a t_surface_end
- diagnostica (dt_before_s, dt_after_s, gap_s, ...)

Nel dataset attuale:
- i fix avvengono ~20 minuti dopo t_surface_end
- quindi la maggior parte dei cicli usa pos_source = nearest


------------------------------------------------------------
2.8 writers.py
------------------------------------------------------------

Responsabilità:
- scrittura dei prodotti su disco

Formati:
- NetCDF (.nc)
- Parquet (.parquet)

Directory di output:
- outputs/preprocess/

Nota:
- outputs/ è ignorata da git


------------------------------------------------------------
2.9 runner.py
------------------------------------------------------------

È l’entry-point principale.

Esegue in sequenza:
1) apertura AUX
2) costruzione ds_cont
3) costruzione ds_cycles e ds_segments
4) apertura TRAJ
5) aggiunta dei surface fixes
6) scrittura degli output

Uso tipico:

    python -m argobvp.preprocess.runner \
        --config configs/4903848.yml \
        --out outputs/preprocess


------------------------------------------------------------
2.10 bvp_ready.py
------------------------------------------------------------

Responsabilita:
- costruisce un dataset "BVP-ready" con solo le finestre di parking
- include solo i cicli valid_for_bvp == True
- espone time, accelerazioni orizzontali, pressione come z proxy, e il vincolo di superficie post-parking

Uso tipico:

    python -m argobvp.preprocess.bvp_ready \
        --cont outputs/preprocess/<platform>_preprocessed_imu.nc \
        --cycles outputs/preprocess/<platform>_cycles.nc \
        --segments outputs/preprocess/<platform>_segments.nc \
        --out outputs/preprocess


------------------------------------------------------------
3. SCRIPT DI DEBUG
------------------------------------------------------------

debug_phases.py
----------------
- ispeziona fasi e segmenti
- stampa quali fasi compaiono e in che ordine
- utile per capire cosa viene riconosciuto dal dataset

debug_surface_fixes.py
----------------------
- riassume come sono stati associati i fix di superficie
- mostra:
    - nearest vs interp
    - pos_age_s
    - tempi dei fix usati

check_traj_surface_fix.py
-------------------------
- diagnostica rapida del file TRAJ
- sampling temporale delle posizioni
- distanza temporale tra fix e finestre di superficie


------------------------------------------------------------
4. COSA NON FA (ANCORA)
------------------------------------------------------------

Il preprocess NON:
- integra traiettorie
- risolve il BVP
- corregge il drift in superficie
- ricostruisce traiettorie durante fasi senza IMU

Queste operazioni sono delegate ai moduli di integrazione
(`integrators.py`, `bvp.py`) e a sviluppi successivi.


------------------------------------------------------------
5. STATO ATTUALE
------------------------------------------------------------

✔ pipeline preprocess funzionante end-to-end  
✔ prodotti per ciclo e segmento  
✔ vincoli di posizione esplicitamente tracciati  
✔ output puliti e non versionati  

Prossimi sviluppi:
- controllo di allineamento temporale IMU ↔ keypoints
- ricostruzione esplicita delle fasi mancanti
- integrazione e BVP per singolo ciclo
