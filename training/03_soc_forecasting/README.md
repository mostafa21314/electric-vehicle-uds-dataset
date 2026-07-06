# 03 — SoC Forecasting (LSTM-Transformer)

First model-training stage of the pipeline. Trains a PyTorch **LSTM → Transformer-encoder**
sequence model to forecast the battery State of Charge of the TUM EV fleet from the past
10 minutes of live diagnostics, following the architecture of

> Feng et al., *Energy consumption prediction strategy for electric vehicle based on
> LSTM-Transformer framework*, Energy 302 (2024) 131780.

Consumes the per-trip 1 Hz parquets built by
[`preprocessing/02_trip_timeseries`](../../preprocessing/02_trip_timeseries/README.md)
(`data/processed/trips/` + `trips_index.parquet`).

## Task formulation

- **Input** — a 600-step (10 min, 1 Hz) window of 7 signals (speed, SoC, pack voltage,
  aux power, battery inlet/outlet temps, ambient), plus derived acceleration and a binary
  missingness flag per signal → **15 channels per timestep**.
- **Target** — **ΔSoC** = `soc[t₀+150 s] − soc[t₀]` in percentage points (scalar).
  ΔSoC rather than absolute SoC because SoC % maps to different energy per vehicle;
  a learned **vehicle embedding** (7 vehicles → 8-d) absorbs per-vehicle capacity /
  efficiency differences. Absolute forecasts are reconstructed as
  `SoC(t₀) + ΔSoC_pred` and chained over a trip (paper Eq. 27).
- **Horizon = 150 s.** (The original 300 s choice was justified by the raw SoC
  quantization noise floor — a 60 s ΔSoC, std ≈ 0.22 pp, was below it, while 300 s ΔSoC,
  std ≈ 1.6 pp, carried real signal. That tradeoff hasn't been re-measured at 150 s;
  rerun `python dataset.py` / check ΔSoC std on train windows if you want to confirm
  it's still above the noise floor.)

## Differences from the paper (deliberate)

| Paper | Here | Why |
|---|---|---|
| 400 s segments → hand-crafted aggregate features | direct 1 Hz sequences | our data is 20× denser (1 Hz vs 0.05 Hz); let the LSTM learn the features |
| per-driver models | one model + vehicle embedding | 7 vehicles share one dataset; embedding carries identity |
| SoC reconstructed by Ah integration | ΔSoC target from logged SoC | no pack-current signal in the UDS logs, so Ah integration / kWh targets are not computable |
| weather-site environmental features | on-board ambient temp only | that's what the fleet logged |

## Data handling

- **Split** — chronological **at trip level**: trips sorted by `start_time`, whole trips
  assigned 70 / 15 / 15 to train/val/test (1544 / 331 / 332 trips). Windows are cut
  *after* the split and never cross trip boundaries, so train and test share no trips.
- **Windows** — length 600, stride 30 s, target 150 s ahead. Trips shorter than
  12.5 min (`INPUT_LEN + HORIZON`) yield no windows. (Window counts change with these
  settings; rerun `python dataset.py` for current split sizes.)
- **Missing data** — never fabricated for the target: a window is kept only if SoC is
  actually observed at both ends of the horizon and SoC/speed coverage in the input is
  ≥ 90%. Input NaNs are ffill/bfill-imputed (train-mean for never-observed channels) and
  the per-channel missingness flags tell the model what was imputed.
- **Scaling** — per-feature z-score fit on **train trips only**, stored in the run dir
  (`scalers.json`, also embedded in checkpoints).

## Model

`Linear(23→64) → LSTM(64, 1 layer) → +sinusoidal PE → TransformerEncoder(4 layers,
4 heads, FF 128, dropout 0.1, pre-norm) → last timestep → MLP head → ΔSoC` — ~173k
parameters. All hyperparameters in [`config.py`](config.py).

## Usage

```bash
# from this directory, venv active (torch CPU wheel:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu)

python dataset.py            # sanity: split sizes, window counts, batch shapes
python model.py              # sanity: forward pass + parameter count

python train.py --max-trips 50 --epochs 2 --run-name smoke   # ~1 min smoke test
python train.py --run-name full_v1 --epochs 40               # full run (CPU: ~25 min/epoch)

python evaluate.py --run full_v1                              # metrics + baselines + plots
```

Training: MAE loss (per paper), Adam 1e-3, ReduceLROnPlateau, early stopping
(patience 10), grad-clip 1.0, seed 42. Outputs land in `<repo>/runs/<run-name>/`
(gitignored): `best.pt`, `last.pt`, `scalers.json`, `metrics.csv`, `plots/`.

## Evaluation

`evaluate.py` reports on the test split:

- **MAE / RMSE in SoC pp** (primary), plus MAPE restricted to |ΔSoC| > 0.5 pp — raw MAPE
  is meaningless when true ΔSoC ≈ 0 (the paper's targets never cross zero; ours do).
- **Baselines the model must beat**: persistence (ΔSoC = 0) and least-squares SoC-slope
  extrapolation (fit on the last 60 s / 300 s of the window).
- **Per-vehicle breakdown** (CUP3 has only 13 train trips — expect it weaker).
- **Trip rollouts**: chained forecasting through the longest test trips
  (`SoC(t+H) = SoC(t) + ΔSoC_pred`), plotted against the true SoC curve.

## Results

<!-- filled after the full_v1 run -->
*Pending — run `python evaluate.py --run full_v1` after training completes.*
