# Dataset Summary — EV Driving-Trip Time-Series

A model-ready intermediate for training an LSTM to **forecast battery State of Charge
(SoC)** from past live diagnostics. Built from the raw TUM UDS logs (`data/uds_data/`) by
the pipeline in [`build_trips.py`](build_trips.py). For how it is produced, see
[`README.md`](README.md).

## What it is

Each raw parquet is a long-format log (`vehicle_id, time, value_id, value`) where every
signal has its own sampling rate and driving / charging / parking are interleaved. The
pipeline converts this into **one clean wide table per driving trip**, resampled onto a
shared **1 Hz grid**, in physical units.

## Scale

| | |
|---|---|
| Driving trips | **2,207** (one parquet each) |
| Total driving time | **789.9 hours** |
| Total rows (1-second steps) | **2,845,786** |
| Date span | **2021-02-02 → 2023-10-10** |
| On-disk size | **76 MB** + `trips_index.parquet` |
| Vehicles | 7 (2× VW ID.3, 5× CUPRA Born) |

## Per-vehicle breakdown

| Vehicle | Trips | Hours | Coverage span |
|---|---:|---:|---|
| ID2 | 774 | 296.1 | 2021-07 → 2023-10 |
| ID1 | 472 | 178.7 | 2021-02 → 2023-06 |
| CUP1 | 371 | 132.9 | 2022-11 → 2023-04 |
| CUP4 | 220 | 71.8 | 2022-11 → 2023-04 |
| CUP5 | 168 | 53.9 | 2022-11 → 2023-04 |
| CUP2 | 158 | 44.1 | 2022-11 → 2023-04 |
| CUP3 | 44 | 12.4 | 2022-11 → 2023-04 |

> The two VW ID.3s carry the long history; all five CUPRAs only start Nov 2022 (~5 months
> each) but with fuller feature coverage.

## Features (columns of each trip file)

| Column | Signal | Unit | value_id | Mean coverage |
|---|---|---|---|---:|
| `time` | timestamp (1 s grid) | — | — | 100% |
| `speed` | vehicle speed | km/h | 4 | 98.7% |
| `soc` | **State of Charge (target)** | % | 900 | 96.6% |
| `pack_voltage` | HV pack voltage | V | 1200 | 98.4% |
| `aux_power` | HV auxiliary power | W | 56 | 93.1% |
| `batt_inlet` | battery inlet temp | °C | 1272 | 89.0% |
| `batt_outlet` | battery outlet temp | °C | 1273 | 89.1% |
| `ambient` | ambient air temp | °C | 15 | 90.8% |

Missing / absent signals are left as **NaN** (not fabricated); the battery temps are
largely NaN on the earliest ID1 trips (Feb–mid 2021) because those sensors weren't logged
yet.

## How it was built

- **Segmentation** — trips cut from the speed signal (gap > 120 s splits trips); kept only
  if 5–90 min long and reaching ≥ 5 km/h (excludes charging / parking).
- **Rate alignment** — fast signals (speed, voltage, aux) averaged into 1 s bins; slow
  signals (SoC, temps, ambient) time-interpolated, but never across gaps > 30 s.
- **Cleaning** — all values clipped to the physical `[min, max]` from
  `data/value_overview.csv`.

## Index file — `data/processed/trips_index.parquet`

One row per trip: `vehicle_id, trip_id, path, start_time, end_time, duration_min, n_rows,
soc_start, soc_end, max_speed`, plus a `cov_<feature>` coverage column for each signal. Use
it to select / filter trips without opening every file.

## Sanity signals

- **96.6%** of trips show SoC decreasing over the drive (physically correct); median drop
  **2.8 pp**, max 74 pp.
- Trip duration: median **17.5 min** (p10 6.7, p90 41.4).
- SoC start values span **9–100%**.

## Deliberately *not* done (left for the training stage)

Windowing / sequence length, forecast horizon, feature scaling / normalization, and the
train / val / test split — so those choices stay free. Suggested next steps: sort the index
by `start_time` for a chronological split (or group by `vehicle_id` for
leave-one-vehicle-out), fit a scaler on the train split only, then slide windows within
each trip file.
