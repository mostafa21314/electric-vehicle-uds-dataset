# 02 – Per-trip time-series builder

Turns the raw long-format UDS logs (`data/uds_data/{vehicle}.parquet`, columns
`[vehicle_id, time, value_id, value]`) into **aligned, per-driving-trip multivariate
time-series** — the intermediate needed to later train an LSTM to forecast State of Charge
(SoC) from past live diagnostics.

## What it produces

- `data/processed/trips/{vehicle}/trip_{NNNN}.parquet` — one file per driving trip. Wide
  format, one row **per second** (1 Hz grid), columns in physical units:

  | column | signal | unit | value_id |
  |---|---|---|---|
  | `time` | timestamp (1 s grid) | — | — |
  | `speed` | vehicle speed | km/h | 4 |
  | `soc` | State of Charge (**target**) | % | 900 |
  | `pack_voltage` | HV pack voltage | V | 1200 |
  | `aux_power` | HV auxiliary power | W | 56 |
  | `batt_inlet` | HV battery inlet temp | °C | 1272 |
  | `batt_outlet` | HV battery outlet temp | °C | 1273 |
  | `ambient` | ambient air temp | °C | 15 |

- `data/processed/trips_index.parquet` — one row per trip: `vehicle_id, trip_id, path,
  start_time, end_time, duration_min, n_rows, soc_start, soc_end, max_speed`, plus a
  `cov_<feature>` column giving each feature's **coverage** (fraction of non-NaN rows).

Both outputs are git-ignored.

## Handling the different sampling rates

Every signal is logged on its own clock (speed & voltage at 200 ms, SoC at 5 s, temps at
5–10 s, etc.). They are all reconciled onto **one common 1 Hz grid**:

- **Fast signals** (`speed`, `pack_voltage`, `aux_power`) are **downsampled** — averaged
  into each 1-second bin (light anti-aliasing).
- **Slow signals** (`soc`, temps, `ambient`) are **upsampled** — time-interpolated to fill
  the in-between seconds, but **never across gaps longer than `MAX_INTERP_GAP_S`** (30 s),
  so genuine dropouts stay `NaN` rather than being fabricated.

## Missing data

Several signals are intermittent in real driving: `aux_power` (~88% of trips), and
`ambient` / `batt_inlet` / `batt_outlet` (the good battery temps only exist from ~Aug
2021; `ambient` is logged mostly while parked). By default (`KEEP_ALL_TRIPS = True`) **every
driving trip is kept and unavailable signals are left as `NaN`** — the `cov_*` columns in
the index let you filter trips/features however you like downstream. Set
`KEEP_ALL_TRIPS = False` to instead drop any trip whose features fall below `MIN_COVERAGE`.

## Trip segmentation

Driving trips are cut from the **speed** signal: a gap > `GAP_SECONDS` (120 s) starts a new
session; a session is kept only if its duration is within `[MIN_TRIP_MIN, MAX_TRIP_MIN]`
(5–90 min) and it reaches `MIN_SPEED_KMH` (5 km/h) — which excludes charging/parking. All
values are clipped to the physical `[min_val, max_val]` from `data/value_overview.csv`.

## Usage

```bash
# from this directory, using the repo's .venv
python build_trips.py                    # all vehicles (config.VEHICLES)
python build_trips.py --vehicles ID1     # one vehicle (smoke test)
python build_trips.py --vehicles ID1 --limit 15   # cap trips per vehicle (quick)
```

All knobs (features, grid frequency, gap/interp limits, trip filters, missing-data policy)
live in [`config.py`](config.py).

## Not done here (left for training)

Windowing / sequence length, forecast horizon, feature scaling/normalization, and the
train/val/test split are intentionally **not** applied — the per-trip tables plus the index
keep those choices free. Typical next steps: sort the index by `start_time` for a
chronological split (or group by `vehicle_id` for leave-one-vehicle-out), fit a scaler on
the train split only, then slide a window within each trip file.
