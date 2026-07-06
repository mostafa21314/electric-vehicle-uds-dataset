"""
Build aligned, per-trip multivariate time-series from the raw UDS logs.

The raw data (``data/uds_data/{vehicle}.parquet``) is a long-format log
``[vehicle_id, time, value_id, value]`` where every signal has its own sampling
rate and driving / charging / parking are interleaved. This script turns it into
one clean wide table per *driving trip*, resampled onto a common time grid
(default 1 Hz), in physical units and free of NaNs.

Outputs
-------
- ``data/processed/trips/{vehicle}/trip_{NNNN}.parquet`` : one file per trip.
- ``data/processed/trips_index.parquet``                : one row per trip.

Windowing, scaling and the train/val/test split are intentionally *not* done
here; they are left to the training code so those choices stay free.

Usage
-----
    python build_trips.py                    # all vehicles in config.VEHICLES
    python build_trips.py --vehicles ID1     # subset (smoke test)
    python build_trips.py --limit 20         # cap trips per vehicle (quick run)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Repository root = two levels up from this file (preprocessing/02_.../build_trips.py)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _abspath(rel: str) -> Path:
    """Resolve a config path relative to the repository root."""
    return REPO_ROOT / rel


def load_value_ranges() -> dict:
    """Return {value_id: (min_val, max_val)} from value_overview.csv for clipping."""
    ov = pd.read_csv(_abspath(config.VALUE_OVERVIEW_CSV))
    return {
        int(r.value_id): (float(r.min_val), float(r.max_val))
        for r in ov.itertuples(index=False)
    }


def segment_trips(speed: pd.DataFrame) -> list:
    """
    Split the speed signal (value_id 4) into driving sessions.

    A gap larger than ``GAP_SECONDS`` between consecutive speed samples starts a
    new session. Returns a list of (start_time, end_time) for sessions that pass
    the duration and motion filters.
    """
    speed = speed.sort_values("time")
    dt = speed["time"].diff().dt.total_seconds()
    session_id = (dt > config.GAP_SECONDS).cumsum()

    trips = []
    for _, grp in speed.groupby(session_id):
        t0, t1 = grp["time"].iloc[0], grp["time"].iloc[-1]
        duration_min = (t1 - t0).total_seconds() / 60.0
        if duration_min < config.MIN_TRIP_MIN or duration_min > config.MAX_TRIP_MIN:
            continue
        if grp["value"].max() < config.MIN_SPEED_KMH:
            continue  # never really moved -> not a driving trip
        trips.append((t0, t1))
    return trips


def resample_signal(samples: pd.DataFrame, grid: pd.DatetimeIndex, name: str) -> pd.Series:
    """
    Put one signal's samples onto the common ``grid``.

    Fast signals (in DOWNSAMPLE_BY_MEAN) are averaged into each grid bin; slow
    signals are time-interpolated, but never across gaps longer than
    ``MAX_INTERP_GAP_S`` (those stay NaN so real dropouts are not fabricated).
    A signal absent from this trip yields an all-NaN column.
    """
    if samples.empty:
        return pd.Series(np.nan, index=grid, name=name)

    s = samples.set_index("time")["value"].sort_index()
    # Collapse any duplicate timestamps up front.
    s = s[~s.index.duplicated(keep="first")]

    if name in config.DOWNSAMPLE_BY_MEAN:
        # Mean over each grid bin (light anti-aliasing), then align to the grid.
        binned = s.resample(config.RESAMPLE_FREQ).mean()
        out = binned.reindex(grid)
        # Bridge sub-gap holes left by empty bins via time interpolation.
        out = out.interpolate(method="time", limit_area="inside")
    else:
        # Union the native timestamps with the grid, interpolate, then sample the grid.
        union = s.reindex(s.index.union(grid))
        union = union.interpolate(method="time", limit_area="inside")
        out = union.reindex(grid)

    # Enforce the max-gap rule: null out grid points whose nearest real sample is
    # farther than MAX_INTERP_GAP_S away on either side.
    _mask_large_gaps(out, s.index)
    return out


def _mask_large_gaps(series: pd.Series, real_times: pd.DatetimeIndex) -> None:
    """Set to NaN any grid point more than MAX_INTERP_GAP_S from a real sample (in place)."""
    if len(real_times) == 0:
        series[:] = np.nan
        return
    grid = series.index
    real = real_times.sort_values()
    pos = real.searchsorted(grid)
    prev_idx = np.clip(pos - 1, 0, len(real) - 1)
    next_idx = np.clip(pos, 0, len(real) - 1)
    prev_gap = (grid - real[prev_idx]).total_seconds().to_numpy()
    next_gap = (real[next_idx] - grid).total_seconds().to_numpy()
    nearest = np.minimum(np.abs(prev_gap), np.abs(next_gap))
    series[nearest > config.MAX_INTERP_GAP_S] = np.nan


def build_trip_frame(window: pd.DataFrame, t0, t1, ranges: dict):
    """
    Build one aligned wide frame for a single trip time window.

    Every feature is resampled onto the common grid; signals that are absent or
    too sparse are left as NaN (no fabrication). Returns ``(frame, coverage)``
    where ``coverage`` maps each feature name to its fraction of non-NaN grid
    rows. Returns ``(None, None)`` only when KEEP_ALL_TRIPS is False and a
    feature falls below MIN_COVERAGE.
    """
    # Snap the grid to whole-frequency boundaries so the downsample bins
    # (resample labels bins on floored boundaries) line up with the grid.
    start = pd.Timestamp(t0).floor(config.RESAMPLE_FREQ)
    end = pd.Timestamp(t1).floor(config.RESAMPLE_FREQ)
    grid = pd.date_range(start, end, freq=config.RESAMPLE_FREQ)
    cols = {}
    coverage = {}
    for name, vid in config.FEATURES.items():
        samples = window[window["value_id"] == vid][["time", "value"]]
        col = resample_signal(samples, grid, name)
        # Clip to physical range from value_overview.csv (NaNs pass through).
        lo, hi = ranges.get(vid, (-np.inf, np.inf))
        col = col.clip(lo, hi)
        coverage[name] = float(col.notna().mean())
        if not config.KEEP_ALL_TRIPS and coverage[name] < config.MIN_COVERAGE:
            return None, None
        cols[name] = col

    df = pd.DataFrame(cols, index=grid)
    df.index.name = "time"
    return df.reset_index(), coverage


def process_vehicle(vehicle: str, ranges: dict, limit: int | None) -> list:
    """Process one vehicle's parquet into per-trip files. Returns index rows."""
    src = _abspath(config.UDS_DATA_DIR) / f"{vehicle}.parquet"
    if not src.exists():
        print(f"  [skip] {src} not found")
        return []

    wanted = set(config.FEATURES.values())
    df = pd.read_parquet(src, columns=["time", "value_id", "value"])
    df = df[df["value_id"].isin(wanted)]
    df["time"] = pd.to_datetime(df["time"])

    speed_vid = config.FEATURES["speed"]
    trips = segment_trips(df[df["value_id"] == speed_vid][["time", "value"]])
    print(f"  {vehicle}: {len(trips)} candidate driving sessions")

    out_dir = _abspath(config.OUTPUT_DIR) / vehicle
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    trip_id = 0
    for (t0, t1) in trips:
        if limit is not None and trip_id >= limit:
            break
        window = df[(df["time"] >= t0) & (df["time"] <= t1)]
        frame, coverage = build_trip_frame(window, t0, t1, ranges)
        if frame is None:
            continue

        _assert_valid(frame, ranges)

        rel_path = f"{config.OUTPUT_DIR}/{vehicle}/trip_{trip_id:04d}.parquet"
        frame.to_parquet(_abspath(rel_path), index=False)

        soc = frame[config.TARGET]
        soc_start = soc.dropna().iloc[0] if soc.notna().any() else np.nan
        soc_end = soc.dropna().iloc[-1] if soc.notna().any() else np.nan
        row = {
            "vehicle_id": vehicle,
            "trip_id": trip_id,
            "path": rel_path,
            "start_time": frame["time"].iloc[0],
            "end_time": frame["time"].iloc[-1],
            "duration_min": round((frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds() / 60.0, 2),
            "n_rows": len(frame),
            "soc_start": round(float(soc_start), 2) if pd.notna(soc_start) else np.nan,
            "soc_end": round(float(soc_end), 2) if pd.notna(soc_end) else np.nan,
            "max_speed": round(float(frame["speed"].max()), 1),
        }
        # Per-feature coverage (fraction of non-NaN rows), prefixed cov_*.
        row.update({f"cov_{name}": round(cov, 3) for name, cov in coverage.items()})
        index_rows.append(row)
        trip_id += 1

    print(f"  {vehicle}: wrote {trip_id} trips")
    return index_rows


def _assert_valid(frame: pd.DataFrame, ranges: dict) -> None:
    """Sanity checks that must hold for every emitted trip (NaN-tolerant)."""
    t = frame["time"]
    assert t.is_monotonic_increasing and t.is_unique, "time index not strictly increasing"
    for name, vid in config.FEATURES.items():
        lo, hi = ranges.get(vid, (-np.inf, np.inf))
        col = frame[name]
        # Non-NaN values must sit within the physical range; NaNs are allowed.
        valid = col.dropna()
        assert (valid >= lo - 1e-6).all() and (valid <= hi + 1e-6).all(), f"{name} out of [{lo},{hi}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vehicles", nargs="+", default=None,
                        help="subset of vehicles to process (default: config.VEHICLES)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max trips per vehicle (quick smoke run)")
    args = parser.parse_args()

    vehicles = args.vehicles or config.VEHICLES
    ranges = load_value_ranges()

    _abspath(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    all_rows = []
    for v in vehicles:
        print(f"Processing {v} ...")
        all_rows.extend(process_vehicle(v, ranges, args.limit))

    if not all_rows:
        print("No trips produced.")
        return 1

    index = pd.DataFrame(all_rows).sort_values(["vehicle_id", "start_time"]).reset_index(drop=True)
    index.to_parquet(_abspath(config.INDEX_PATH), index=False)
    print(f"\nDone. {len(index)} trips across {index['vehicle_id'].nunique()} vehicles.")
    print(f"Index -> {config.INDEX_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
