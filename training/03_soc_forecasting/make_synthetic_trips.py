"""
Generate synthetic "cold-start" trips for probing model behaviour when there is
not enough real data at the start of a window.

Each synthetic trip is a copy of a real trip whose first ``INPUT_LEN // 2`` rows
have every feature column overwritten. The earliest window (start offset 0) then
has its first half filled with the placeholder value and its second half real,
tapering back to a fully-real window by offset ``INPUT_LEN // 2``. The forecast
anchor (last input step) and the target step always land in the real second
half, so the target is still computed from genuine SoC values.

Fill modes:
  zero  (default) -- write literal 0.0. The dataset's missingness mask stays 1
                     (the model treats these as *observed* zeros), and the
                     window survives the SoC/speed coverage filters.
  nan             -- write NaN, i.e. genuinely missing. The mask goes to 0 and
                     imputation kicks in, but a half-missing window is dropped by
                     the default MIN_SOC_COV / MIN_SPEED_COV = 0.90 filters --
                     lower those in config.py to keep such windows.

Output goes to a SEPARATE folder (default data/processed/trips_synthetic/), with
a synthetic_index.parquet mirroring the real trips_index schema so the files can
be loaded the same way. Nothing here touches the real dataset.

Usage:
    python make_synthetic_trips.py                    # 8 trips, zero fill
    python make_synthetic_trips.py --n-trips 12 --mode nan
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from dataset import REPO_ROOT

FEATURES = config.FEATURES


def build_index_row(vehicle_id, trip_id, path, df):
    """Recompute the trips_index fields from a (possibly modified) trip frame."""
    soc = df["soc"].to_numpy()
    soc_obs = soc[~np.isnan(soc)]
    row = {
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "path": path,
        "start_time": df["time"].iloc[0],
        "end_time": df["time"].iloc[-1],
        "duration_min": (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds() / 60.0,
        "n_rows": len(df),
        "soc_start": float(soc_obs[0]) if soc_obs.size else np.nan,
        "soc_end": float(soc_obs[-1]) if soc_obs.size else np.nan,
        "max_speed": float(np.nanmax(df["speed"].to_numpy())),
    }
    for feat in FEATURES:
        row[f"cov_{feat}"] = float(df[feat].notna().mean())
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trips", type=int, default=8, help="how many synthetic trips to make")
    p.add_argument("--mode", choices=["zero", "nan"], default="zero",
                   help="fill leading half-window with literal 0.0 or with NaN (missing)")
    p.add_argument("--out", default="data/processed/trips_synthetic",
                   help="output folder (repo-relative)")
    args = p.parse_args()

    lead = config.INPUT_LEN // 2                      # rows to overwrite at the start
    min_rows = config.INPUT_LEN + config.HORIZON      # need one valid window with a target
    fill = 0.0 if args.mode == "zero" else np.nan

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_parquet(REPO_ROOT / config.INDEX_PATH)
    eligible = index[index.n_rows >= min_rows].sort_values("n_rows", ascending=False)
    chosen = eligible.head(args.n_trips)
    if len(chosen) < args.n_trips:
        print(f"warning: only {len(chosen)} trips have >= {min_rows} rows")

    rows = []
    for src in chosen.itertuples():
        src_path = Path(src.path)
        if not src_path.is_absolute():
            src_path = REPO_ROOT / src_path
        df = pd.read_parquet(src_path)
        df.loc[df.index[:lead], list(FEATURES)] = fill  # zero/blank the leading half-window

        veh_dir = out_dir / src.vehicle_id
        veh_dir.mkdir(exist_ok=True)
        dst = veh_dir / f"trip_{src.trip_id:04d}.parquet"
        df.to_parquet(dst)

        rel = dst.relative_to(REPO_ROOT).as_posix()
        rows.append(build_index_row(src.vehicle_id, int(src.trip_id), rel, df))
        print(f"  {src.vehicle_id} trip {src.trip_id}: {len(df)} rows, "
              f"first {lead} zeroed ({args.mode}) -> {rel}")

    idx_path = out_dir / "synthetic_index.parquet"
    pd.DataFrame(rows).to_parquet(idx_path)

    print(f"\nWrote {len(rows)} synthetic trips + index to {out_dir}")
    print(f"Leading {lead} of {config.INPUT_LEN} input steps are '{args.mode}'-filled "
          f"=> earliest window (offset 0) is 50% placeholder, fully real by offset {lead}.")
    if args.mode == "nan":
        print("NOTE: half-missing windows fail the default 0.90 coverage filters; "
              "lower MIN_SOC_COV / MIN_SPEED_COV in config.py to keep them.")


if __name__ == "__main__":
    main()
