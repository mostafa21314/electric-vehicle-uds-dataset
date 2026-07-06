"""
Run a trained checkpoint on the synthetic cold-start trips from
make_synthetic_trips.py, to see how prediction error behaves as the fraction
of fake/missing leading input steps changes.

Reuses the trained scaler (never refit) and the standard windowing/coverage
logic from dataset.py, so synthetic windows are processed identically to real
ones -- only the trip source (data/processed/trips_synthetic/) differs.

Usage:
    python test_synthetic.py --run full_v1
    python test_synthetic.py --run full_v1 --min-cov 0.0   # for --mode nan synthetic trips
"""

import argparse

import numpy as np
import pandas as pd
import torch

import config
from dataset import REPO_ROOT, load_trips, finalize_trips, build_window_index, TripWindowDataset
from model import LSTMTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--synthetic-index", default="data/processed/trips_synthetic/synthetic_index.parquet")
    p.add_argument("--min-cov", type=float, default=None,
                   help="override MIN_SOC_COV/MIN_SPEED_COV (e.g. 0.0 for --mode nan synthetic trips, "
                        "which fail the default 0.90 coverage filters)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.min_cov is not None:
        config.MIN_SOC_COV = args.min_cov
        config.MIN_SPEED_COV = args.min_cov

    run_dir = REPO_ROOT / config.RUNS_DIR / args.run
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    target_std = scaler["target_std"]
    device = torch.device(args.device)

    snap = ckpt.get("config", {})
    model = LSTMTransformer(
        d_model=snap.get("arg_d_model", config.D_MODEL),
        n_encoder_layers=snap.get("arg_n_layers", config.N_ENCODER_LAYERS),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"loaded {run_dir / 'best.pt'} (epoch {ckpt['epoch']}, val MAE {ckpt['val_mae_pp']:.4f} pp)")

    index = pd.read_parquet(REPO_ROOT / args.synthetic_index)
    trips = load_trips(index)
    finalize_trips(trips, scaler)  # impute/scale with the TRAINED scaler, not refit

    keys = list(trips.keys())
    windows = build_window_index(trips, keys)
    print(f"{len(windows)} synthetic windows from {len(keys)} trips "
          f"(MIN_SOC_COV={config.MIN_SOC_COV}, MIN_SPEED_COV={config.MIN_SPEED_COV})")
    if not windows:
        print("no windows survived the coverage filters -- try --min-cov 0.0 for --mode nan trips")
        return

    ds = TripWindowDataset(trips, windows, target_std)
    lead = config.INPUT_LEN // 2  # rows overwritten at the start of each synthetic trip

    rows = []
    with torch.no_grad():
        for i, (key, s) in enumerate(windows):
            x, veh, y = ds[i]
            pred_pp = model(x.unsqueeze(0).to(device), veh.unsqueeze(0).to(device)).item() * target_std
            true_pp = y.item() * target_std
            fake_steps = max(0, lead - s)
            rows.append({
                "vehicle": key[0], "trip": key[1], "offset": s,
                "fake_steps": fake_steps, "fake_frac": fake_steps / config.INPUT_LEN,
                "pred_pp": pred_pp, "true_pp": true_pp, "abs_err": abs(pred_pp - true_pp),
            })
    df = pd.DataFrame(rows)

    print("\n=== Error vs. fraction of fake leading input (averaged across trips/vehicles) ===")
    by_offset = (
        df.groupby("offset")
        .agg(fake_frac=("fake_frac", "first"), mae_pp=("abs_err", "mean"), n=("abs_err", "size"))
        .reset_index()
    )
    print(by_offset.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nOverall synthetic MAE: {df.abs_err.mean():.4f} pp  (n={len(df)})")
    fully_real = df[df.fake_steps == 0]
    if len(fully_real):
        print(f"Fully-real-window subset MAE: {fully_real.abs_err.mean():.4f} pp  (n={len(fully_real)}) "
              f"-- compare this to the model's real test-set MAE (evaluate.py) as a sanity check")


if __name__ == "__main__":
    main()
