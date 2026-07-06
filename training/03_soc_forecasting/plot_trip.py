"""
Plot true vs. chained-predicted SoC over time for one trip from a given split
(default: the longest validation trip). Reuses the trained scaler and the
chained rollout logic from evaluate.py (SoC(t+H) = SoC(t) + DeltaSoC_pred).

Usage:
    python plot_trip.py --run full_v1                       # longest val trip
    python plot_trip.py --run full_v1 --vehicle ID2 --trip-id 465
    python plot_trip.py --run full_v1 --split test
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # headless-safe; script only saves PNGs, doesn't need a display
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import REPO_ROOT, build_datasets
from evaluate import rollout_trip
from model import LSTMTransformer

TRUE_COLOR = "#2a78d6"  # categorical slot 1 (blue)
PRED_COLOR = "#1baf7a"  # categorical slot 2 (aqua)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--vehicle", default=None, help="e.g. ID2 (default: pick longest trip in --split)")
    p.add_argument("--trip-id", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

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

    datasets, _, index = build_datasets(scaler=scaler)
    ds = datasets[args.split]
    split_index = index[index.split == args.split].sort_values("duration_min", ascending=False)

    if args.vehicle is not None and args.trip_id is not None:
        key = (args.vehicle, args.trip_id)
        if key not in ds.trips:
            raise SystemExit(f"{key} is not in the {args.split} split")
        row = split_index[(split_index.vehicle_id == args.vehicle) & (split_index.trip_id == args.trip_id)].iloc[0]
    else:
        row = split_index.iloc[0]
        key = (row.vehicle_id, row.trip_id)
    t = ds.trips[key]

    result = rollout_trip(t, model, device, target_std)
    if result is None:
        raise SystemExit(f"{key}: no valid rollout (SoC not observed at first anchor)")
    recon_t, recon_v = result

    true_soc = t.soc_raw
    obs = ~np.isnan(true_soc)
    minutes = np.arange(len(true_soc)) / 60.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(minutes[obs], true_soc[obs], color=TRUE_COLOR, lw=2, label="true SoC")
    ax.plot(recon_t / 60.0, recon_v, color=PRED_COLOR, lw=2, ls="--", marker="o", ms=4,
            label="predicted (chained)")
    ax.set_xlabel("minutes into trip")
    ax.set_ylabel("SoC (%)")
    ax.set_title(f"{row.vehicle_id} trip {row.trip_id} -- {args.split} split ({row.duration_min:.0f} min)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / f"trip_{args.split}_{row.vehicle_id}_{row.trip_id}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
