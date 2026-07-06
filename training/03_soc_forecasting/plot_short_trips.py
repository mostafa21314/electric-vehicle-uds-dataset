"""
Plot true vs. predicted SoC for real trips, predicting from the very start of
the trip (t=0) instead of waiting for a full real window like
evaluate.rollout_trip. Defaults to trips shorter than INPUT_LEN (600 s /
10 min), but --min-duration/--max-duration select any duration range (e.g.
trips around 20 min, which do reach a fully-real window partway through).

At each stride step (30 s) the input window is built from whatever real rows
precede the anchor, zero/mask-padded on the left up to length L -- the same
convention finalize_trips() already uses for a channel that's never observed
(value = 0 = scaled train mean, mask = 0). Once the trip has accumulated L
real rows (anchor+1 >= L), the padded portion is empty and the window is
exactly the normal full real window, so predictions naturally transition from
"padded" to "sliding" as more of the trip becomes available -- no separate
code path needed, and this works the same way regardless of trip length.
Each prediction uses the trip's *true* SoC at the anchor (not chained off a
previous prediction), isolating the effect of padding from compounding
rollout error.

Usage:
    python plot_short_trips.py --run full_v1                                  # 4 shortest trips (<10 min)
    python plot_short_trips.py --run full_v1 --min-duration 19 --max-duration 21 --n-trips 4
    python plot_short_trips.py --run full_v1 --vehicle ID2 --trip-id 187
"""

import argparse

import matplotlib
matplotlib.use("Agg")  # headless-safe; script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config
from dataset import REPO_ROOT, load_trips, finalize_trips
from model import LSTMTransformer

TRUE_COLOR = "#2a78d6"  # categorical slot 1 (blue)
PRED_COLOR = "#1baf7a"  # categorical slot 2 (aqua)


def predict_from_start(t, model, device, target_std):
    """Direct (non-chained) forecast at every stride step from t=0. Returns
    (anchor_seconds, predicted_soc_at_anchor_plus_H, pad_frac_at_anchor)."""
    L, H, stride = config.INPUT_LEN, config.HORIZON, config.STRIDE
    T = len(t.soc_raw)
    n_channels = config.N_INPUT_CHANNELS

    anchor_s, pred_soc, pad_frac = [], [], []
    model.eval()
    with torch.no_grad():
        for a in range(0, T - H, stride):
            if np.isnan(t.soc_raw[a]) or np.isnan(t.soc_raw[a + H]):
                continue
            avail = min(L, a + 1)
            start = a + 1 - avail
            x = np.zeros((L, n_channels), dtype=np.float32)
            x[L - avail:, :7] = t.values[start:a + 1]
            x[L - avail:, 7] = t.accel[start:a + 1]
            x[L - avail:, 8:] = t.masks[start:a + 1]
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            veh = torch.tensor([t.vehicle_idx], dtype=torch.long, device=device)
            d_pp = model(xt, veh).item() * target_std

            anchor_s.append(a + H)
            pred_soc.append(float(t.soc_raw[a]) + d_pp)  # true anchor SoC, not chained
            pad_frac.append(1 - avail / L)

    if not anchor_s:
        return None
    return np.array(anchor_s), np.array(pred_soc), np.array(pad_frac)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n-trips", type=int, default=4)
    p.add_argument("--min-duration", type=float, default=None,
                    help="minutes; select trips >= this duration (default: no lower bound)")
    p.add_argument("--max-duration", type=float, default=None,
                    help="minutes; select trips <= this duration (default: INPUT_LEN, i.e. short trips)")
    p.add_argument("--vehicle", default=None)
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

    index = pd.read_parquet(REPO_ROOT / config.INDEX_PATH)
    min_dur = args.min_duration if args.min_duration is not None else 0.0
    max_dur = args.max_duration if args.max_duration is not None else config.INPUT_LEN / 60.0
    pool = index[(index.duration_min >= min_dur) & (index.duration_min <= max_dur)].copy()
    print(f"{len(pool)} real trips with duration in [{min_dur:.1f}, {max_dur:.1f}] min "
          f"out of {len(index)} total (INPUT_LEN = {config.INPUT_LEN / 60:.1f} min)")

    if args.vehicle is not None and args.trip_id is not None:
        chosen = pool[(pool.vehicle_id == args.vehicle) & (pool.trip_id == args.trip_id)]
        if chosen.empty:
            raise SystemExit(f"{args.vehicle} trip {args.trip_id} is not in the index or not in that duration range")
    else:
        chosen = pool.sort_values("duration_min", ascending=True).head(args.n_trips)

    trips = load_trips(chosen)
    finalize_trips(trips, scaler)  # use the TRAINED scaler, do not refit

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    L = config.INPUT_LEN

    for row in chosen.itertuples():
        key = (row.vehicle_id, row.trip_id)
        t = trips[key]
        result = predict_from_start(t, model, device, target_std)
        if result is None:
            print(f"{key}: skipped, no anchor has a full {config.HORIZON}s horizon inside the trip")
            continue
        anchor_s, pred_soc, pad_frac = result

        true_soc = t.soc_raw
        obs = ~np.isnan(true_soc)
        minutes = np.arange(len(true_soc)) / 60.0

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(minutes[obs], true_soc[obs], color=TRUE_COLOR, lw=2, label="true SoC")
        ax.plot(anchor_s / 60.0, pred_soc, color=PRED_COLOR, lw=1.5, ls="--", marker="o", ms=3,
                label="predicted (from true anchor)")
        if anchor_s[-1] >= L:  # trip long enough to reach a fully-real window at some point
            ax.axvline(L / 60.0, color="#898781", lw=1, ls=":")
            ax.text(L / 60.0, ax.get_ylim()[1], " window full from here", va="top",
                    ha="left", fontsize=8, color="#52514e")
        ax.set_xlabel("minutes into trip")
        ax.set_ylabel("SoC (%)")
        ax.set_title(f"{row.vehicle_id} trip {row.trip_id} -- {row.duration_min:.1f} min trip")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

        out_path = plots_dir / f"short_trip_{row.vehicle_id}_{row.trip_id}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"{key}: {row.duration_min:.1f} min, {len(anchor_s)} prediction(s), "
              f"pad_frac {pad_frac[0]:.0%} -> {pad_frac[-1]:.0%} -> {out_path}")


if __name__ == "__main__":
    main()
