"""
Evaluate a trained run on the test split.

    python evaluate.py --run full_v1

Reports MAE / RMSE (SoC pp) and guarded MAPE against two baselines
(persistence and linear SoC-slope extrapolation), a per-vehicle breakdown,
and reconstructs full-trip SoC trajectories by chaining predictions
(paper Eq. 27: SoC(t+H) = SoC(t) + DeltaSoC_pred) on the longest test trips.
Plots go to <run dir>/plots/.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from dataset import REPO_ROOT, build_datasets, make_dataloaders
from model import LSTMTransformer


def predict(model, loader, device, target_std):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, veh, y in loader:
            pred = model(x.to(device), veh.to(device))
            preds.append(pred.cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds) * target_std, np.concatenate(trues) * target_std


def slope_baseline(ds, lookback: int) -> np.ndarray:
    """Least-squares SoC slope over the last `lookback` s of the input window,
    extrapolated over the horizon. Uses only observed SoC values."""
    L, H = config.INPUT_LEN, config.HORIZON
    out = np.zeros(len(ds.windows), np.float32)
    for i, (key, s) in enumerate(ds.windows):
        t = ds.trips[key]
        anchor = s + L - 1
        seg = t.soc_raw[anchor - lookback + 1 : anchor + 1]
        idx = np.flatnonzero(~np.isnan(seg))
        if len(idx) >= 2 and np.ptp(idx) > 0:
            slope = np.polyfit(idx.astype(np.float32), seg[idx], 1)[0]
            out[i] = slope * H
    return out


def metrics(pred_pp, true_pp):
    err = pred_pp - true_pp
    mae = np.abs(err).mean()
    rmse = np.sqrt((err**2).mean())
    big = np.abs(true_pp) > 0.5
    mape = np.abs(err[big] / true_pp[big]).mean() * 100 if big.any() else float("nan")
    return mae, rmse, mape


def make_window_tensor(t, s):
    L = config.INPUT_LEN
    x = np.concatenate([t.values[s : s + L], t.accel[s : s + L, None], t.masks[s : s + L]], axis=1)
    return torch.from_numpy(x)


def rollout_trip(t, model, device, target_std):
    """Chain predictions through a trip in hops of HORIZON (paper Eq. 27).
    Inputs are the recorded windows; error accumulates through the summed
    DeltaSoC predictions."""
    L, H = config.INPUT_LEN, config.HORIZON
    T = len(t.soc_raw)
    anchors = [a for a in range(L - 1, T - H, H)]
    if not anchors or np.isnan(t.soc_raw[anchors[0]]):
        return None
    recon_t = [anchors[0]]
    recon_v = [float(t.soc_raw[anchors[0]])]
    model.eval()
    with torch.no_grad():
        for a in anchors:
            x = make_window_tensor(t, a - L + 1).unsqueeze(0).to(device)
            veh = torch.tensor([t.vehicle_idx], dtype=torch.long, device=device)
            d_pp = model(x, veh).item() * target_std
            recon_t.append(a + H)
            recon_v.append(recon_v[-1] + d_pp)
    return np.array(recon_t), np.array(recon_v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max-trips", type=int, default=None, help="must match training if used")
    p.add_argument("--n-plot-trips", type=int, default=6)
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
    model.to(device)
    print(f"loaded {run_dir / 'best.pt'} (epoch {ckpt['epoch']}, val MAE {ckpt['val_mae_pp']:.4f} pp)")

    datasets, _, index = build_datasets(args.max_trips, scaler=scaler)
    test_ds = datasets["test"]
    loaders = make_dataloaders(datasets)
    print(f"test windows: {len(test_ds)}")

    pred_pp, true_pp = predict(model, loaders["test"], device, target_std)
    base_persist = np.zeros_like(true_pp)
    base_slope60 = slope_baseline(test_ds, 60)
    base_slope300 = slope_baseline(test_ds, config.INPUT_LEN)

    print("\n=== Test metrics (DeltaSoC over %ds, SoC percentage points) ===" % config.HORIZON)
    header = f"{'model':<22}{'MAE':>8}{'RMSE':>8}{'MAPE*':>9}"
    print(header)
    for name, pp in [
        ("LSTM-Transformer", pred_pp),
        ("persistence (d=0)", base_persist),
        ("slope 60s x H", base_slope60),
        ("slope 300s x H", base_slope300),
    ]:
        mae, rmse, mape = metrics(pp, true_pp)
        print(f"{name:<22}{mae:>8.4f}{rmse:>8.4f}{mape:>8.1f}%")
    print("*MAPE only on |DeltaSoC_true| > 0.5 pp "
          f"({(np.abs(true_pp) > 0.5).mean() * 100:.0f}% of windows); raw MAPE is "
          "meaningless near zero targets.")

    print("\n=== Per-vehicle MAE (pp) ===")
    veh_of_window = np.array([test_ds.trips[k].vehicle for k, _ in test_ds.windows])
    print(f"{'vehicle':<10}{'n':>7}{'model':>9}{'persist':>9}{'slope300':>10}")
    for veh in config.VEHICLE_VOCAB:
        m = veh_of_window == veh
        if not m.any():
            print(f"{veh:<10}{0:>7}")
            continue
        print(
            f"{veh:<10}{m.sum():>7}"
            f"{np.abs(pred_pp[m] - true_pp[m]).mean():>9.4f}"
            f"{np.abs(true_pp[m]).mean():>9.4f}"
            f"{np.abs(base_slope300[m] - true_pp[m]).mean():>10.4f}"
        )

    # --- Trip trajectory reconstruction -------------------------------------
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    test_index = index[index.split == "test"].sort_values("n_rows", ascending=False)
    end_errors = []
    plotted = 0
    for row in test_index.itertuples():
        if plotted >= args.n_plot_trips:
            break
        t = test_ds.trips.get((row.vehicle_id, row.trip_id))
        if t is None:
            continue
        result = rollout_trip(t, model, device, target_std)
        if result is None:
            continue
        recon_t, recon_v = result
        true_soc = t.soc_raw
        obs = ~np.isnan(true_soc)
        minutes = np.arange(len(true_soc)) / 60.0

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(minutes[obs], true_soc[obs], lw=1.2, label="true SoC")
        ax.plot(recon_t / 60.0, recon_v, "o-", ms=4, lw=1.2, label="reconstructed (chained)")
        ax.set_xlabel("minutes into trip")
        ax.set_ylabel("SoC (%)")
        ax.set_title(f"{row.vehicle_id} trip {row.trip_id} ({row.duration_min:.0f} min)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"trip_{row.vehicle_id}_{row.trip_id}.png", dpi=120)
        plt.close(fig)

        last_a = recon_t[-1]
        if not np.isnan(true_soc[last_a]):
            end_errors.append(abs(recon_v[-1] - true_soc[last_a]))
        plotted += 1

    if end_errors:
        print(
            f"\ntrajectory rollout on {plotted} longest test trips: "
            f"mean end-of-trip |error| {np.mean(end_errors):.2f} pp "
            f"(max {np.max(end_errors):.2f} pp) -> plots in {plots_dir}"
        )


if __name__ == "__main__":
    main()
