"""
Train the LSTM-Transformer DeltaSoC forecaster.

Usage:
    python train.py --run-name full_v1
    python train.py --max-trips 50 --epochs 2 --run-name smoke   # smoke test

Outputs to <repo>/runs/<run-name>/: best.pt, last.pt, scalers.json, metrics.csv.
Checkpoints bundle the scaler stats and a config snapshot so evaluate.py is
self-contained.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from dataset import REPO_ROOT, build_datasets, make_dataloaders
from model import LSTMTransformer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_snapshot(args) -> dict:
    snap = {k: v for k, v in vars(config).items() if k.isupper()}
    snap.update({f"arg_{k}": v for k, v in vars(args).items()})
    return snap


def run_epoch(model, loader, device, target_std, optimizer=None):
    """One pass; returns MAE in SoC percentage points."""
    training = optimizer is not None
    model.train(training)
    loss_fn = torch.nn.L1Loss()
    total_abs, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, veh, y in tqdm(loader, leave=False, desc="train" if training else "val"):
            x, veh, y = x.to(device), veh.to(device), y.to(device)
            pred = model(x, veh)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
            total_abs += (pred - y).abs().sum().item()
            n += len(y)
    return total_abs / max(n, 1) * target_std  # back to pp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="run")
    p.add_argument("--max-trips", type=int, default=None)
    p.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--d-model", type=int, default=config.D_MODEL)
    p.add_argument("--n-layers", type=int, default=config.N_ENCODER_LAYERS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(config.SEED)
    device = torch.device(args.device)
    run_dir = REPO_ROOT / config.RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device: {device} | run dir: {run_dir}")
    t0 = time.time()
    datasets, scaler, index = build_datasets(args.max_trips)
    loaders = make_dataloaders(datasets, args.batch_size)
    print(
        f"data ready in {time.time() - t0:.0f}s | windows: "
        + " ".join(f"{k}={len(v)}" for k, v in datasets.items())
    )
    with open(run_dir / "scalers.json", "w") as f:
        json.dump(scaler, f, indent=2)

    model = LSTMTransformer(d_model=args.d_model, n_encoder_layers=args.n_layers).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=config.LR_PATIENCE
    )

    best_val, best_epoch = float("inf"), -1
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_mae_pp", "val_mae_pp", "lr", "seconds"])

    target_std = scaler["target_std"]
    for epoch in range(args.epochs):
        t_ep = time.time()
        train_mae = run_epoch(model, loaders["train"], device, target_std, optimizer)
        val_mae = run_epoch(model, loaders["val"], device, target_std)
        scheduler.step(val_mae)
        lr_now = optimizer.param_groups[0]["lr"]
        secs = time.time() - t_ep
        print(
            f"epoch {epoch:3d} | train MAE {train_mae:.4f} pp | "
            f"val MAE {val_mae:.4f} pp | lr {lr_now:.1e} | {secs:.0f}s"
        )
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_mae:.5f}", f"{val_mae:.5f}", lr_now, f"{secs:.1f}"])

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_mae_pp": val_mae,
            "scaler": scaler,
            "config": config_snapshot(args),
        }
        torch.save(state, run_dir / "last.pt")
        if val_mae < best_val:
            best_val, best_epoch = val_mae, epoch
            torch.save(state, run_dir / "best.pt")
        elif epoch - best_epoch >= config.EARLY_STOP_PATIENCE:
            print(f"early stop: no val improvement for {config.EARLY_STOP_PATIENCE} epochs")
            break

    print(f"best val MAE {best_val:.4f} pp @ epoch {best_epoch} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
