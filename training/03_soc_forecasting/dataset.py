"""
Data pipeline for SoC forecasting: chronological trip-level split, in-memory
trip loading, imputation with missingness masks, train-only scaling and
sliding-window generation.

Windows never cross trip boundaries: the window index is a list of
(trip_key, start) pairs built per trip, and each trip belongs to exactly one
split. The target is DeltaSoC = soc[anchor + HORIZON] - soc[anchor] in
percentage points, computed only from *observed* SoC values (never imputed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config

REPO_ROOT = Path(__file__).resolve().parents[2]

_FEAT = config.FEATURES
_N_FEAT = len(_FEAT)
_SOC_IDX = _FEAT.index(config.TARGET)
_SPEED_IDX = _FEAT.index("speed")


# --- Split ---------------------------------------------------------------------

def load_split_index(max_trips: int | None = None) -> pd.DataFrame:
    """Load trips_index, sort chronologically and assign whole trips to splits."""
    index = pd.read_parquet(REPO_ROOT / config.INDEX_PATH)
    index = index.sort_values("start_time").reset_index(drop=True)
    if max_trips is not None:
        index = index.head(max_trips).copy()
    n = len(index)
    n_train = int(n * config.TRAIN_FRAC)
    n_val = int(n * (config.TRAIN_FRAC + config.VAL_FRAC)) - n_train
    split = np.full(n, "test", dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    index["split"] = split
    return index


# --- Loading -------------------------------------------------------------------

class Trip:
    """One trip held in memory.

    values : (T, 7) float32, imputed + z-scored feature channels
    masks  : (T, 7) float32, 1.0 where the raw value was observed
    accel  : (T,)  float32, scaled speed diff (0 where speed unobserved)
    soc_raw: (T,)  float32, observed SoC in %, NaN where unobserved
    """

    __slots__ = ("vehicle", "trip_id", "vehicle_idx", "values", "masks", "accel", "soc_raw")

    def __init__(self, vehicle, trip_id, values, masks, soc_raw):
        self.vehicle = vehicle
        self.trip_id = trip_id
        self.vehicle_idx = config.VEHICLE_VOCAB[vehicle]
        self.values = values
        self.masks = masks
        self.soc_raw = soc_raw
        self.accel = None  # filled in finalize_trips()


def load_trips(index: pd.DataFrame) -> dict:
    """Read every trip parquet into raw (unscaled, NaN-preserving) arrays."""
    trips = {}
    for row in index.itertuples():
        path = Path(row.path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        df = pd.read_parquet(path, columns=_FEAT)
        values = df.to_numpy(dtype=np.float32)
        masks = (~np.isnan(values)).astype(np.float32)
        soc_raw = values[:, _SOC_IDX].copy()
        trips[(row.vehicle_id, row.trip_id)] = Trip(row.vehicle_id, row.trip_id, values, masks, soc_raw)
    return trips


# --- Scaling -------------------------------------------------------------------

def fit_scaler(trips: dict, train_keys: list, train_windows: list) -> dict:
    """Per-feature z-score stats over *observed* train values, plus accel and
    target (DeltaSoC) stats. Target std is fit on the actual train windows."""
    obs = [[] for _ in range(_N_FEAT)]
    accel_obs = []
    for key in train_keys:
        t = trips[key]
        for j in range(_N_FEAT):
            col = t.values[:, j]
            obs[j].append(col[~np.isnan(col)])
        speed = t.values[:, _SPEED_IDX]
        d = np.diff(speed)
        accel_obs.append(d[~np.isnan(d)])

    feat_mean, feat_std = np.zeros(_N_FEAT, np.float32), np.ones(_N_FEAT, np.float32)
    for j in range(_N_FEAT):
        v = np.concatenate(obs[j]) if obs[j] else np.array([0.0])
        if v.size:
            feat_mean[j] = v.mean()
            feat_std[j] = max(v.std(), 1e-6)
    a = np.concatenate(accel_obs) if accel_obs else np.array([0.0])
    accel_mean, accel_std = float(a.mean()), max(float(a.std()), 1e-6)

    targets = [trips[k].soc_raw[s + config.INPUT_LEN - 1 + config.HORIZON]
               - trips[k].soc_raw[s + config.INPUT_LEN - 1]
               for k, s in train_windows]
    targets = np.asarray(targets, np.float32)
    target_std = max(float(targets.std()), 1e-6) if targets.size else 1.0

    return {
        "features": _FEAT,
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "target_std": target_std,
    }


def finalize_trips(trips: dict, scaler: dict) -> None:
    """Impute (ffill -> bfill -> train mean), z-score, and derive accel, in place."""
    feat_mean = np.asarray(scaler["feat_mean"], np.float32)
    feat_std = np.asarray(scaler["feat_std"], np.float32)
    for t in trips.values():
        df = pd.DataFrame(t.values)
        df = df.ffill().bfill()
        values = df.to_numpy(dtype=np.float32)
        # Channels never observed in this trip fall back to the train mean
        # (i.e. 0 after scaling); the mask channel tells the model they're absent.
        nan_cols = np.isnan(values)
        values[nan_cols] = np.broadcast_to(feat_mean, values.shape)[nan_cols]

        speed_kmh = values[:, _SPEED_IDX]  # imputed but still physical units here
        accel = np.zeros(len(values), np.float32)
        accel[1:] = np.diff(speed_kmh)
        speed_obs = t.masks[:, _SPEED_IDX] > 0
        accel[1:] *= (speed_obs[1:] & speed_obs[:-1]).astype(np.float32)
        t.accel = (accel - scaler["accel_mean"]) / scaler["accel_std"]

        t.values = (values - feat_mean) / feat_std


# --- Windowing -----------------------------------------------------------------

def build_window_index(trips: dict, keys: list) -> list:
    """Valid (trip_key, start) pairs. Anchor = start + INPUT_LEN - 1; requires
    SoC observed at anchor and anchor + HORIZON, and SoC/speed coverage inside
    the input window."""
    L, H, stride = config.INPUT_LEN, config.HORIZON, config.STRIDE
    windows = []
    for key in keys:
        t = trips[key]
        T = len(t.soc_raw)
        soc_obs = t.masks[:, _SOC_IDX]
        speed_obs = t.masks[:, _SPEED_IDX]
        soc_cum = np.concatenate(([0.0], np.cumsum(soc_obs)))
        speed_cum = np.concatenate(([0.0], np.cumsum(speed_obs)))
        for s in range(0, T - L - H + 1, stride):
            anchor = s + L - 1
            if np.isnan(t.soc_raw[anchor]) or np.isnan(t.soc_raw[anchor + H]):
                continue
            if (soc_cum[s + L] - soc_cum[s]) / L < config.MIN_SOC_COV:
                continue
            if (speed_cum[s + L] - speed_cum[s]) / L < config.MIN_SPEED_COV:
                continue
            windows.append((key, s))
    return windows


class TripWindowDataset(Dataset):
    """Slices windows on the fly from the in-memory trip arrays.

    x: (INPUT_LEN, 15) = 7 scaled features + scaled accel + 7 missingness flags
    y: standardized DeltaSoC (divide-by-train-std); y_pp = raw pp value
    """

    def __init__(self, trips: dict, windows: list, target_std: float):
        self.trips = trips
        self.windows = windows
        self.target_std = target_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        key, s = self.windows[i]
        t = self.trips[key]
        L, H = config.INPUT_LEN, config.HORIZON
        anchor = s + L - 1
        x = np.concatenate(
            [t.values[s:s + L], t.accel[s:s + L, None], t.masks[s:s + L]], axis=1
        )
        y_pp = t.soc_raw[anchor + H] - t.soc_raw[anchor]
        return (
            torch.from_numpy(x),
            torch.tensor(t.vehicle_idx, dtype=torch.long),
            torch.tensor(y_pp / self.target_std, dtype=torch.float32),
        )


# --- Entry point ----------------------------------------------------------------

def build_datasets(max_trips: int | None = None, scaler: dict | None = None):
    """Full pipeline. Returns (datasets_dict, scaler, index).

    Pass a saved ``scaler`` (from a checkpoint) to reuse train stats at
    evaluation time; otherwise stats are fit on the train split.
    """
    index = load_split_index(max_trips)
    trips = load_trips(index)
    keys = {
        name: [(r.vehicle_id, r.trip_id) for r in index[index.split == name].itertuples()]
        for name in ("train", "val", "test")
    }
    windows = {name: build_window_index(trips, keys[name]) for name in keys}
    if scaler is None:
        scaler = fit_scaler(trips, keys["train"], windows["train"])
    finalize_trips(trips, scaler)
    datasets = {
        name: TripWindowDataset(trips, windows[name], scaler["target_std"])
        for name in keys
    }
    return datasets, scaler, index


def make_dataloaders(datasets: dict, batch_size: int = config.BATCH_SIZE) -> dict:
    return {
        name: DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=0
        )
        for name, ds in datasets.items()
    }


if __name__ == "__main__":
    import sys

    max_trips = int(sys.argv[1]) if len(sys.argv) > 1 else None
    datasets, scaler, index = build_datasets(max_trips)

    print("Trips per split:", index.split.value_counts().to_dict())
    print(
        "Trips per split per vehicle:\n",
        index.groupby(["split", "vehicle_id"]).size().unstack(fill_value=0),
    )
    for name, ds in datasets.items():
        print(f"{name}: {len(ds)} windows from {len(set(k for k, _ in ds.windows))} trips")

    total_bytes = sum(
        t.values.nbytes + t.masks.nbytes + t.accel.nbytes + t.soc_raw.nbytes
        for t in datasets["train"].trips.values()
    )
    print(f"In-memory footprint: {total_bytes / 1e6:.0f} MB")
    print(f"Scaler target_std: {scaler['target_std']:.3f} pp")

    y_pp = np.array(
        [datasets["train"].trips[k].soc_raw[s + config.INPUT_LEN - 1 + config.HORIZON]
         - datasets["train"].trips[k].soc_raw[s + config.INPUT_LEN - 1]
         for k, s in datasets["train"].windows]
    )
    if y_pp.size:
        print(
            "train DeltaSoC pp: mean {:.3f}  std {:.3f}  p5 {:.2f}  p95 {:.2f}".format(
                y_pp.mean(), y_pp.std(), *np.percentile(y_pp, [5, 95])
            )
        )

    loaders = make_dataloaders(datasets)
    x, veh, y = next(iter(loaders["train"]))
    print("batch shapes:", tuple(x.shape), tuple(veh.shape), tuple(y.shape))
    assert not torch.isnan(x).any(), "NaNs in inputs"
    assert not torch.isnan(y).any(), "NaNs in targets"
    print("no NaNs in batch OK")
