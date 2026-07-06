"""
Configuration for the SoC-forecasting training stage (LSTM-Transformer).

All tunables live here so experiments can be adjusted without touching the
training logic. Consumes the per-trip 1 Hz parquets produced by
``preprocessing/02_trip_timeseries/build_trips.py``.
"""

# --- Paths -------------------------------------------------------------------
# Resolved relative to the repository root at runtime (see dataset.py).
TRIPS_DIR = "data/processed/trips"
INDEX_PATH = "data/processed/trips_index.parquet"
RUNS_DIR = "runs"  # per-run checkpoints / scalers / metrics / plots

# --- Vehicles ----------------------------------------------------------------
# Fixed vocabulary -> embedding index. Order must stay stable across runs so
# saved checkpoints keep meaning.
VEHICLE_VOCAB = {"ID1": 0, "ID2": 1, "CUP1": 2, "CUP2": 3, "CUP3": 4, "CUP4": 5, "CUP5": 6}

# --- Features ----------------------------------------------------------------
# Columns read from each trip parquet (order defines channel order).
FEATURES = ["speed", "soc", "pack_voltage", "aux_power", "batt_inlet", "batt_outlet", "ambient"]
TARGET = "soc"
# Derived channel: acceleration = 1 Hz diff of speed (km/h per s).
# Model input = FEATURES + accel + one missingness flag per FEATURE -> 15 channels.
N_INPUT_CHANNELS = len(FEATURES) + 1 + len(FEATURES)

# --- Split -------------------------------------------------------------------
# Chronological at trip level: trips sorted by start_time, whole trips assigned
# to a split at these quantile cuts (train / val / test).
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test = remainder

# --- Windowing ---------------------------------------------------------------
INPUT_LEN = 300      # input window length in steps (1 Hz -> 5 min)
HORIZON = 300        # forecast horizon in seconds; target = soc[t0+H] - soc[t0]
STRIDE = 30          # window start spacing within a trip (seconds)
# Validity filters (windows failing these are dropped at index-build time):
MIN_SOC_COV = 0.90   # min fraction of observed (non-imputed) SoC inside input window
MIN_SPEED_COV = 0.90 # same for speed
# SoC must be *observed* at t0 and t0+H; the target is never computed from
# imputed values.

# --- Model -------------------------------------------------------------------
D_MODEL = 64         # input projection / LSTM hidden / transformer width
LSTM_LAYERS = 1
N_HEADS = 4
N_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
VEHICLE_EMB_DIM = 8

# --- Training ----------------------------------------------------------------
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5      # ReduceLROnPlateau patience (epochs), factor 0.5
GRAD_CLIP = 1.0
SEED = 42
