"""
Configuration for the per-trip time-series builder (build_trips.py).

All tunables live here so the pipeline can be adjusted without touching the
build logic. Signals are referenced by their ``value_id`` (see
``data/value_overview.csv``).
"""

# --- Paths -------------------------------------------------------------------
# Resolved relative to the repository root at runtime (see build_trips.py).
UDS_DATA_DIR = "data/uds_data"
VALUE_OVERVIEW_CSV = "data/value_overview.csv"
OUTPUT_DIR = "data/processed/trips"
INDEX_PATH = "data/processed/trips_index.parquet"

# --- Vehicles ----------------------------------------------------------------
# Parquet files are named ``{vehicle}.parquet``. None => all found in UDS_DATA_DIR.
VEHICLES = ["ID1", "ID2", "CUP1", "CUP2", "CUP3", "CUP4", "CUP5"]

# --- Feature set (value_ids) -------------------------------------------------
# Order here defines the column order of every output trip table.
# name -> value_id. The TARGET (SoC) is included as a feature too.
#
# NOTE on coverage: speed / soc / pack_voltage are present in ~99% of driving
# trips. aux_power is intermittent (~88%). ambient (15) and the min/max battery
# temps (1208/1209) are logged mostly while parked, so they are almost always
# absent during driving. batt_inlet / batt_outlet (1272/1273) are the usable
# battery temperatures but only exist from ~Aug 2021 onward. We keep all of
# these and simply leave the missing values as NaN (see KEEP_ALL_TRIPS below);
# per-feature coverage is recorded in the trips index so trips/features can be
# filtered later.
FEATURES = {
    "speed": 4,           # vehicle speed (km/h), 200 ms
    "soc": 900,           # State of Charge (%), 5 s        <- forecast target
    "pack_voltage": 1200,  # HV pack voltage (V), 200 ms
    "aux_power": 56,      # HV auxiliary power (W), 1 s
    "batt_inlet": 1272,   # HV battery pack inlet temp (degC), 5 s  (from ~Aug 2021)
    "batt_outlet": 1273,  # HV battery pack outlet temp (degC), 5 s (from ~Aug 2021)
    "ambient": 15,        # ambient air temperature (degC), 10 s  (sparse in driving)
}
TARGET = "soc"

# Signals faster than the grid are averaged into each bin; the rest are
# interpolated. Anything not listed here defaults to interpolation.
DOWNSAMPLE_BY_MEAN = {"speed", "pack_voltage", "aux_power"}

# --- Resampling --------------------------------------------------------------
RESAMPLE_FREQ = "1s"     # common time grid (pandas offset alias)
MAX_INTERP_GAP_S = 20    # do not interpolate across gaps longer than this (seconds)

# --- Trip segmentation & filtering ------------------------------------------
GAP_SECONDS = 120        # a gap in the speed signal larger than this splits trips
MIN_TRIP_MIN = 5         # drop trips shorter than this (minutes)
MAX_TRIP_MIN = 90        # drop trips longer than this (minutes)
MIN_SPEED_KMH = 5.0      # a trip must reach at least this speed (real motion)

# Missing-data policy. When True (default) every driving trip is kept and absent
# / sparse signals are left as NaN; per-feature coverage is written to the index
# so filtering can happen later. Set False to instead drop any trip whose
# features fall below MIN_COVERAGE (stricter, smaller dataset).
KEEP_ALL_TRIPS = True
MIN_COVERAGE = 0.90      # only used when KEEP_ALL_TRIPS is False
