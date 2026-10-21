"""
Configuration and hyperparameters for GBS StatArb experiments.

All confirmed parameter values come from the paper. Assumed values are
explicitly documented. ASSUMPTION comments indicate non-confirmed values.
"""
import os
import math

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PRICE_CACHE_FILE = os.path.join(DATA_DIR, "prices_cache.parquet")
FF_CACHE_FILE = os.path.join(DATA_DIR, "ff_factors.parquet")

# ---------------------------------------------------------------------------
# GBS sampling parameters (confirmed from paper)
# ---------------------------------------------------------------------------
GBS_BOOST_BUDGET: float = 0.7   # H = floor(0.7 * N * ln(N)) [confirmed]
GBS_ROOTS_BUDGET: float = 0.6   # H = floor(0.6 * N * ln(N)) [confirmed]
GBS_QIC_BUDGET: float = 1.0     # H = floor(1.0 * N * ln(N)) [confirmed]
GBS_SCALING_BUDGET: float = 5.0 # H = floor(5.0 * N * ln(N)) [exp_5, confirmed]

# ---------------------------------------------------------------------------
# Statistical arbitrage strategy parameters (confirmed from paper)
# ---------------------------------------------------------------------------
BETA_WINDOW: int = 60            # rolling beta estimation window [confirmed]
CORRELATION_WINDOW: int = 5      # residual correlation look-back [confirmed]
K_WINDOW: int = 20               # dynamic K estimation window [confirmed]
FORMATION_WINDOW: int = 5        # winner/loser classification window [confirmed = CORRELATION_WINDOW]
HOLD_DAYS: int = 3               # maximum holding period [confirmed]
STOP_WIN_THRESHOLD: float = 0.02 # cumulative stop-win threshold [confirmed]
P_THRESHOLD: float = 0.0         # winner/loser deviation threshold [confirmed: p=0]

# ---------------------------------------------------------------------------
# Classical clustering parameters (confirmed from paper)
# ---------------------------------------------------------------------------
K_ALPHA: float = 0.05     # eigenvalue spacing significance level [confirmed]
K_PERM_SAMPLES: int = 1000  # permutation null replicates [confirmed]
MIN_CLUSTERS: int = 2       # minimum number of clusters
MAX_CLUSTERS: int = 15      # maximum clusters [ASSUMPTION]
KMEANS_SEED: int = 42       # K-means++ base seed [ASSUMPTION]

# SPONGE regularisation
SPONGE_TAU: float = 0.1  # ASSUMPTION: regularisation to avoid singular Laplacians

# ---------------------------------------------------------------------------
# Performance statistics (ASSUMPTION: standard conventions)
# ---------------------------------------------------------------------------
ANNUAL_FACTOR: int = 252     # trading days per year [ASSUMPTION]
RISK_FREE_RATE: float = 0.0  # ASSUMPTION: zero risk-free rate for StatArb

# ---------------------------------------------------------------------------
# Experiment run counts
# ---------------------------------------------------------------------------
# exp_1 (100 stocks): GBS ~20 runs, classical ~100 runs [confirmed: "approximately"]
EXP1_GBS_RUNS: int = int(os.environ.get("EXP1_GBS_RUNS", "5"))
EXP1_CLASSICAL_RUNS: int = int(os.environ.get("EXP1_CLASSICAL_RUNS", "5"))

# exp_2 (50 stocks, loss sweep): 50 quantum, 100-200 classical
EXP2_GBS_RUNS: int = int(os.environ.get("EXP2_GBS_RUNS", "3"))
EXP2_CLASSICAL_RUNS: int = int(os.environ.get("EXP2_CLASSICAL_RUNS", "3"))

# exp_3 (50 stocks, cross-regime): from released code
EXP3_GBS_RUNS: int = int(os.environ.get("EXP3_GBS_RUNS", "5"))
EXP3_CLASSICAL_RUNS: int = int(os.environ.get("EXP3_CLASSICAL_RUNS", "5"))

# exp_4 (30 stocks, sample budget): 100/50/10 for low/mid/high F
EXP4_RUNS_LOW_F: int = int(os.environ.get("EXP4_RUNS_LOW_F", "5"))
EXP4_RUNS_MID_F: int = int(os.environ.get("EXP4_RUNS_MID_F", "5"))
EXP4_RUNS_HIGH_F: int = int(os.environ.get("EXP4_RUNS_HIGH_F", "3"))

# exp_5 (universe scaling 20-30, 12-50): 15-120 runs
EXP5_GBS_RUNS: int = int(os.environ.get("EXP5_GBS_RUNS", "5"))
EXP5_CLASSICAL_RUNS: int = int(os.environ.get("EXP5_CLASSICAL_RUNS", "5"))

# Universe sizes for exp_5
EXP5_SIZES_FINE = list(range(20, 31))       # 20 to 30 inclusive
EXP5_SIZES_BROAD = [12, 16, 20, 25, 30, 35, 40, 45, 50]  # ASSUMPTION: equal spacing

# Sample budget factors for exp_4
EXP4_F_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]

# Photon loss rates for exp_2
EXP2_LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Years for exp_3
EXP3_YEARS = [2008, 2017, 2020, 2022]
