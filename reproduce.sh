#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — Run all GBS StatArb experiments
#
# Usage: bash reproduce.sh [--fast]
#   --fast: Use minimal run counts for quick validation (default for CI)
#
# Environment variables for run counts (override defaults in config.py):
#   EXP1_GBS_RUNS, EXP1_CLASSICAL_RUNS   (default: 5 each)
#   EXP2_GBS_RUNS, EXP2_CLASSICAL_RUNS   (default: 3 each)
#   EXP3_GBS_RUNS, EXP3_CLASSICAL_RUNS   (default: 5 each)
#   EXP4_RUNS_LOW_F, EXP4_RUNS_MID_F, EXP4_RUNS_HIGH_F
#   EXP5_GBS_RUNS, EXP5_CLASSICAL_RUNS
#
# Set USE_THEWALRUS=1 to use real thewalrus GBS simulation (requires
# ~5-10 minute numba JIT warm-up; otherwise uses fast numpy approximation).
#
# Paper: "Gaussian Boson Sampling for Asset Clustering in Statistical
#         Arbitrage Portfolios"
# =============================================================================

set -euo pipefail

FAST_MODE=0
for arg in "$@"; do
    if [ "$arg" = "--fast" ]; then
        FAST_MODE=1
    fi
done

if [ "$FAST_MODE" = "1" ]; then
    echo "Running in FAST mode (minimal run counts)..."
    export EXP1_GBS_RUNS=3
    export EXP1_CLASSICAL_RUNS=3
    export EXP2_GBS_RUNS=2
    export EXP2_CLASSICAL_RUNS=2
    export EXP3_GBS_RUNS=3
    export EXP3_CLASSICAL_RUNS=3
    export EXP4_RUNS_LOW_F=3
    export EXP4_RUNS_MID_F=3
    export EXP4_RUNS_HIGH_F=2
    export EXP5_GBS_RUNS=3
    export EXP5_CLASSICAL_RUNS=3
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "======================================================================"
echo "GBS Statistical Arbitrage — Paper Replication"
echo "======================================================================"
echo "Working directory: $ROOT"
echo ""

# Install/check dependencies
echo "Checking dependencies..."
python -c "import numpy, pandas, scipy, sklearn, matplotlib, yfinance" && echo "Core deps OK"

# Run tests
echo ""
echo "Running unit tests..."
python -m pytest tests/ -v --tb=short -q 2>&1 | tail -20 || echo "Some tests failed (continuing)"

# Create results dir
mkdir -p results

# Run experiments
echo ""
echo "======================================================================"
echo "EXP 1: Lossless 100-stock benchmark (2020)"
echo "======================================================================"
python exp/exp_1_lossless_100.py

echo ""
echo "======================================================================"
echo "EXP 2: Photon-loss robustness (50 stocks)"
echo "======================================================================"
python exp/exp_2_photon_loss.py

echo ""
echo "======================================================================"
echo "EXP 3: Cross-regime comparison (50 stocks)"
echo "======================================================================"
python exp/exp_3_cross_regime.py

echo ""
echo "======================================================================"
echo "EXP 4: Sample budget scaling (30 stocks)"
echo "======================================================================"
python exp/exp_4_sample_budget.py

echo ""
echo "======================================================================"
echo "EXP 5: Universe scaling (N=12 to 50)"
echo "======================================================================"
python exp/exp_5_universe_scaling.py

# Compile RESULTS.md
echo ""
echo "Generating RESULTS.md..."
python - <<'PYEOF'
import os, pandas as pd, json
from pathlib import Path

RESULTS = Path("results")
lines = [
    "# Results Summary",
    "",
    "Paper: *Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios*",
    "",
    "> **GBS Backend**: Fast numpy approximation (QIC-GBS-style edge sampling).",
    "> Set `USE_THEWALRUS=1` to enable real thewalrus GBS simulation.",
    "",
]

for exp_id, csv_name, title in [
    ("1", "exp1_results.csv", "EXP1: Lossless 100-stock benchmark (2020)"),
    ("2", "exp2_results.csv", "EXP2: Photon-loss sweep (50 stocks)"),
    ("3", "exp3_results.csv", "EXP3: Cross-regime (50 stocks)"),
    ("4", "exp4_results.csv", "EXP4: Sample budget scaling (30 stocks)"),
    ("5", "exp5_results.csv", "EXP5: Universe scaling (N=12-50)"),
]:
    lines.append(f"## {title}")
    csv_path = RESULTS / csv_name
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        lines.append("")
        lines.append(df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
    else:
        lines.append("")
        lines.append(f"*Results file `{csv_name}` not found.*")
        lines.append("")

# Welch test
welch_path = RESULTS / "exp1_welch_test.csv"
if welch_path.exists():
    lines.append("## EXP1: Welch t-test (GBS Roots vs SPONGE terminal returns)")
    lines.append("")
    df_w = pd.read_csv(welch_path)
    lines.append(df_w.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("Paper reports: p=0.023 for terminal returns, p=0.949 for Sharpe.")
    lines.append("")

lines += [
    "## Methodology Deviations and Assumptions",
    "",
    "| # | Item | Status | Notes |",
    "|---|------|--------|-------|",
    "| 1 | GBS Sampling | ASSUMED | Fast numpy edge-sampling approximation (QIC-GBS-style). Real thewalrus GBS requires `USE_THEWALRUS=1`. |",
    "| 2 | Stock universe | ASSUMED | 100-stock stratified list from major S&P 500 sectors; exact CRSP PERMNOs unavailable without WRDS. |",
    "| 3 | Price data | ASSUMED | yfinance dividend-adjusted prices approximate CRSP total returns. |",
    "| 4 | RMT cleaning | MISSING | Not applied (ambiguous ordering in paper; disabled by default). |",
    "| 5 | Dynamic K | IMPLEMENTED | Eigenvalue spacings with B=1000 permutations, alpha=0.05. |",
    "| 6 | Risk-free rate | ASSUMED | Zero (paper does not specify). |",
    "| 7 | Annualisation | ASSUMED | sqrt(252) for Sharpe/Sortino. |",
    "| 8 | Displacement compensation | APPROXIMATED | A_eff = eta*A; exact displacement amplitudes require released code. |",
    "| 9 | GBS Roots queue order | ASSUMED | WD-descending sort (Algorithm 1 ambiguity). |",
    "| 10 | Sector stratification | ASSUMED | Manual assignment of tickers to 11 GICS sectors. |",
    "",
]

with open(RESULTS / "RESULTS.md", "w") as f:
    f.write("\n".join(lines))
print("RESULTS.md written.")
PYEOF

echo ""
echo "======================================================================"
echo "All experiments completed. Results saved to results/"
echo "======================================================================"
ls -la results/
