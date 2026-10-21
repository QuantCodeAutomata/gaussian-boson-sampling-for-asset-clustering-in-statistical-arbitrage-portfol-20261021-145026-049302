# Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios

Replication of experiments from the paper:
> *Gaussian Boson Sampling for Asset Clustering in Statistical Arbitrage Portfolios*

## Overview

This repository implements a market-neutral statistical-arbitrage strategy where stock universes
are clustered using Gaussian Boson Sampling (GBS) and compared against classical spectral methods.
The strategy shorts previous winners and buys previous losers within each cluster, holding for
at most 3 trading days with a 2% stop-win threshold.

## Experiments

| ID | Title | Universe | Year |
|----|-------|----------|------|
| exp_1 | Lossless 100-stock benchmark | 100 stocks | 2020 |
| exp_2 | Photon-loss robustness | 50 stocks | 2020 |
| exp_3 | Cross-regime comparison | 50 stocks | 2008/2017/2020/2022 |
| exp_4 | Sample budget scaling | 30 stocks | 2020 |
| exp_5 | Universe scaling (N=12–50) | 12–50 stocks | 2020 |

## Clustering Methods

- **GBS Boost**: Recursive max-density sample extraction (H = 0.7 N ln N)
- **GBS Roots**: Recursive max-WD sample with greedy union (H = 0.6 N ln N)
- **QIC-GBS**: Classical edge-Poisson approximation + Boost extraction (H = N ln N)
- **Spectral**: Normalised spectral clustering with dynamic K from eigenvalue spacings
- **Symmetric SPONGE**: Signed Laplacian generalised eigenproblem

## GBS Sampling Backend

By default, this repository uses a **fast numpy approximation** of GBS threshold sampling
(independent edge-Poisson model, equivalent to the QIC-GBS classical approximation).
This avoids the ~5-10 minute numba JIT compilation required by `thewalrus`.

To use real thewalrus GBS simulation:
```bash
export USE_THEWALRUS=1
```

## Repository Structure

```
├── src/          # Core algorithms and utilities
├── exp/          # Experiment scripts (exp_1 through exp_5)
├── tests/        # pytest unit tests
├── data/         # Cached price data (gitignored)
├── results/      # Output metrics, plots, RESULTS.md
├── reproduce.sh  # Entry-point: runs all experiments
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all experiments (fast mode: ~3 runs per method)
bash reproduce.sh --fast

# Run full experiments (paper-scale run counts)
bash reproduce.sh

# Run individual experiment
python exp/exp_1_lossless_100.py
```

## Methodology Parameters (confirmed from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Beta window | 60 days | Paper methodology step 2 |
| Correlation window | 5 days | Paper step 3 |
| K estimation window | 20 days | Paper step 9 |
| K bootstrap samples | B=1000 | Paper step 9 |
| K significance level | α=0.05 | Paper step 9 |
| Holding period | 3 days | Paper step 11 |
| Stop-win threshold | 2% | Paper step 11 |
| GBS Boost budget | F=0.7 | Paper step 5 |
| GBS Roots budget | F=0.6 | Paper step 5 |
| QIC-GBS budget | F=1.0 | Paper step 5 |
| Scaling exp budget | F=5.0 | Paper Appendix J |
| Transaction costs | 0 | Paper step 12 |

## Key Assumptions

1. **GBS Sampling**: Fast edge-sampling approximation (QIC-GBS-style) used by default
2. **Price Data**: yfinance dividend-adjusted prices (CRSP unavailable without WRDS)
3. **Risk-free Rate**: 0% (paper does not specify)
4. **Annualisation**: √252 trading days (paper does not specify)
5. **RMT Cleaning**: Disabled (paper ordering of operations is ambiguous)
6. **Displacement compensation**: A_eff = η·A approximation
7. **Sector allocation**: Manual 11-GICS-sector stratification using standard S&P 500 tickers

## Expected Results (Paper Table I, 100 stocks, 2020)

| Method | Mean Return | Std | Sharpe |
|--------|-------------|-----|--------|
| Spectral | 0.200 | 0.051 | 2.035 |
| SPONGE | 0.214 | 0.064 | 2.242 |
| GBS Boost | 0.239 | 0.035 | 2.050 |
| GBS Roots | 0.236 | 0.031 | 2.248 |
| QIC-GBS | 0.195 | 0.099 | 1.781 |

Paper also reports Welch t-test p=0.023 for GBS Roots vs SPONGE terminal returns.

## Testing

```bash
python -m pytest tests/ -v
```
