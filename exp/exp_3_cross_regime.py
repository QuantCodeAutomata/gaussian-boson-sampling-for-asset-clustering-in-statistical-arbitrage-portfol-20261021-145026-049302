"""
Experiment 3: Cross-regime comparison of lossless 50-stock portfolios.

Tests 2008, 2017, 2020, 2022 market regimes. Compares GBS Boost/Roots,
QIC-GBS, Spectral, SPONGE with a 1/N buy-and-hold benchmark.

Paper reference: Section IV.C (pp. 8-9), Figure 5.
ASSUMPTION: yfinance total-return approximation for all regime years.
ASSUMPTION: Equal-weight B&H uses same 50 tickers; no rebalancing.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    EXP3_GBS_RUNS, EXP3_CLASSICAL_RUNS, EXP3_YEARS,
    RESULTS_DIR, GBS_BOOST_BUDGET, GBS_ROOTS_BUDGET, GBS_QIC_BUDGET,
)
from src.data_loader import download_prices, download_fama_french, get_sp500_tickers_n
from src.backtester import run_multiple_seeds
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster
from src.sponge import sponge_cluster
from src.returns import compute_total_returns

N_STOCKS = 50
os.makedirs(RESULTS_DIR, exist_ok=True)


def buy_and_hold_return(prices: pd.DataFrame, year: int) -> float:
    """Compute equal-weight B&H return for the backtest year."""
    year_prices = prices[prices.index.year == year]
    if len(year_prices) < 2:
        return 0.0
    daily_returns = year_prices.pct_change().iloc[1:]
    eq_returns = daily_returns.mean(axis=1)
    return float((1 + eq_returns).prod() - 1)


def make_clusterer(method: str, seed: int):
    if method == "gbs_boost":
        return lambda A: gbs_boost_cluster(A, budget_factor=GBS_BOOST_BUDGET, loss=0.0, seed=seed)
    elif method == "gbs_roots":
        return lambda A: gbs_roots_cluster(A, budget_factor=GBS_ROOTS_BUDGET, loss=0.0, seed=seed)
    elif method == "qic_gbs":
        return lambda A: qic_gbs_cluster(A, budget_factor=GBS_QIC_BUDGET, loss=0.0, seed=seed)
    elif method == "spectral":
        def fn(A):
            N = A.shape[0]; K = max(2, min(8, N // 7))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return spectral_cluster(C, K, seed=seed)
        return fn
    elif method == "sponge":
        def fn(A):
            N = A.shape[0]; K = max(2, min(8, N // 7))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return sponge_cluster(C, K, seed=seed)
        return fn


def run_exp3():
    print("=" * 70)
    print("EXP 3: Cross-regime comparison (50 stocks, 4 years)")
    print("=" * 70)

    methods = {
        "Spectral": ("spectral", EXP3_CLASSICAL_RUNS),
        "SPONGE": ("sponge", EXP3_CLASSICAL_RUNS),
        "GBS Boost": ("gbs_boost", EXP3_GBS_RUNS),
        "GBS Roots": ("gbs_roots", EXP3_GBS_RUNS),
        "QIC-GBS": ("qic_gbs", EXP3_GBS_RUNS),
    }

    rows = []
    regime_results = {}

    for year in EXP3_YEARS:
        print(f"\n--- Year {year} ---")
        lookback_start = f"{year-2}-01-01"
        data_end = f"{year+1}-01-15"

        tickers = get_sp500_tickers_n(N_STOCKS)
        prices = download_prices(tickers, lookback_start, data_end)
        valid = prices.dropna(thresh=int(0.9*len(prices)), axis=1).ffill()
        available = [t for t in tickers if t in valid.columns][:N_STOCKS]
        prices = valid[available]

        ff = download_fama_french()
        mkt = ff["Mkt-RF"]

        # Buy-and-hold benchmark
        bh_return = buy_and_hold_return(prices, year)
        print(f"  Buy-and-hold: {bh_return:.3f}")

        regime_results[year] = {"Buy-and-Hold": {"mean_total_return": bh_return, "mean_sharpe": bh_return / 0.2}}

        for method_name, (method_key, n_runs) in methods.items():
            print(f"  {method_name:12s}...", end=" ", flush=True)
            t0 = time.perf_counter()
            agg = run_multiple_seeds(
                prices, mkt,
                clusterer_factory=lambda seed, mk=method_key: make_clusterer(mk, seed),
                n_runs=n_runs,
                year=year,
            )
            regime_results[year][method_name] = agg
            print(f"return={agg['mean_total_return']:.3f} ({time.perf_counter()-t0:.1f}s)")

            rows.append({
                "Year": year,
                "Method": method_name,
                "Mean Return": agg["mean_total_return"],
                "Std Return": agg["std_total_return"],
                "Mean Sharpe": agg["mean_sharpe"],
            })

        rows.append({
            "Year": year, "Method": "Buy-and-Hold",
            "Mean Return": bh_return, "Std Return": 0.0, "Mean Sharpe": 0.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp3_results.csv"), index=False)

    # Plot grouped bar chart
    method_names = list(methods.keys()) + ["Buy-and-Hold"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"]
    x = np.arange(len(EXP3_YEARS))
    width = 0.14

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, mname in enumerate(method_names):
        returns = []
        for year in EXP3_YEARS:
            yr_data = regime_results[year].get(mname)
            if yr_data:
                returns.append(yr_data.get("mean_total_return", 0.0))
            else:
                returns.append(0.0)
        offset = (i - len(method_names)/2 + 0.5) * width
        ax.bar(x + offset, returns, width, label=mname, color=colors[i % len(colors)], alpha=0.85)

    ax.set_title("EXP3: Total Returns by Regime and Method")
    ax.set_xlabel("Market Regime Year")
    ax.set_ylabel("Total Return")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in EXP3_YEARS])
    ax.legend(fontsize=8, loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_regimes.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR}/exp3_regimes.png")

    # Paper expected values
    paper_expected = {
        2008: {"Buy-and-Hold": -0.48, "QIC-GBS": 0.45, "SPONGE": 0.30, "GBS Boost": 0.38, "GBS Roots": 0.39, "Spectral": 0.17},
        2017: {"Buy-and-Hold": 0.29, "QIC-GBS": 0.01, "SPONGE": 0.08, "GBS Boost": 0.06, "GBS Roots": 0.06, "Spectral": 0.11},
        2020: {"Buy-and-Hold": 0.50, "QIC-GBS": 0.16, "SPONGE": 0.21, "GBS Boost": 0.31, "GBS Roots": 0.23, "Spectral": 0.22},
        2022: {"Buy-and-Hold": 0.00, "QIC-GBS": 0.12, "SPONGE": 0.13, "GBS Boost": 0.08, "GBS Roots": 0.14, "Spectral": 0.17},
    }
    pd.DataFrame(paper_expected).to_csv(os.path.join(RESULTS_DIR, "exp3_paper_expected.csv"))

    return regime_results


if __name__ == "__main__":
    run_exp3()
