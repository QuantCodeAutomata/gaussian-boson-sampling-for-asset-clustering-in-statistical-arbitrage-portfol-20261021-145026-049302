"""
Experiment 1: Lossless 100-stock benchmark of GBS clustering vs classical.

Backtest all 5 methods on 100-stock S&P 500 universe during 2020.
Compare: GBS Boost, GBS Roots, QIC-GBS, Spectral, Symmetric SPONGE.

Key metrics: total return, Sharpe, Sortino, weighted density, global
cluster value, consecutive Jaccard stability, Welch t-test.

Paper reference: Section IV.A (pp. 6-7), Appendix A.
ASSUMPTION: yfinance prices approximate CRSP total returns.
ASSUMPTION: Run counts reduced for feasibility; paper uses ~20 GBS, ~100 classical.
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
from scipy import stats

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import (
    EXP1_GBS_RUNS, EXP1_CLASSICAL_RUNS, RESULTS_DIR,
    GBS_BOOST_BUDGET, GBS_ROOTS_BUDGET, GBS_QIC_BUDGET,
)
from src.data_loader import download_prices, download_fama_french, get_sp500_tickers
from src.backtester import run_multiple_seeds, run_backtest
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster, estimate_k_from_spacings
from src.sponge import sponge_cluster
from src.metrics import welch_t_test
from src.returns import compute_total_returns, compute_residual_returns
from src.correlation import compute_correlation_matrix, correlation_to_adjacency


N_STOCKS = 100
YEAR = 2020
LOOKBACK_START = "2018-01-01"  # enough history for 60-day beta
DATA_END = "2021-01-10"
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_clusterer(method: str, seed: int):
    """Factory: return a clusterer function for the given method and seed."""
    def get_k(returns_window: np.ndarray) -> int:
        return max(2, min(10, returns_window.shape[1] // 8))

    if method == "gbs_boost":
        return lambda A: gbs_boost_cluster(A, budget_factor=GBS_BOOST_BUDGET, loss=0.0, seed=seed)
    elif method == "gbs_roots":
        return lambda A: gbs_roots_cluster(A, budget_factor=GBS_ROOTS_BUDGET, loss=0.0, seed=seed)
    elif method == "qic_gbs":
        return lambda A: qic_gbs_cluster(A, budget_factor=GBS_QIC_BUDGET, loss=0.0, seed=seed)
    elif method == "spectral":
        def spectral_fn(A):
            N = A.shape[0]
            K = max(2, min(10, N // 8))
            C = A + np.eye(N)
            np.fill_diagonal(C, 1.0)
            return spectral_cluster(C, K, seed=seed)
        return spectral_fn
    elif method == "sponge":
        def sponge_fn(A):
            N = A.shape[0]
            K = max(2, min(10, N // 8))
            C = A + np.eye(N)
            np.fill_diagonal(C, 1.0)
            return sponge_cluster(C, K, seed=seed)
        return sponge_fn
    else:
        raise ValueError(f"Unknown method: {method}")


def run_exp1():
    print("=" * 70)
    print("EXP 1: Lossless 100-stock benchmark (2020)")
    print("=" * 70)

    # Download data
    print(f"\nDownloading {N_STOCKS} S&P 500 tickers...")
    tickers = get_sp500_tickers(N_STOCKS)
    prices = download_prices(tickers, LOOKBACK_START, DATA_END)

    # Filter to tickers with complete data
    valid = prices.dropna(thresh=int(0.9 * len(prices)), axis=1)
    prices = valid.ffill().dropna(how="all")
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available[:N_STOCKS]]
    print(f"  Using {len(prices.columns)} stocks with sufficient data")

    # Download Fama-French
    print("Downloading Fama-French factors...")
    ff = download_fama_french()
    mkt = ff["Mkt-RF"]

    # Methods and run counts
    methods = {
        "Spectral": ("spectral", EXP1_CLASSICAL_RUNS),
        "SPONGE": ("sponge", EXP1_CLASSICAL_RUNS),
        "GBS Boost": ("gbs_boost", EXP1_GBS_RUNS),
        "GBS Roots": ("gbs_roots", EXP1_GBS_RUNS),
        "QIC-GBS": ("qic_gbs", EXP1_GBS_RUNS),
    }

    results = {}
    for method_name, (method_key, n_runs) in methods.items():
        print(f"\nRunning {method_name} ({n_runs} runs)...")
        t0 = time.perf_counter()

        agg = run_multiple_seeds(
            prices, mkt,
            clusterer_factory=lambda seed, mk=method_key: make_clusterer(mk, seed),
            n_runs=n_runs,
            year=YEAR,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        results[method_name] = agg
        print(f"  Total return: {agg['mean_total_return']:.3f} ± {agg['std_total_return']:.3f}")
        print(f"  Sharpe:       {agg['mean_sharpe']:.3f} ± {agg['std_sharpe']:.3f}")
        print(f"  Runtime: {elapsed:.1f}s")

    # Welch t-test: GBS Roots vs SPONGE terminal returns
    roots_returns = np.array(results["GBS Roots"]["terminal_returns"])
    sponge_returns = np.array(results["SPONGE"]["terminal_returns"])
    if len(roots_returns) > 1 and len(sponge_returns) > 1:
        test_result = welch_t_test(roots_returns, sponge_returns)
        print(f"\nWelch t-test GBS Roots vs SPONGE terminal returns:")
        print(f"  t={test_result['t']:.3f}, df={test_result['df']:.1f}, p={test_result['p_value']:.4f}")
        print(f"  Paper reports: p=0.023")
    else:
        test_result = {"t": 0, "df": 0, "p_value": 1.0, "note": "insufficient runs"}

    # Save results table
    rows = []
    for method_name, agg in results.items():
        rows.append({
            "Method": method_name,
            "Mean Return": agg["mean_total_return"],
            "Std Return": agg["std_total_return"],
            "Mean Sharpe": agg["mean_sharpe"],
            "Std Sharpe": agg["std_sharpe"],
            "Mean Sortino": agg["mean_sortino"],
            "Mean WD": agg["mean_wd"],
            "Mean GCV": agg["mean_gcv"],
            "Mean Jaccard": agg["mean_jaccard"],
            "N Runs": agg["n_runs"],
        })
    df_results = pd.DataFrame(rows)
    df_results.to_csv(os.path.join(RESULTS_DIR, "exp1_results.csv"), index=False)
    print("\nResults table:")
    print(df_results.to_string(index=False))

    # Plot equity curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (method_name, agg) in enumerate(results.items()):
        pnl = agg["combined_pnl"]
        cum = (1 + pnl.sort_index()).cumprod() - 1
        # Downsample for plot
        axes[0].plot(cum.values, label=method_name, alpha=0.7, color=colors[i])

    axes[0].set_title("EXP1: Cumulative Returns (2020, 100 stocks)")
    axes[0].set_xlabel("Trading Day")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Bar chart of mean total returns
    method_names = list(results.keys())
    mean_rets = [results[m]["mean_total_return"] for m in method_names]
    std_rets = [results[m]["std_total_return"] for m in method_names]
    axes[1].bar(method_names, mean_rets, yerr=std_rets, capsize=5,
                color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("EXP1: Mean Total Return by Method")
    axes[1].set_ylabel("Total Return")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "exp1_returns.png")
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {fig_path}")

    # Save Welch test
    pd.DataFrame([test_result]).to_csv(
        os.path.join(RESULTS_DIR, "exp1_welch_test.csv"), index=False
    )

    # Paper expected values for comparison
    paper_expected = {
        "Spectral":   {"mean_return": 0.200, "std_return": 0.051, "sharpe": 2.035},
        "SPONGE":     {"mean_return": 0.214, "std_return": 0.064, "sharpe": 2.242},
        "GBS Boost":  {"mean_return": 0.239, "std_return": 0.035, "sharpe": 2.050},
        "GBS Roots":  {"mean_return": 0.236, "std_return": 0.031, "sharpe": 2.248},
        "QIC-GBS":    {"mean_return": 0.195, "std_return": 0.099, "sharpe": 1.781},
    }

    return {
        "results": results,
        "welch_test": test_result,
        "paper_expected": paper_expected,
        "results_df": df_results,
    }


if __name__ == "__main__":
    run_exp1()
