"""
Experiment 2: Photon-loss robustness in a 50-stock GBS StatArb portfolio.

Evaluates GBS Boost, GBS Roots, QIC-GBS over loss rates 0..0.9.
Compares with Spectral, SPONGE, and random clustering baselines.

ASSUMPTION: Displacement compensation is approximated by scaling A_eff = eta*A
and preserving n_bar (paper states exact displacement vectors require released code).
ASSUMPTION: "Physical loss" modelled via A_eff = eta*A (paper Appendix approximation).

Paper reference: Section IV.B (pp. 7-8), Appendix G.
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
    EXP2_GBS_RUNS, EXP2_CLASSICAL_RUNS,
    EXP2_LOSS_RATES, RESULTS_DIR,
    GBS_BOOST_BUDGET, GBS_ROOTS_BUDGET, GBS_QIC_BUDGET,
)
from src.data_loader import download_prices, download_fama_french, get_sp500_tickers_n
from src.backtester import run_multiple_seeds
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster
from src.sponge import sponge_cluster

N_STOCKS = 50
YEAR = 2020
LOOKBACK_START = "2018-01-01"
DATA_END = "2021-01-10"
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_clusterer_loss(method: str, loss: float, seed: int):
    if method == "gbs_boost":
        return lambda A: gbs_boost_cluster(A, budget_factor=GBS_BOOST_BUDGET, loss=loss, seed=seed)
    elif method == "gbs_roots":
        return lambda A: gbs_roots_cluster(A, budget_factor=GBS_ROOTS_BUDGET, loss=loss, seed=seed)
    elif method == "qic_gbs":
        return lambda A: qic_gbs_cluster(A, budget_factor=GBS_QIC_BUDGET, loss=loss, seed=seed)
    elif method == "spectral":
        def fn(A):
            N = A.shape[0]
            K = max(2, min(8, N // 7))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return spectral_cluster(C, K, seed=seed)
        return fn
    elif method == "sponge":
        def fn(A):
            N = A.shape[0]
            K = max(2, min(8, N // 7))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return sponge_cluster(C, K, seed=seed)
        return fn
    elif method == "random":
        def fn(A):
            N = A.shape[0]
            K = max(2, min(8, N // 7))
            rng = np.random.default_rng(seed)
            labels = rng.integers(0, K, size=N)
            return [list(np.where(labels == k)[0]) for k in range(K) if (labels == k).any()]
        return fn
    else:
        raise ValueError(f"Unknown method: {method}")


def run_exp2():
    print("=" * 70)
    print("EXP 2: Photon-loss robustness (50 stocks, 2020)")
    print("=" * 70)

    tickers = get_sp500_tickers_n(N_STOCKS)
    prices = download_prices(tickers, LOOKBACK_START, DATA_END)
    valid = prices.dropna(thresh=int(0.9 * len(prices)), axis=1).ffill()
    available = [t for t in tickers if t in valid.columns][:N_STOCKS]
    prices = valid[available]
    print(f"  Using {len(prices.columns)} stocks")

    ff = download_fama_french()
    mkt = ff["Mkt-RF"]

    methods = ["gbs_boost", "gbs_roots", "qic_gbs", "spectral", "sponge", "random"]
    run_counts = {
        "gbs_boost": EXP2_GBS_RUNS, "gbs_roots": EXP2_GBS_RUNS,
        "qic_gbs": EXP2_GBS_RUNS, "spectral": EXP2_CLASSICAL_RUNS,
        "sponge": EXP2_CLASSICAL_RUNS, "random": EXP2_CLASSICAL_RUNS,
    }

    all_results = {}  # method -> {loss_rate -> agg}

    for method in methods:
        all_results[method] = {}
        n_runs = run_counts[method]
        for lr in EXP2_LOSS_RATES:
            print(f"  {method:12s} loss={lr:.1f}...", end=" ", flush=True)
            t0 = time.perf_counter()
            agg = run_multiple_seeds(
                prices, mkt,
                clusterer_factory=lambda seed, m=method, l=lr: make_clusterer_loss(m, l, seed),
                n_runs=n_runs,
                year=YEAR,
            )
            print(f"return={agg['mean_total_return']:.3f} ({time.perf_counter()-t0:.1f}s)")
            all_results[method][lr] = agg

    # Build results DataFrame
    rows = []
    for method in methods:
        for lr in EXP2_LOSS_RATES:
            agg = all_results[method][lr]
            rows.append({
                "Method": method,
                "Loss Rate": lr,
                "Mean Return": agg["mean_total_return"],
                "Std Return": agg["std_total_return"],
                "Mean Sharpe": agg["mean_sharpe"],
                "Mean WD": agg["mean_wd"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp2_results.csv"), index=False)

    # Plot total return vs loss rate
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {
        "gbs_boost": "#2ca02c", "gbs_roots": "#d62728",
        "qic_gbs": "#9467bd", "spectral": "#1f77b4",
        "sponge": "#ff7f0e", "random": "#7f7f7f",
    }
    labels = {
        "gbs_boost": "GBS Boost", "gbs_roots": "GBS Roots",
        "qic_gbs": "QIC-GBS", "spectral": "Spectral",
        "sponge": "SPONGE", "random": "Random",
    }

    for method in methods:
        returns = [all_results[method][lr]["mean_total_return"] for lr in EXP2_LOSS_RATES]
        axes[0].plot(EXP2_LOSS_RATES, returns, "o-",
                     label=labels[method], color=colors[method])

    axes[0].set_title("EXP2: Total Return vs Photon Loss Rate (50 stocks)")
    axes[0].set_xlabel("Photon Loss Rate")
    axes[0].set_ylabel("Mean Total Return")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for method in methods:
        wds = [all_results[method][lr]["mean_wd"] for lr in EXP2_LOSS_RATES]
        axes[1].plot(EXP2_LOSS_RATES, wds, "o-",
                     label=labels[method], color=colors[method])

    axes[1].set_title("EXP2: Weighted Density vs Photon Loss Rate")
    axes[1].set_xlabel("Photon Loss Rate")
    axes[1].set_ylabel("Mean Weighted Density")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_loss_sweep.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR}/exp2_loss_sweep.png")
    return all_results


if __name__ == "__main__":
    run_exp2()
