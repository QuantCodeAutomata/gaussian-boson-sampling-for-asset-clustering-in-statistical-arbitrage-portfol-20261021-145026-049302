"""
Experiment 5: Stock-universe scaling analysis of returns and cluster stability.

Tests N from 12 to 50 (nested subsets). Compares all 5 clustering methods.
Key metrics: median total return, IQR, consecutive Jaccard stability.

Paper reference: Appendix J (pp. 21-23), Figure 9.
ASSUMPTION: H = 5*N*ln(N) per GBS call [confirmed for exp_5 from paper].
ASSUMPTION: Nested universe = first N tickers from sorted 50-ticker list.
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
    EXP5_GBS_RUNS, EXP5_CLASSICAL_RUNS,
    EXP5_SIZES_FINE, EXP5_SIZES_BROAD,
    RESULTS_DIR, GBS_SCALING_BUDGET,
)
from src.data_loader import download_prices, download_fama_french, get_sp500_tickers_n
from src.backtester import run_multiple_seeds
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster
from src.sponge import sponge_cluster

YEAR = 2020
LOOKBACK_START = "2018-01-01"
DATA_END = "2021-01-10"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ASSUMPTION: Paper uses H=5*N*ln(N) but real GBS hardware is far faster than
# our numpy approximation. We use F=0.5 throughout EXP 5 to keep the scaling
# study feasible while preserving the qualitative N-dependence.
F_SCALING = 0.5
F_SCALING_LARGE_N = 0.5
N_LARGE_THRESHOLD = 35


def get_budget_factor(N: int) -> float:
    return F_SCALING if N < N_LARGE_THRESHOLD else F_SCALING_LARGE_N


def make_clusterer_n(method: str, seed: int, N: int = 30):
    bf = get_budget_factor(N)
    if method == "gbs_boost":
        return lambda A: gbs_boost_cluster(A, budget_factor=bf, loss=0.0, seed=seed)
    elif method == "gbs_roots":
        return lambda A: gbs_roots_cluster(A, budget_factor=bf, loss=0.0, seed=seed)
    elif method == "qic_gbs":
        return lambda A: qic_gbs_cluster(A, budget_factor=bf, loss=0.0, seed=seed)
    elif method == "spectral":
        def fn(A):
            N = A.shape[0]; K = max(2, min(8, N // 6))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return spectral_cluster(C, K, seed=seed)
        return fn
    elif method == "sponge":
        def fn(A):
            N = A.shape[0]; K = max(2, min(8, N // 6))
            C = A + np.eye(N); np.fill_diagonal(C, 1.0)
            return sponge_cluster(C, K, seed=seed)
        return fn


def run_size_experiment(sizes, prices_full, mkt, label):
    methods = {
        "Spectral": ("spectral", EXP5_CLASSICAL_RUNS),
        "SPONGE": ("sponge", EXP5_CLASSICAL_RUNS),
        "GBS Boost": ("gbs_boost", EXP5_GBS_RUNS),
        "GBS Roots": ("gbs_roots", EXP5_GBS_RUNS),
        "QIC-GBS": ("qic_gbs", EXP5_GBS_RUNS),
    }

    rows = []
    all_cols = prices_full.columns.tolist()

    for N in sizes:
        cols = all_cols[:N]
        prices_n = prices_full[cols]
        print(f"  N={N}...", end=" ", flush=True)

        for method_name, (method_key, n_runs) in methods.items():
            t0 = time.perf_counter()
            agg = run_multiple_seeds(
                prices_n, mkt,
                clusterer_factory=lambda seed, mk=method_key, n=N: make_clusterer_n(mk, seed, n),
                n_runs=n_runs,
                year=YEAR,
            )
            elapsed = time.perf_counter() - t0

            terminal = np.array(agg["terminal_returns"])
            rows.append({
                "Label": label, "N": N, "Method": method_name,
                "Mean Return": agg["mean_total_return"],
                "Std Return": agg["std_total_return"],
                "Median Return": float(np.median(terminal)) if len(terminal) > 0 else 0.0,
                "Q25 Return": float(np.quantile(terminal, 0.25)) if len(terminal) > 0 else 0.0,
                "Q75 Return": float(np.quantile(terminal, 0.75)) if len(terminal) > 0 else 0.0,
                "Mean Jaccard": agg["mean_jaccard"],
                "Mean WD": agg["mean_wd"],
                "Runtime_s": elapsed,
            })

        print(f"done")

    return rows


def run_exp5():
    print("=" * 70)
    print("EXP 5: Universe scaling analysis (N=20-30, N=12-50)")
    print("=" * 70)

    # Download 50-stock data (superset for all universes)
    tickers_50 = get_sp500_tickers_n(50)
    prices_50 = download_prices(tickers_50, LOOKBACK_START, DATA_END)
    valid = prices_50.dropna(thresh=int(0.9*len(prices_50)), axis=1).ffill()
    available = [t for t in tickers_50 if t in valid.columns][:50]
    prices_50 = valid[available]
    print(f"  Master universe: {len(prices_50.columns)} stocks")

    ff = download_fama_french()
    mkt = ff["Mkt-RF"]

    all_rows = []

    print("\nFine-grained analysis (N=20 to 30):")
    rows_fine = run_size_experiment(EXP5_SIZES_FINE, prices_50, mkt, "fine")
    all_rows.extend(rows_fine)

    print("\nBroad analysis (N=12 to 50):")
    sizes_broad = [s for s in EXP5_SIZES_BROAD if s <= len(prices_50.columns)]
    rows_broad = run_size_experiment(sizes_broad, prices_50, mkt, "broad")
    all_rows.extend(rows_broad)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp5_results.csv"), index=False)

    # Plots
    methods_gbs = ["GBS Boost", "GBS Roots", "QIC-GBS"]
    methods_cls = ["Spectral", "SPONGE"]
    colors = {
        "GBS Boost": "#2ca02c", "GBS Roots": "#d62728",
        "QIC-GBS": "#9467bd", "Spectral": "#1f77b4", "SPONGE": "#ff7f0e",
    }

    for label_tag, label_title in [("fine", "N=20-30 (Fine)"), ("broad", "N=12-50 (Broad)")]:
        sub = df[df["Label"] == label_tag]
        if sub.empty:
            continue
        sizes = sorted(sub["N"].unique())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for method in list(methods_gbs) + list(methods_cls):
            m_sub = sub[sub["Method"] == method].sort_values("N")
            axes[0].plot(m_sub["N"], m_sub["Median Return"], "o-",
                         label=method, color=colors[method])
            axes[1].plot(m_sub["N"], m_sub["Mean Jaccard"], "o-",
                         label=method, color=colors[method])

        axes[0].set_title(f"EXP5 ({label_title}): Median Return vs Universe Size")
        axes[0].set_xlabel("N (stocks)"); axes[0].set_ylabel("Median Total Return")
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"EXP5 ({label_title}): Cluster Stability vs Universe Size")
        axes[1].set_xlabel("N (stocks)"); axes[1].set_ylabel("Mean Jaccard Similarity")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = os.path.join(RESULTS_DIR, f"exp5_{label_tag}_scaling.png")
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")

    return df


if __name__ == "__main__":
    run_exp5()
