"""
Experiment 4: GBS sample-budget scaling for lossless 30-stock universe.

Tests F in {0.01, 0.1, 1, 10, 100} for H = F*N*ln(N).
Measures WD, total return, and runtime for GBS Boost, Roots, QIC-GBS.

Paper reference: Appendix J (pp. 20-21), Figure 8.
ASSUMPTION: 30-stock subset = first 30 from standard 100-ticker list.
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
    EXP4_F_VALUES, EXP4_RUNS_LOW_F, EXP4_RUNS_MID_F, EXP4_RUNS_HIGH_F,
    RESULTS_DIR,
)
from src.data_loader import download_prices, download_fama_french, get_sp500_tickers_n
from src.backtester import run_multiple_seeds
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster
from src.sponge import sponge_cluster

N_STOCKS = 30
YEAR = 2020
LOOKBACK_START = "2018-01-01"
DATA_END = "2021-01-10"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_counts_for_F(F: float) -> int:
    if F <= 0.1:
        return EXP4_RUNS_LOW_F
    elif F <= 1.0:
        return EXP4_RUNS_MID_F
    else:
        return EXP4_RUNS_HIGH_F


def run_exp4():
    print("=" * 70)
    print("EXP 4: Sample budget scaling (30 stocks, 2020)")
    print("=" * 70)

    tickers = get_sp500_tickers_n(N_STOCKS)
    prices = download_prices(tickers, LOOKBACK_START, DATA_END)
    valid = prices.dropna(thresh=int(0.9*len(prices)), axis=1).ffill()
    available = [t for t in tickers if t in valid.columns][:N_STOCKS]
    prices = valid[available]
    print(f"  Using {len(prices.columns)} stocks")

    ff = download_fama_french()
    mkt = ff["Mkt-RF"]

    rows = []

    for F in EXP4_F_VALUES:
        n_runs = run_counts_for_F(F)
        print(f"\nF={F:.2f} ({n_runs} runs per method):")

        for method_name, mk in [("GBS Boost", "boost"), ("GBS Roots", "roots"), ("QIC-GBS", "qic")]:
            def make_clusterer(seed, _mk=mk, _F=F):
                if _mk == "boost":
                    return lambda A: gbs_boost_cluster(A, budget_factor=_F, loss=0.0, seed=seed)
                elif _mk == "roots":
                    return lambda A: gbs_roots_cluster(A, budget_factor=_F, loss=0.0, seed=seed)
                else:
                    return lambda A: qic_gbs_cluster(A, budget_factor=_F, loss=0.0, seed=seed)

            t0 = time.perf_counter()
            agg = run_multiple_seeds(prices, mkt, make_clusterer, n_runs=n_runs, year=YEAR)
            elapsed = time.perf_counter() - t0

            print(f"  {method_name:12s}: return={agg['mean_total_return']:.3f}, "
                  f"WD={agg['mean_wd']:.4f}, runtime={elapsed:.2f}s")

            rows.append({
                "F": F, "Method": method_name,
                "Mean Return": agg["mean_total_return"],
                "Std Return": agg["std_total_return"],
                "Mean WD": agg["mean_wd"],
                "Mean GCV": agg["mean_gcv"],
                "Runtime_s": elapsed,
                "N Runs": n_runs,
            })

        # Classical references (only run once at F=1)
        if abs(F - 1.0) < 0.01:
            for method_name, mk in [("Spectral", "spectral"), ("SPONGE", "sponge")]:
                def make_cls(seed, _mk=mk):
                    def fn(A):
                        N = A.shape[0]; K = max(2, min(8, N // 6))
                        C = A + np.eye(N); np.fill_diagonal(C, 1.0)
                        if _mk == "spectral":
                            return spectral_cluster(C, K, seed=seed)
                        return sponge_cluster(C, K, seed=seed)
                    return fn

                t0 = time.perf_counter()
                agg = run_multiple_seeds(prices, mkt, make_cls, n_runs=5, year=YEAR)
                elapsed = time.perf_counter() - t0
                print(f"  {method_name:12s}: return={agg['mean_total_return']:.3f}, runtime={elapsed:.2f}s")
                for f_ref in EXP4_F_VALUES:
                    rows.append({
                        "F": f_ref, "Method": method_name,
                        "Mean Return": agg["mean_total_return"],
                        "Std Return": agg["std_total_return"],
                        "Mean WD": agg["mean_wd"],
                        "Mean GCV": agg["mean_gcv"],
                        "Runtime_s": elapsed,
                        "N Runs": 5,
                    })
                break  # Add classical results for all F values once

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "exp4_results.csv"), index=False)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    gbs_methods = ["GBS Boost", "GBS Roots", "QIC-GBS"]
    colors = {"GBS Boost": "#2ca02c", "GBS Roots": "#d62728",
              "QIC-GBS": "#9467bd", "Spectral": "#1f77b4", "SPONGE": "#ff7f0e"}

    for method in gbs_methods:
        sub = df[df["Method"] == method].drop_duplicates("F").sort_values("F")
        axes[0].semilogx(sub["F"], sub["Mean WD"], "o-", label=method, color=colors[method])
        axes[1].semilogx(sub["F"], sub["Mean Return"], "o-", label=method, color=colors[method])
        axes[2].loglog(sub["F"], sub["Runtime_s"], "o-", label=method, color=colors[method])

    for ax in axes:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlabel("Sample Factor F (log scale)")

    axes[0].set_title("EXP4: WD vs Sample Budget"); axes[0].set_ylabel("Mean Weighted Density")
    axes[1].set_title("EXP4: Return vs Sample Budget"); axes[1].set_ylabel("Mean Total Return")
    axes[2].set_title("EXP4: Runtime vs Sample Budget"); axes[2].set_ylabel("Runtime (s, log scale)")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_scaling.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR}/exp4_scaling.png")
    return df


if __name__ == "__main__":
    run_exp4()
