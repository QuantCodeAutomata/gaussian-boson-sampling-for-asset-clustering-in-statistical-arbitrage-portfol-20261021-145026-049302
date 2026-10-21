"""
Main backtest engine for GBS StatArb strategy.

Rolling rebalance loop:
- At each date T, estimate betas and compute residual returns
- Compute 5-day correlation matrix and adjacency
- Generate partition from chosen method
- Build contrarian portfolio weights
- Simulate holding period (max 3 days, stop-win 2%)
- Advance to next rebalance date
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Dict, Any, Tuple

from src.config import (
    HOLD_DAYS, STOP_WIN_THRESHOLD, BETA_WINDOW,
    CORRELATION_WINDOW, K_WINDOW, FORMATION_WINDOW,
)
from src.returns import compute_total_returns, compute_residual_returns
from src.correlation import compute_correlation_matrix, correlation_to_adjacency
from src.portfolio import build_portfolio_weights, simulate_holding_period
from src.metrics import (
    sharpe_ratio, sortino_ratio, total_return,
    partition_weighted_density, global_cluster_value, partition_jaccard,
)


ClustererFn = Callable[[np.ndarray], List[List[int]]]


def precompute_inputs(
    prices: pd.DataFrame,
    market_returns: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-compute stock returns and residuals once, reuse across seeds.

    Args:
        prices: daily adjusted close prices
        market_returns: Fama-French daily excess market return

    Returns:
        (stock_returns, residuals) DataFrames
    """
    stock_returns = compute_total_returns(prices)
    mkt = market_returns.reindex(stock_returns.index).fillna(0.0)
    residuals = compute_residual_returns(stock_returns, mkt, window=BETA_WINDOW)
    return stock_returns, residuals


def run_backtest(
    prices: pd.DataFrame,
    market_returns: pd.Series,
    clusterer: ClustererFn,
    year: Optional[int] = None,
    verbose: bool = False,
    _precomputed: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """
    Run a full-year StatArb backtest with the given clustering function.

    Args:
        prices: daily adjusted close prices (rows=dates, cols=tickers)
        market_returns: Fama-French daily excess market return
        clusterer: function(A) -> partition (list of lists of indices)
        year: restrict backtest to this year (None = all data)
        verbose: print progress
        _precomputed: optional (stock_returns, residuals) from precompute_inputs()

    Returns:
        Dict with:
          - daily_pnl: pd.Series of daily PnL
          - total_return: float
          - sharpe: float
          - sortino: float
          - mean_wd: float (average weighted density)
          - mean_gcv: float (average global cluster value)
          - partitions: list of (date, partition) tuples
          - jaccard_scores: list of Jaccard similarity scores
    """
    if _precomputed is not None:
        stock_returns, residuals = _precomputed
    else:
        stock_returns = compute_total_returns(prices)
        mkt = market_returns.reindex(stock_returns.index).fillna(0.0)
        residuals = compute_residual_returns(stock_returns, mkt, window=BETA_WINDOW)

    # Filter to backtest year
    if year is not None:
        residuals = residuals[residuals.index.year == year]
        stock_returns = stock_returns[stock_returns.index.year == year]

    dates = residuals.index.tolist()
    N = residuals.shape[1]

    if len(dates) < BETA_WINDOW + CORRELATION_WINDOW + 1:
        return _empty_result()

    # Backtesting state
    all_pnl: List[float] = []
    all_dates: List[Any] = []
    partitions: List[Tuple] = []
    wd_scores: List[float] = []
    gcv_scores: List[float] = []
    jaccard_scores: List[float] = []
    prev_partition: Optional[List[List[int]]] = None

    t = CORRELATION_WINDOW  # start after enough history for correlation

    while t < len(dates):
        # Formation window: past CORRELATION_WINDOW days
        form_start = max(0, t - CORRELATION_WINDOW)
        form_data = residuals.iloc[form_start:t]
        form_returns_arr = form_data.values

        # Compute correlation and adjacency
        C = compute_correlation_matrix(form_data, window=None)
        A = correlation_to_adjacency(C)

        # Cluster
        try:
            partition = clusterer(A)
        except Exception as e:
            if verbose:
                print(f"  Clustering failed at t={t}: {e}")
            t += 1
            continue

        if not partition:
            t += 1
            continue

        # Build portfolio weights
        weights = build_portfolio_weights(partition, form_returns_arr, N)
        if weights is None:
            t += 1
            continue

        # Holding period
        hold_end = min(t + HOLD_DAYS, len(dates))
        holding_data = stock_returns.iloc[t:hold_end]

        if len(holding_data) == 0:
            break

        daily_pnl, n_held = simulate_holding_period(
            weights, holding_data, max_hold=HOLD_DAYS, stop_win=STOP_WIN_THRESHOLD
        )

        # Record PnL
        for day_i, pnl_val in enumerate(daily_pnl):
            if t + day_i < len(dates):
                all_pnl.append(pnl_val)
                all_dates.append(dates[t + day_i])

        # Metrics
        wd_scores.append(partition_weighted_density(partition, A))
        gcv_scores.append(global_cluster_value(partition, A))

        if prev_partition is not None:
            jac = partition_jaccard(partition, prev_partition)
            jaccard_scores.append(jac)

        partitions.append((dates[t], partition))
        prev_partition = partition

        if verbose and len(partitions) % 20 == 0:
            print(f"  Processed {len(partitions)} rebalance points, t={t}/{len(dates)}")

        # Advance to next rebalance date
        t += n_held

    if not all_pnl:
        return _empty_result()

    pnl_series = pd.Series(all_pnl, index=all_dates)

    return {
        "daily_pnl": pnl_series,
        "total_return": total_return(pnl_series),
        "sharpe": sharpe_ratio(pnl_series),
        "sortino": sortino_ratio(pnl_series),
        "mean_wd": float(np.mean(wd_scores)) if wd_scores else 0.0,
        "mean_gcv": float(np.mean(gcv_scores)) if gcv_scores else 0.0,
        "partitions": partitions,
        "jaccard_scores": jaccard_scores,
        "mean_jaccard": float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
    }


def run_multiple_seeds(
    prices: pd.DataFrame,
    market_returns: pd.Series,
    clusterer_factory: Callable[[int], ClustererFn],
    n_runs: int,
    year: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run backtest with n_runs independent random seeds and aggregate results.

    Pre-computes financial inputs (returns, residuals) once and shares them
    across all seeds to avoid redundant computation.

    Args:
        prices: price data
        market_returns: FF market return
        clusterer_factory: function(seed) -> clusterer function
        n_runs: number of independent runs
        year: backtest year
        verbose: print progress

    Returns:
        Dict with mean/std of total_return, sharpe, sortino, wd, gcv, jaccard
        plus list of per-run results
    """
    # Pre-compute expensive financial inputs ONCE across all seeds
    if verbose:
        print("  Pre-computing returns and residuals...")
    precomputed = precompute_inputs(prices, market_returns)

    run_results = []
    for run_i in range(n_runs):
        if verbose:
            print(f"  Run {run_i+1}/{n_runs}...")
        clusterer = clusterer_factory(seed=run_i * 42 + 7)
        result = run_backtest(
            prices, market_returns, clusterer, year=year,
            _precomputed=precomputed,
        )
        run_results.append(result)

    def _agg(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in run_results]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    mean_ret, std_ret = _agg("total_return")
    mean_sharpe, std_sharpe = _agg("sharpe")
    mean_sortino, std_sortino = _agg("sortino")
    mean_wd, std_wd = _agg("mean_wd")
    mean_gcv, std_gcv = _agg("mean_gcv")
    mean_jac, std_jac = _agg("mean_jaccard")

    # Combine all daily pnl for full equity curve
    all_pnl = pd.concat([r["daily_pnl"] for r in run_results if len(r["daily_pnl"]) > 0])

    return {
        "mean_total_return": mean_ret,
        "std_total_return": std_ret,
        "mean_sharpe": mean_sharpe,
        "std_sharpe": std_sharpe,
        "mean_sortino": mean_sortino,
        "std_sortino": std_sortino,
        "mean_wd": mean_wd,
        "std_wd": std_wd,
        "mean_gcv": mean_gcv,
        "std_gcv": std_gcv,
        "mean_jaccard": mean_jac,
        "std_jaccard": std_jac,
        "n_runs": n_runs,
        "run_results": run_results,
        "combined_pnl": all_pnl,
        "terminal_returns": [r["total_return"] for r in run_results],
        "sharpe_per_run": [r["sharpe"] for r in run_results],
    }


def _empty_result() -> Dict[str, Any]:
    """Return empty result dict when backtest produces no trades."""
    return {
        "daily_pnl": pd.Series(dtype=float),
        "total_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "mean_wd": 0.0,
        "mean_gcv": 0.0,
        "partitions": [],
        "jaccard_scores": [],
        "mean_jaccard": 0.0,
    }
