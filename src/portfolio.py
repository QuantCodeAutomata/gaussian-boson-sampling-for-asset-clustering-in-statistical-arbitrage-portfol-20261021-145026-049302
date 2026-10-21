"""
Statistical arbitrage portfolio construction and backtesting.

Contrarian strategy (p=0):
- Classify each cluster member as winner (positive cum. deviation) or loser
- Short winners, long losers with equal weights within each side
- Hold max 3 days or until 2% stop-win
- Zero transaction costs, no borrow fees, no slippage

Confirmed: p=0, max hold=3 days, stop-win q=0.02.
ASSUMPTION: Capital aggregation: equal weight across active clusters.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

from src.config import HOLD_DAYS, STOP_WIN_THRESHOLD


def classify_winners_losers(
    cluster: List[int],
    formation_returns: np.ndarray,
    p: float = 0.0,
) -> Tuple[List[int], List[int]]:
    """
    Classify cluster members as winners (short) or losers (long).

    p=0: winners have POSITIVE cumulative deviation above cluster mean;
         losers have NEGATIVE cumulative deviation.

    Args:
        cluster: list of column indices in the formation window
        formation_returns: (T_form, N) array of residual returns
        p: deviation threshold (confirmed p=0)

    Returns:
        (winners, losers) as lists of column indices
    """
    if not cluster:
        return [], []

    cluster_returns = formation_returns[:, cluster]
    cum_returns = cluster_returns.sum(axis=0)  # cumulative sum
    mean_cum = cum_returns.mean()

    deviations = cum_returns - mean_cum
    winners = [cluster[i] for i, d in enumerate(deviations) if d > p]
    losers = [cluster[i] for i, d in enumerate(deviations) if d < -p]

    return winners, losers


def build_portfolio_weights(
    partition: List[List[int]],
    formation_returns: np.ndarray,
    n_stocks: int,
    p: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Build zero-cost contrarian portfolio weights across all clusters.

    Within each cluster: equal-weight long and short sides.
    Across clusters: equal weight (1/K_active).
    Net portfolio: w_long - w_short, normalised so |long| = |short|.

    ASSUMPTION: Inactive clusters (only winners or only losers) are skipped.

    Args:
        partition: list of clusters (each is list of stock indices)
        formation_returns: (T_form, N) residual returns
        n_stocks: total number of stocks N
        p: winner/loser threshold

    Returns:
        Weight vector of length N; None if no active cluster
    """
    active_clusters = []
    for cluster in partition:
        if len(cluster) < 2:
            continue
        winners, losers = classify_winners_losers(cluster, formation_returns, p)
        if winners and losers:
            active_clusters.append((cluster, winners, losers))

    if not active_clusters:
        return None

    weights = np.zeros(n_stocks)
    K = len(active_clusters)

    for _cluster, winners, losers in active_clusters:
        # Equal weight within each side, normalised to 1/-1
        w_long = 1.0 / (len(losers) * K)
        w_short = -1.0 / (len(winners) * K)
        for i in losers:
            weights[i] += w_long
        for i in winners:
            weights[i] += w_short

    return weights


def simulate_holding_period(
    weights: np.ndarray,
    holding_returns: pd.DataFrame,
    max_hold: int = HOLD_DAYS,
    stop_win: float = STOP_WIN_THRESHOLD,
) -> Tuple[pd.Series, int]:
    """
    Simulate holding period: liquidate after max_hold days or at stop-win.

    Daily portfolio return = weights.T @ stock_returns[t]

    Args:
        weights: portfolio weight vector
        holding_returns: DataFrame of stock returns during holding window
        max_hold: maximum holding period in trading days (confirmed: 3)
        stop_win: cumulative return threshold for early liquidation (confirmed: 0.02)

    Returns:
        (daily_pnl series, actual hold days)
    """
    pnl_list = []
    cum_return = 0.0
    n_hold = min(max_hold, len(holding_returns))

    for t in range(n_hold):
        row = holding_returns.iloc[t].fillna(0.0).values
        daily_ret = float(weights @ row)
        pnl_list.append(daily_ret)
        cum_return += daily_ret
        if cum_return >= stop_win:
            break

    return pd.Series(pnl_list), len(pnl_list)
