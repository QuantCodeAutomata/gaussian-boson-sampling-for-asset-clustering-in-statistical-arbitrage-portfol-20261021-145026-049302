"""
Performance and partition quality metrics.

Partition metrics: weighted density, global cluster value (Value(C)), Jaccard.
Performance metrics: Sharpe, Sortino (with paper-consistent conventions).

ASSUMPTION: Annualisation = sqrt(252) for Sharpe (252 trading days/year).
ASSUMPTION: Risk-free rate = 0 (paper does not specify; most common for StatArb).
ASSUMPTION: Downside deviation uses daily returns < 0 (semi-deviation).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.config import ANNUAL_FACTOR, RISK_FREE_RATE


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualised Sharpe ratio with zero risk-free rate."""
    if len(returns) < 2:
        return 0.0
    excess = returns - RISK_FREE_RATE / ANNUAL_FACTOR
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(ANNUAL_FACTOR) * excess.mean() / std)


def sortino_ratio(returns: pd.Series) -> float:
    """Annualised Sortino ratio (downside deviation uses returns < 0)."""
    if len(returns) < 2:
        return 0.0
    excess = returns - RISK_FREE_RATE / ANNUAL_FACTOR
    downside = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std(ddof=1)
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(ANNUAL_FACTOR) * excess.mean() / downside_std)


def total_return(returns: pd.Series) -> float:
    """Cumulative (compound) total return."""
    if len(returns) == 0:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def partition_weighted_density(
    partition: List[List[int]],
    A: np.ndarray,
) -> float:
    """
    Mean weighted density across all clusters.

    WD_partition = mean over clusters of WD(cluster)

    ASSUMPTION: Mean over clusters (not weighted by cluster size).

    Args:
        partition: list of clusters
        A: adjacency weight matrix

    Returns:
        Mean weighted density
    """
    from src.gbs_sampling import compute_weighted_density
    if not partition:
        return 0.0
    wds = [compute_weighted_density(c, A) for c in partition]
    return float(np.mean(wds))


def global_cluster_value(
    partition: List[List[int]],
    A: np.ndarray,
) -> float:
    """
    Global cluster value (Value(C)) = sum over all within-cluster pairs.

    Value(C) = sum_{k} sum_{u<v in C_k} w_{uv}

    Confirmed from paper Appendix L (Metrics section).
    """
    total = 0.0
    for cluster in partition:
        if len(cluster) < 2:
            continue
        s = np.array(cluster)
        sub = A[np.ix_(s, s)]
        total += np.triu(sub, k=1).sum()
    return float(total)


def jaccard_similarity(cluster_a: List[int], cluster_b: List[int]) -> float:
    """Jaccard index between two clusters."""
    sa, sb = set(cluster_a), set(cluster_b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def partition_jaccard(
    new_partition: List[List[int]],
    old_partition: List[List[int]],
) -> float:
    """
    Consecutive-window Jaccard similarity.

    For each new cluster, find maximum Jaccard overlap with any old cluster.
    Return mean of these maxima.

    Confirmed from paper Appendix L.

    Args:
        new_partition: clusters at current rebalance window
        old_partition: clusters at previous rebalance window

    Returns:
        Mean maximum Jaccard score in [0, 1]
    """
    if not new_partition or not old_partition:
        return 0.0
    scores = []
    for new_c in new_partition:
        best = max(jaccard_similarity(new_c, old_c) for old_c in old_partition)
        scores.append(best)
    return float(np.mean(scores))


def welch_t_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> dict:
    """
    Two-sided Welch unequal-variance t-test.

    Returns dict with: t, df, p_value, ci_lower, ci_upper, n_a, n_b,
    cohens_d (effect size).

    Confirmed: scipy.stats.ttest_ind(..., equal_var=False) per paper.
    """
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(group_a, group_b, equal_var=False)
    n_a, n_b = len(group_a), len(group_b)

    # Welch degrees of freedom
    s_a = np.var(group_a, ddof=1)
    s_b = np.var(group_b, ddof=1)
    denom = (s_a / n_a + s_b / n_b) ** 2
    num = (s_a / n_a) ** 2 / (n_a - 1) + (s_b / n_b) ** 2 / (n_b - 1)
    df = denom / num if num > 0 else n_a + n_b - 2

    se = np.sqrt(s_a / n_a + s_b / n_b)
    ci = stats.t.interval(0.95, df, loc=np.mean(group_a) - np.mean(group_b), scale=se)

    # Cohen's d (pooled)
    pooled_std = np.sqrt((s_a + s_b) / 2.0)
    cohens_d = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0.0

    return {
        "t": float(t_stat),
        "df": float(df),
        "p_value": float(p_val),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "cohens_d": float(cohens_d),
        "mean_a": float(np.mean(group_a)),
        "mean_b": float(np.mean(group_b)),
    }
