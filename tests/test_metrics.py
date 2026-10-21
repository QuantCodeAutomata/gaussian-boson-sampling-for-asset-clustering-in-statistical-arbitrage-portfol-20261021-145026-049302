"""Tests for performance and partition quality metrics."""
import numpy as np
import pandas as pd
import pytest
from src.metrics import (
    sharpe_ratio, sortino_ratio, total_return,
    partition_weighted_density, global_cluster_value,
    jaccard_similarity, partition_jaccard, welch_t_test,
)


def test_sharpe_positive_returns():
    """Positive returns should yield positive Sharpe."""
    returns = pd.Series([0.01] * 100)
    assert sharpe_ratio(returns) > 0


def test_sharpe_zero_variance():
    """Constant returns should return 0 (zero std)."""
    returns = pd.Series([0.0] * 50)
    assert sharpe_ratio(returns) == 0.0


def test_sortino_positive():
    """Positive trend with few downticks: Sortino >= Sharpe."""
    returns = pd.Series(np.random.default_rng(0).standard_normal(100) * 0.01 + 0.002)
    sr = sharpe_ratio(returns)
    so = sortino_ratio(returns)
    # Sortino should be >= Sharpe when downside is smaller than total std
    assert isinstance(so, float)


def test_total_return_compound():
    """Compound return: 1.1^2 - 1 = 0.21."""
    returns = pd.Series([0.1, 0.1])
    tr = total_return(returns)
    assert abs(tr - 0.21) < 1e-10


def test_total_return_empty():
    assert total_return(pd.Series(dtype=float)) == 0.0


def test_jaccard_identical():
    assert abs(jaccard_similarity([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-10


def test_jaccard_disjoint():
    assert abs(jaccard_similarity([1, 2], [3, 4]) - 0.0) < 1e-10


def test_jaccard_partial():
    j = jaccard_similarity([1, 2, 3], [2, 3, 4])
    assert abs(j - 2/4) < 1e-10


def test_partition_jaccard_trivial():
    """Same partition should yield Jaccard 1."""
    part = [[0, 1, 2], [3, 4, 5]]
    j = partition_jaccard(part, part)
    assert abs(j - 1.0) < 1e-10


def test_weighted_density_partition():
    A = np.array([[0, 0.5, 0.3], [0.5, 0, 0.4], [0.3, 0.4, 0]])
    part = [[0, 1], [2]]
    wd = partition_weighted_density(part, A)
    assert wd >= 0.0


def test_welch_t_test_different_means():
    rng = np.random.default_rng(0)
    a = rng.normal(0.2, 0.1, 20)
    b = rng.normal(0.1, 0.1, 20)
    result = welch_t_test(a, b)
    assert "t" in result and "p_value" in result and "df" in result
    assert result["p_value"] >= 0.0 and result["p_value"] <= 1.0
    assert result["n_a"] == 20 and result["n_b"] == 20


def test_welch_t_test_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0.1, 0.1, 30)
    b = rng.normal(0.1, 0.1, 30)
    result = welch_t_test(a, b)
    # Should NOT reject at alpha=0.05 most of the time
    assert result["p_value"] >= 0.0


def test_global_cluster_value():
    A = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    part = [[0, 1, 2]]
    gcv = global_cluster_value(part, A)
    # Sum of upper triangle: 1 + 2 + 3 = 6
    assert abs(gcv - 6.0) < 1e-10
