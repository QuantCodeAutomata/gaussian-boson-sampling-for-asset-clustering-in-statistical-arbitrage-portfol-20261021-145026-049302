"""Tests for portfolio construction."""
import numpy as np
import pandas as pd
import pytest
from src.portfolio import (
    classify_winners_losers, build_portfolio_weights, simulate_holding_period,
)


def make_formation_returns(T=5, N=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, N)) * 0.01


def test_classify_winners_losers_sum_to_N():
    """Winners + losers + neutral = N for p=0."""
    formation = make_formation_returns(5, 6)
    cluster = list(range(6))
    winners, losers = classify_winners_losers(cluster, formation, p=0.0)
    assert len(winners) + len(losers) <= len(cluster)


def test_classify_winners_losers_empty_cluster():
    """Empty cluster returns empty winners/losers."""
    formation = make_formation_returns()
    w, l = classify_winners_losers([], formation, p=0.0)
    assert w == [] and l == []


def test_build_portfolio_weights_net_zero():
    """Portfolio weights should sum to ~0 (zero cost)."""
    N = 8
    partition = [[0, 1, 2, 3], [4, 5, 6, 7]]
    formation = make_formation_returns(5, N)
    weights = build_portfolio_weights(partition, formation, N)
    if weights is not None:
        assert abs(weights.sum()) < 1e-10, f"Net weight not zero: {weights.sum()}"


def test_build_portfolio_weights_long_short_equal():
    """Long and short gross exposure should be equal."""
    N = 6
    partition = [[0, 1, 2, 3, 4, 5]]
    formation = make_formation_returns(5, N)
    weights = build_portfolio_weights(partition, formation, N)
    if weights is not None:
        long_exp = weights[weights > 0].sum()
        short_exp = -weights[weights < 0].sum()
        assert abs(long_exp - short_exp) < 1e-10


def test_simulate_holding_stop_win():
    """Stop-win trigger at 2% cumulative return."""
    N = 4
    weights = np.array([0.5, 0.5, -0.5, -0.5])
    returns_data = pd.DataFrame(
        [[0.03, 0.03, -0.01, -0.01]] * 10,
        columns=["A", "B", "C", "D"],
    )
    pnl, n_held = simulate_holding_period(weights, returns_data, max_hold=10, stop_win=0.02)
    # Should stop early at 2% stop-win
    assert n_held < 10


def test_simulate_holding_max_hold():
    """Should hold exactly max_hold days if no stop-win."""
    weights = np.array([0.5, -0.5])
    returns_data = pd.DataFrame([[0.005, -0.005]] * 5, columns=["A", "B"])
    pnl, n_held = simulate_holding_period(weights, returns_data, max_hold=3, stop_win=0.02)
    assert n_held == 3


def test_simulate_holding_empty_data():
    weights = np.array([1.0, -1.0])
    empty_df = pd.DataFrame(columns=["A", "B"])
    pnl, n_held = simulate_holding_period(weights, empty_df, max_hold=3)
    assert n_held == 0 and len(pnl) == 0
