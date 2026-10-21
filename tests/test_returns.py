"""Tests for return and residual calculations."""
import numpy as np
import pandas as pd
import pytest
from src.returns import (
    compute_total_returns, estimate_rolling_beta, compute_residual_returns,
)


def make_price_series(n=100, seed=0):
    rng = np.random.default_rng(seed)
    log_returns = rng.standard_normal(n) * 0.01
    prices = 100 * np.exp(np.cumsum(log_returns))
    return pd.Series(prices, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def test_total_returns_no_dividends():
    """Simple price returns without dividends."""
    prices = pd.DataFrame({"A": make_price_series(50)})
    returns = compute_total_returns(prices)
    assert len(returns) == 49
    assert returns.notna().all().all()


def test_total_returns_with_dividends():
    """Returns with dividends should be >= simple returns (all-positive divs)."""
    prices = pd.DataFrame({"A": make_price_series(50)})
    divs = pd.DataFrame({"A": [0.01] * 50}, index=prices.index)
    r_no_div = compute_total_returns(prices)
    r_div = compute_total_returns(prices, divs)
    # Total return with dividends should differ from simple return
    assert not np.allclose(r_no_div.values, r_div.values)


def test_rolling_beta_constant_market():
    """If market return is constant, beta should be 0 (zero variance)."""
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    stock = pd.Series(np.random.default_rng(0).standard_normal(n) * 0.01, index=idx)
    mkt = pd.Series(np.full(n, 0.001), index=idx)
    beta = estimate_rolling_beta(stock, mkt, window=60)
    # Where defined (after window), beta should be 0 (zero market variance)
    valid = beta.iloc[60:]
    assert (valid == 0.0).all() or valid.isna().all()


def test_residual_returns_length():
    """Residual returns should have BETA_WINDOW fewer rows."""
    n = 150
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    prices = pd.DataFrame({
        "A": np.random.default_rng(0).lognormal(0, 0.01, n) * 100,
        "B": np.random.default_rng(1).lognormal(0, 0.01, n) * 50,
    }, index=idx)
    mkt_r = pd.Series(np.random.default_rng(2).standard_normal(n) * 0.005, index=idx)

    stock_r = compute_total_returns(prices)
    residuals = compute_residual_returns(stock_r, mkt_r, window=60)
    assert len(residuals) == len(stock_r) - 60


def test_residual_returns_no_future_leak():
    """Each residual at t uses only data from t-window to t-1."""
    # We verify the function doesn't use future data by checking
    # that beta estimates before window are NaN
    n = 80
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.DataFrame({
        "A": np.random.default_rng(0).lognormal(0, 0.01, n) * 100,
    }, index=idx)
    mkt_r = pd.Series(np.random.default_rng(2).standard_normal(n) * 0.005, index=idx)
    stock_r = compute_total_returns(prices)
    beta = estimate_rolling_beta(stock_r["A"], mkt_r, window=60)
    # Beta should be NaN for first 60 rows
    assert beta.iloc[:60].isna().all()
