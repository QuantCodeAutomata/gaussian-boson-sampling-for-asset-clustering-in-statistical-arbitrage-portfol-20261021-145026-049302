"""
Return calculations and beta residualisation.

Daily total return: R[t,i] = (P[t,i] + D[t,i]) / P[t-1,i] - 1
Beta estimation: 60-day OLS regression on excess market return
Residual return: R_res = R - beta * R_mkt

Confirmed: 60-day beta window, single-factor market model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from src.config import BETA_WINDOW


def compute_total_returns(
    prices: pd.DataFrame,
    dividends: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute daily total returns including dividends.

    R[t,i] = (P[t,i] + D[t,i]) / P[t-1,i] - 1

    If prices are already dividend-adjusted (e.g., from yfinance "auto_adjust"),
    simple returns R[t] = P[t]/P[t-1] - 1 are used and dividends ignored.

    Args:
        prices: DataFrame of adjusted close prices (rows=dates, cols=tickers)
        dividends: Optional dividend DataFrame aligned with prices

    Returns:
        DataFrame of daily total returns, same shape as prices[1:]
    """
    if dividends is not None:
        div_aligned = dividends.reindex(prices.index).fillna(0.0)
        numerator = prices + div_aligned
        returns = (numerator / prices.shift(1)) - 1.0
    else:
        returns = prices.pct_change()

    return returns.iloc[1:]  # drop first NaN row


def estimate_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.Series:
    """
    Estimate rolling OLS beta over preceding `window` trading days.

    beta_i = Cov(R_i, R_mkt) / Var(R_mkt)

    Vectorised via pandas rolling — no Python date loop.

    Args:
        stock_returns: daily stock return series
        market_returns: Fama-French excess market return series
        window: look-back window in trading days (default 60)

    Returns:
        Series of rolling beta estimates (same index as stock_returns)
    """
    mkt_aligned = market_returns.reindex(stock_returns.index).fillna(0.0)
    roll_cov = stock_returns.rolling(window, min_periods=window).cov(mkt_aligned)
    roll_var = mkt_aligned.rolling(window, min_periods=window).var()
    # shift(1): beta at position t uses window [t-window, t-1] — no look-ahead.
    # Matches original loop which assigned beta[t] from rows [t-window : t].
    # Positions 0..window-1 remain NaN (look-back window not yet full).
    beta = (roll_cov / roll_var.where(roll_var > 1e-12)).shift(1)
    return beta


def compute_residual_returns(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.DataFrame:
    """
    Compute market-residual returns for all stocks.

    R_res[t,i] = R[t,i] - beta_i(t) * R_mkt[t]

    Uses only information available before each date (confirmed).
    Fully vectorised: rolling covariance / rolling variance across all stocks.

    Args:
        stock_returns: (T, N) DataFrame of daily returns
        market_returns: Fama-French excess market return
        window: beta look-back window

    Returns:
        (T - window, N) residual return DataFrame
    """
    mkt_aligned = market_returns.reindex(stock_returns.index).fillna(0.0)
    # Rolling beta: shift(1) so beta[t] uses window [t-window, t-1] — no look-ahead.
    roll_cov = stock_returns.rolling(window, min_periods=window).cov(mkt_aligned)
    roll_var = mkt_aligned.rolling(window, min_periods=window).var()
    beta_df = roll_cov.div(roll_var.where(roll_var > 1e-12), axis=0).shift(1).fillna(0.0)

    residuals = stock_returns - beta_df.mul(mkt_aligned, axis=0)
    # Drop rows where betas were NaN before fill (first `window` rows)
    residuals = residuals.iloc[window:]
    return residuals.dropna(how="all")
