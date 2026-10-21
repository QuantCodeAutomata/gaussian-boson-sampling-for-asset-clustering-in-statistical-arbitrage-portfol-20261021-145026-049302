"""
Data loading for GBS StatArb experiments.

Downloads dividend-adjusted prices from Yahoo Finance and
Fama-French 3-factor data (for market excess return).

ASSUMPTION: Uses yfinance for price data (CRSP/WRDS unavailable without
institutional subscription). Prices are adjusted for splits and dividends.
ASSUMPTION: yfinance "auto_adjust=True" provides total-return-equivalent
adjusted prices; simple returns approximate total returns.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

from src.config import DATA_DIR, PRICE_CACHE_FILE, FF_CACHE_FILE


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices from Yahoo Finance.

    ASSUMPTION: Prices are dividend-adjusted (yfinance auto_adjust=True).

    Args:
        tickers: list of Yahoo Finance tickers
        start: start date string "YYYY-MM-DD"
        end: end date string "YYYY-MM-DD"
        use_cache: load from cache file if available

    Returns:
        DataFrame of adjusted close prices (rows=dates, cols=tickers)
    """
    import hashlib
    ticker_hash = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    cache_path = os.path.join(DATA_DIR, f"prices_{start}_{end}_{ticker_hash}.parquet")
    if use_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        # yfinance >= 0.2.x returns MultiIndex columns (Price, Ticker) when
        # multiple tickers are requested; single-ticker gives flat columns.
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"]
            # Ensure columns are ticker strings
            prices.columns = [str(c) for c in prices.columns]
        else:
            # Single ticker or already flat
            if "Close" in data.columns:
                prices = data[["Close"]]
                prices.columns = [str(tickers[0])] if len(tickers) == 1 else tickers
            else:
                prices = data

        # Retain only requested tickers that were downloaded
        available = [t for t in tickers if t in prices.columns]
        prices = prices[available]
        prices = prices.dropna(how="all").ffill().dropna(how="all")

    except Exception as e:
        raise RuntimeError(f"Failed to download prices: {e}") from e

    os.makedirs(DATA_DIR, exist_ok=True)
    prices.to_parquet(cache_path)
    return prices


def download_fama_french(
    start: str = "2000-01",
    end: str = "2024-12",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download Fama-French 3-factor data using pandas_datareader.

    Returns monthly data with "Mkt-RF" as excess market return.
    Converted to daily assuming 21 trading days/month.

    ASSUMPTION: Daily market return approximated from monthly FF data
    by dividing by 21. For better accuracy, use CRSP daily data.

    Args:
        start: start date "YYYY-MM"
        end: end date "YYYY-MM"
        use_cache: use cached file if available

    Returns:
        DataFrame with columns ["Mkt-RF", "SMB", "HML", "RF"] and daily index
    """
    cache_path = os.path.join(DATA_DIR, "fama_french_daily.parquet")
    if use_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        import pandas_datareader.data as web
        ff = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench")[0]
        ff = ff / 100.0  # convert from percent to decimal
        ff.index = pd.to_datetime(ff.index, format="%Y%m%d")
    except Exception as e:
        # Fallback: use market return proxy via SPY
        print(f"Warning: Fama-French download failed ({e}). Using SPY as market proxy.")
        ff = _get_spy_market_return(start, end)

    os.makedirs(DATA_DIR, exist_ok=True)
    ff.to_parquet(cache_path)
    return ff


def _get_spy_market_return(start: str, end: str) -> pd.DataFrame:
    """Fallback: use SPY returns as market proxy when FF data is unavailable."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start[:4]+"-01-01",
                         end=end[:4]+"-12-31", auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            close = spy["Close"].iloc[:, 0]
        else:
            close = spy["Close"] if "Close" in spy.columns else spy.iloc[:, 0]
        mkt = close.pct_change().dropna()
        df = pd.DataFrame({"Mkt-RF": mkt.values, "RF": 0.0}, index=mkt.index)
        return df
    except Exception:
        # Last resort: return zeros
        idx = pd.date_range(start, end, freq="B")
        return pd.DataFrame({"Mkt-RF": 0.0, "RF": 0.0}, index=idx)


def load_vix(
    start: str = "2000-01-01",
    end: str = "2024-12-31",
) -> pd.Series:
    """
    Load CBOE VIX daily data from Yahoo Finance (^VIX).

    Args:
        start, end: date range strings

    Returns:
        Daily VIX series
    """
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            return vix["Close"]["^VIX"]
        return vix["Close"]
    except Exception as e:
        print(f"Warning: VIX download failed: {e}")
        return pd.Series(dtype=float)


def get_sp500_tickers(n: int = 100, year: int = 2020) -> List[str]:
    """
    Return a list of S&P 500 tickers for backtesting.

    ASSUMPTION: Uses a hardcoded list of stable S&P 500 components
    from the paper's sector-stratified sample. The exact 100 PERMNOs
    from the paper are not available without CRSP access; we use a
    representative stratified sample across all 11 GICS sectors.

    Sector allocation for N=100:
      Energy(5) Materials(5) Industrials(10) ConsDiscr(12) ConsStaples(7)
      Healthcare(13) Financials(13) IT(15) CommServices(9) Utilities(5) RealEstate(6)

    These tickers are selected from S&P 500 components active throughout
    the 2020 period with sufficient trading volume.
    """
    sp500_sectors = {
        "Energy": ["XOM", "CVX", "COP", "PSX", "VLO"],
        "Materials": ["LIN", "APD", "NEM", "FCX", "NUE"],
        "Industrials": ["HON", "UPS", "RTX", "CAT", "DE", "FDX", "NOC", "GD", "EMR", "ITW"],
        "ConsumerDiscretionary": [
            "AMZN", "HD", "MCD", "NKE", "TGT", "SBUX", "LOW", "TJX", "BKNG", "GM", "F", "DHI"
        ],
        "ConsumerStaples": ["PG", "KO", "PEP", "WMT", "COST", "CL", "MO"],
        "Healthcare": [
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR",
            "CVS", "MDT", "ISRG", "BMY", "GILD"
        ],
        "Financials": [
            "BRK-B", "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK",
            "AXP", "SPGI", "CB", "MMC", "TRV"
        ],
        "InformationTechnology": [
            "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ACN", "CSCO",
            "IBM", "INTC", "QCOM", "TXN", "AMD", "ADBE", "NOW"
        ],
        "CommunicationServices": [
            "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "EA"
        ],
        "Utilities": ["NEE", "DUK", "SO", "D", "EXC"],
        "RealEstate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG"],
    }

    all_tickers = []
    for sector, tickers in sp500_sectors.items():
        all_tickers.extend(tickers[:min(len(tickers), n)])

    # Deduplicate and limit
    seen = set()
    result = []
    for t in all_tickers:
        if t not in seen and len(result) < n:
            seen.add(t)
            result.append(t)

    return result[:n]


def get_sp500_tickers_n(n: int) -> List[str]:
    """Return first N tickers from the standard 100-ticker list (nested subsets)."""
    full = get_sp500_tickers(100)
    return full[:n]
