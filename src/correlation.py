"""
Correlation matrix construction from residual returns.

5-day sample correlation with (ddof=1). Optional RMT cleaning.
Adjacency matrix A = max(0, C - I) (diagonal=0, negative=0).

Confirmed: 5-day lookback, sample correlation. Paper step 3.
ASSUMPTION: RMT cleaning not applied by default (paper is ambiguous on exact placement).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from src.config import CORRELATION_WINDOW


def compute_correlation_matrix(
    residual_returns: pd.DataFrame,
    window: Optional[int] = CORRELATION_WINDOW,
) -> np.ndarray:
    """
    Compute sample correlation matrix from the last `window` rows.

    Uses ddof=1 (confirmed from paper: "sample-correlation formula with
    denominator w-1").

    Args:
        residual_returns: (T, N) residual return DataFrame
        window: look-back in trading days (default 5)

    Returns:
        (N, N) correlation matrix clipped to [-1, 1]
    """
    if window is not None and len(residual_returns) > window:
        data = residual_returns.iloc[-window:]
    else:
        data = residual_returns

    N = data.shape[1]
    if len(data) < 2 or N < 2:
        return np.eye(N)

    arr = data.values.astype(float)
    # Remove any all-NaN columns
    valid = ~np.isnan(arr).all(axis=0)
    arr[:, ~valid] = 0.0

    C = np.corrcoef(arr.T)  # uses ddof=N-1 (equivalent to ddof=1 for each pair)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def correlation_to_adjacency(C: np.ndarray) -> np.ndarray:
    """
    Build GBS adjacency matrix from correlation matrix.

    A = max(0, C - I): zero diagonal, negative correlations set to zero.

    Confirmed from paper methodology step 4.

    Args:
        C: (N, N) correlation matrix

    Returns:
        (N, N) non-negative symmetric adjacency matrix
    """
    A = np.maximum(C - np.eye(C.shape[0]), 0.0)
    return A


def rmt_clean_correlation(C: np.ndarray, T: int) -> np.ndarray:
    """
    Optional RMT (Random Matrix Theory) cleaning of correlation matrix.

    Removes eigenvalues within the Marchenko-Pastur noise band.
    The market mode (largest eigenvalue) is also removed.

    ASSUMPTION: Applied before forming adjacency (paper step 3 is ambiguous).
    This is disabled by default.

    Args:
        C: (N, N) correlation matrix
        T: number of observations used (for q=N/T ratio)

    Returns:
        Cleaned (N, N) correlation matrix
    """
    N = C.shape[0]
    if N < 2 or T < 2:
        return C

    q = N / T
    # Marchenko-Pastur bounds
    lambda_plus = (1 + np.sqrt(q)) ** 2
    lambda_minus = (1 - np.sqrt(q)) ** 2

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Market mode: largest eigenvalue
    market_idx = np.argmax(eigenvalues)

    # Noise band: eigenvalues between lambda_minus and lambda_plus
    noise_mask = (eigenvalues >= lambda_minus) & (eigenvalues <= lambda_plus)
    noise_mask[market_idx] = True  # also remove market mode

    # Replace noise eigenvalues with mean of noise eigenvalues
    noise_mean = eigenvalues[noise_mask].mean() if noise_mask.any() else 1.0

    eigenvalues_clean = eigenvalues.copy()
    eigenvalues_clean[noise_mask] = noise_mean

    C_clean = (eigenvectors * eigenvalues_clean) @ eigenvectors.T
    # Rescale to unit diagonal
    d = np.sqrt(np.diag(C_clean))
    d[d == 0] = 1.0
    C_clean = C_clean / np.outer(d, d)
    C_clean = np.clip(C_clean, -1.0, 1.0)
    np.fill_diagonal(C_clean, 1.0)
    return C_clean
