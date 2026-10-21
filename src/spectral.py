"""
Spectral clustering with dynamic cluster count K.

K is estimated from eigenvalue spacings of the 20-day residual-return
correlation matrix using permutation null distribution (alpha=0.05, B=1000).

ASSUMPTION: Spectral embedding uses normalised Laplacian; ties broken by
smallest empirical spacing exceeding threshold.

Confirmed: 20-day window, alpha=0.05, B=1000, K-means++.
"""
from __future__ import annotations
import numpy as np
from typing import Optional

from src.config import (
    K_WINDOW, K_PERM_SAMPLES, K_ALPHA,
    MAX_CLUSTERS, MIN_CLUSTERS, KMEANS_SEED,
)
SPECTRAL_WINDOW = K_WINDOW  # alias for backward compat


def estimate_k_from_spacings(
    returns: np.ndarray,
    alpha: float = K_ALPHA,
    B: int = K_PERM_SAMPLES,
    seed: int = KMEANS_SEED,
) -> int:
    """
    Estimate K from eigenvalue spacings of the correlation matrix.

    Standardise each column, compute ordered eigenvalues, compute spacings
    s_k = lambda_k - lambda_{k+1}. Generate B permutation null spacings,
    use alpha quantile threshold. Count significant spacings.

    ASSUMPTION: Threshold is the (1-alpha) quantile of max null spacing
    (conservative; paper Appendix B).

    Args:
        returns: (T, N) return matrix; T >= N+1
        alpha: significance level
        B: number of permutation null replicates

    Returns:
        K in [MIN_CLUSTERS, MAX_CLUSTERS]
    """
    T, N = returns.shape
    if T < 2 or N < 2:
        return MIN_CLUSTERS

    # Standardise columns
    std = returns.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    Z = (returns - returns.mean(axis=0)) / std

    # Correlation eigenvalues descending
    C = np.corrcoef(Z.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(C))[::-1]

    # Spacings s_k = lambda_k - lambda_{k+1}
    n_eigs = min(len(eigenvalues), MAX_CLUSTERS + 1)
    spacings = eigenvalues[:n_eigs - 1] - eigenvalues[1:n_eigs]

    # Permutation null
    rng = np.random.default_rng(seed)
    null_max_spacings = np.zeros(B)
    for b in range(B):
        Zp = np.apply_along_axis(rng.permutation, 0, Z)
        Cp = np.corrcoef(Zp.T)
        ep = np.sort(np.linalg.eigvalsh(Cp))[::-1]
        ep = ep[:n_eigs]
        sp = ep[:-1] - ep[1:]
        null_max_spacings[b] = sp.max() if len(sp) > 0 else 0.0

    threshold = np.quantile(null_max_spacings, 1.0 - alpha)

    # Count significant spacings sequentially
    K = MIN_CLUSTERS
    for k, s in enumerate(spacings):
        if s > threshold:
            K = k + 2  # eigenvalues are 1-indexed; K >= 2
        else:
            break

    return int(np.clip(K, MIN_CLUSTERS, min(MAX_CLUSTERS, N - 1)))


def spectral_cluster(
    C: np.ndarray,
    K: int,
    seed: int = KMEANS_SEED,
) -> list:
    """
    Spectral clustering on correlation matrix C.

    Args:
        C: N x N correlation matrix
        K: number of clusters
        seed: K-means++ seed

    Returns:
        List of K cluster lists (sorted vertex indices).
    """
    from sklearn.cluster import SpectralClustering
    N = C.shape[0]
    if K >= N:
        return [[i] for i in range(N)]
    if K <= 1:
        return [list(range(N))]

    # Use correlation as affinity (shifted to [0,1])
    affinity = np.clip((C + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)

    sc = SpectralClustering(
        n_clusters=K,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    labels = sc.fit_predict(affinity)
    clusters = [[] for _ in range(K)]
    for i, lb in enumerate(labels):
        clusters[lb].append(i)
    return [sorted(c) for c in clusters if c]
