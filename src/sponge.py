"""
Symmetric SPONGE (Signed Positive Over Negative Generalised Eigenproblem).

Implements the symmetric signed-Laplacian generalised eigenproblem for
K-way clustering of signed correlation networks.

ASSUMPTION: Laplacian construction uses L+ = D+ - A+ and L- = D- + A-
where A+ = max(C, 0) - I (positive edges), A- = min(C, 0) (negative edges).
Regularisation tau=0.1 applied to both D+ and D-. K-means++ on eigenvectors.

ASSUMPTION: Symmetric SPONGE uses same dynamic K estimation as Spectral.
"""
from __future__ import annotations
import numpy as np
from typing import List

from src.config import KMEANS_SEED, SPONGE_TAU


def sponge_cluster(C: np.ndarray, K: int, seed: int = KMEANS_SEED) -> List[List[int]]:
    """
    Symmetric SPONGE clustering on correlation matrix C.

    Solves: (L+ + tau*I) v = lambda (L- + tau*I) v (symmetric form)
    and applies K-means++ on K-1 non-trivial eigenvectors.

    ASSUMPTION: Regularisation tau=0.1 added to avoid singularity.

    Args:
        C: N x N correlation matrix
        K: number of clusters
        seed: K-means++ seed

    Returns:
        List of K cluster lists.
    """
    from sklearn.cluster import KMeans
    import scipy.linalg

    N = C.shape[0]
    if K >= N:
        return [[i] for i in range(N)]
    if K <= 1:
        return [list(range(N))]

    # Build positive and negative adjacency from correlation matrix
    # Remove diagonal (self-loops)
    C_off = C - np.diag(np.diag(C))
    A_pos = np.maximum(C_off, 0.0)
    A_neg = np.maximum(-C_off, 0.0)  # absolute values of negative entries

    # Degree matrices
    D_pos = np.diag(A_pos.sum(axis=1) + SPONGE_TAU)
    D_neg = np.diag(A_neg.sum(axis=1) + SPONGE_TAU)

    # Signed Laplacians
    L_pos = D_pos - A_pos
    L_neg = D_neg - A_neg  # positive-semi-definite with negatives as edges

    # Generalised eigenproblem: L_pos v = lambda L_neg v
    # Solve as L_pos v = lambda * L_neg v for smallest K-1 eigenvalues
    try:
        eigvals, eigvecs = scipy.linalg.eigh(
            L_pos, L_neg,
            subset_by_index=[0, min(K - 1, N - 1)],
            check_finite=False,
        )
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to standard spectral if generalised fails
        from src.spectral import spectral_cluster
        return spectral_cluster(C, K, seed)

    # Use first K-1 eigenvectors (skip trivial first eigenvector if needed)
    embedding = eigvecs[:, :K - 1] if eigvecs.shape[1] >= K - 1 else eigvecs

    # Normalise rows
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding_norm = embedding / norms

    # K-means++
    km = KMeans(n_clusters=K, init="k-means++", n_init=10, random_state=seed)
    labels = km.fit_predict(embedding_norm)

    clusters = [[] for _ in range(K)]
    for i, lb in enumerate(labels):
        clusters[lb].append(i)
    return [sorted(c) for c in clusters if c]
