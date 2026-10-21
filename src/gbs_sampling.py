"""
Gaussian Boson Sampling (GBS) simulation for financial clustering.

Two backends are available:
1. FAST (default): numpy-based edge-sampling approximation of GBS.
   This implements the independent-pairs-and-singles classical approximation
   (equivalent to QIC-GBS baseline), which runs instantly without JIT compilation.
   ASSUMPTION: This is a classical approximation, not true quantum simulation.

2. THEWALRUS (optional): Uses thewalrus.samples.torontonian_sample_graph.
   Requires ~5-10 minute numba JIT warm-up on first call. Set
   USE_THEWALRUS=True in the environment to enable.

The paper's GBS Boost and GBS Roots algorithms differ only in their
EXTRACTION LOGIC, not their sampling distribution. Both use H samples
from the same distribution and select different subsets.

Reference paper: "Gaussian Boson Sampling for Asset Clustering in
Statistical Arbitrage Portfolios"
"""
from __future__ import annotations
import math
import os
import numpy as np
from typing import List, Optional, Tuple

USE_THEWALRUS = os.environ.get("USE_THEWALRUS", "0") == "1"


# ---------------------------------------------------------------------------
# Budget and mean photon number utilities (confirmed from paper)
# ---------------------------------------------------------------------------

def compute_n_bar(N: int) -> float:
    """
    Target mean photon number: n_bar = sqrt(N).

    Confirmed from paper methodology: "set the target mean photon number
    to n_bar = sqrt(dim(A))".
    """
    return float(np.sqrt(max(N, 1)))


def compute_sample_budget(budget_factor: float, N: int) -> int:
    """
    Integer sample budget H = floor(F * N * ln(N)).

    ASSUMPTION: Floor rounding (paper does not specify; floor is conservative).

    Args:
        budget_factor: F coefficient (0.7 for Boost, 0.6 for Roots, 1.0 for QIC)
        N: number of modes in current residual graph

    Returns:
        Integer H >= 1
    """
    if N < 2:
        return 1
    return max(1, int(math.floor(budget_factor * N * math.log(N))))


# ---------------------------------------------------------------------------
# Fast numpy GBS approximation (default backend)
# ---------------------------------------------------------------------------

def _fast_gbs_threshold(
    A: np.ndarray,
    n_mean: float,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Fast classical approximation of threshold GBS sampling.

    Implementation of independent-pairs-and-singles (QIC-GBS approximation):
    - Each edge (i,j) is independently included with probability ∝ A[i,j]^2
    - A node is clicked (threshold=1) if ANY of its sampled edges is included
    - Edge inclusion probabilities are normalised so expected clicks ≈ n_mean

    This matches the classical QIC-GBS distribution described in the paper
    (Appendix / QIC-GBS reference) and avoids numba JIT compilation overhead.

    GBS Boost and GBS Roots use the same sampling distribution but apply
    different cluster extraction algorithms on the resulting samples.

    ASSUMPTION: Using classical approximation instead of true quantum GBS
    simulation to avoid thewalrus numba compilation overhead.

    Args:
        A: N x N symmetric non-negative adjacency matrix
        n_mean: target expected click count
        n_samples: number of threshold patterns to generate
        rng: numpy random generator

    Returns:
        (n_samples, N) binary click pattern array
    """
    N = A.shape[0]
    if N < 2:
        return np.zeros((n_samples, N), dtype=np.int32)

    # Compute edge squared weights for inclusion probabilities
    # P(edge i,j clicked) ∝ A[i,j]^2
    A_sq = A ** 2
    # Upper triangular edge indices
    ui, vi = np.triu_indices(N, k=1)
    edge_weights = A_sq[ui, vi]
    total_weight = edge_weights.sum()

    if total_weight <= 1e-12:
        # No edges: return empty patterns
        return np.zeros((n_samples, N), dtype=np.int32)

    # Scale so expected active nodes ≈ n_mean
    # Expected active nodes = sum_i [1 - prod_j (1 - p_ij)]
    # Approximation: p_ij = scale * A_sq[i,j], expected ≈ scale * total_weight_per_node
    # Use: scale * sum(A_sq above diagonal) * 2 / N ≈ n_mean
    scale = n_mean * N / (2.0 * total_weight + 1e-12)
    # Clip probabilities to [0, 0.95]
    p_edges = np.minimum(scale * edge_weights, 0.95)

    # Vectorized sampling across n_samples
    # Shape: (n_samples, n_edges)
    rand_edges = rng.random((n_samples, len(ui)))
    clicked_edges = rand_edges < p_edges[np.newaxis, :]  # (n_samples, n_edges)

    # Each node clicked if any adjacent edge is clicked
    samples = np.zeros((n_samples, N), dtype=np.int32)
    for k in range(len(ui)):
        if clicked_edges[:, k].any():
            mask = clicked_edges[:, k]
            samples[mask, ui[k]] = 1
            samples[mask, vi[k]] = 1

    return samples


# ---------------------------------------------------------------------------
# Thewalrus backend (optional, requires JIT warm-up)
# ---------------------------------------------------------------------------

def _thewalrus_gbs_threshold(
    A: np.ndarray,
    n_mean: float,
    n_samples: int,
    loss: float,
    seed: Optional[int],
) -> np.ndarray:
    """Thewalrus torontonian_sample_graph backend (requires numba JIT compilation)."""
    from thewalrus.samples import torontonian_sample_graph

    if seed is not None:
        np.random.seed(seed)

    eta = 1.0 - loss
    A_eff = eta * A
    A_eff = (A_eff + A_eff.T) / 2.0
    A_eff = np.maximum(A_eff, 0.0)

    svd_max = np.linalg.norm(A_eff, ord=2)
    if svd_max <= 1e-12:
        return np.zeros((n_samples, A.shape[0]), dtype=np.int32)
    if svd_max >= 0.999:
        A_eff = A_eff * 0.999 / svd_max

    raw = torontonian_sample_graph(A_eff, n_mean, n_samples)
    return np.array(raw, dtype=np.int32)


# ---------------------------------------------------------------------------
# Public sampling interface
# ---------------------------------------------------------------------------

def gbs_sample_threshold(
    A: np.ndarray,
    n_mean: float,
    n_samples: int,
    loss: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample threshold click patterns from GBS distribution on graph A.

    Selects backend based on USE_THEWALRUS environment variable:
    - 0 (default): fast numpy approximation (instant, documented above)
    - 1: thewalrus torontonian_sample_graph (requires JIT warm-up)

    For loss > 0: applies A_eff = (1-loss)*A approximation (small-squeezing).
    ASSUMPTION: A_eff = eta*A is the small-squeezing loss approximation.

    Args:
        A: N x N symmetric non-negative adjacency matrix
        n_mean: target mean photon number / expected clicks
        n_samples: number of threshold click patterns
        loss: photon loss rate [0, 1)
        seed: random seed for reproducibility

    Returns:
        (n_samples, N) int32 array of 0/1 click patterns
    """
    N = A.shape[0]
    if N < 2 or n_samples <= 0:
        return np.zeros((max(n_samples, 0), N), dtype=np.int32)

    # Apply loss
    eta = 1.0 - loss
    A_eff = eta * np.maximum((A + A.T) / 2.0, 0.0)

    if USE_THEWALRUS:
        return _thewalrus_gbs_threshold(A_eff, n_mean, n_samples, 0.0, seed)

    rng = np.random.default_rng(seed)
    return _fast_gbs_threshold(A_eff, n_mean, n_samples, rng)


# ---------------------------------------------------------------------------
# Subset utilities
# ---------------------------------------------------------------------------

def clicks_to_subsets(samples: np.ndarray) -> List[List[int]]:
    """Convert (n_samples, N) click array to list of vertex-index lists."""
    return [list(np.where(row)[0]) for row in samples]


def filter_valid_samples(subsets: List[List[int]], min_size: int = 2) -> List[List[int]]:
    """
    Discard patterns with fewer than min_size clicks.

    Implements paper post-selection: "Discard samples with fewer than
    two distinct clicks."
    """
    return [s for s in subsets if len(s) >= min_size]


# ---------------------------------------------------------------------------
# Graph metrics (used by Boost, Roots, QIC-GBS)
# ---------------------------------------------------------------------------

def compute_unweighted_density(subset: List[int], A: np.ndarray) -> float:
    """
    Unweighted graph density of the induced subgraph on vertex set S.

    density(S) = |{(i,j) in S : A[i,j] > 0}| / (|S|*(|S|-1)/2)

    Args:
        subset: list of vertex indices
        A: N x N adjacency matrix (edge exists if A[i,j] > 0)

    Returns:
        Density in [0, 1]; 0 if |subset| < 2
    """
    n = len(subset)
    if n < 2:
        return 0.0
    max_edges = n * (n - 1) / 2
    if max_edges == 0:
        return 0.0
    s = np.array(subset)
    sub = A[np.ix_(s, s)]
    edge_count = np.sum(sub > 0) / 2  # symmetric, count upper triangle
    return float(edge_count / max_edges)


def compute_weighted_density(subset: List[int], A: np.ndarray) -> float:
    """
    Weighted density of the induced subgraph on vertex set S.

    WD(S) = 2 * sum_{u<v in S} w_{uv} / (|S| * (|S|-1))

    Confirmed from paper Appendix L (Metrics section).

    Args:
        subset: vertex indices
        A: weighted adjacency matrix

    Returns:
        Weighted density; 0 if |subset| < 2
    """
    n = len(subset)
    if n < 2:
        return 0.0
    denom = n * (n - 1)
    s = np.array(subset)
    sub = A[np.ix_(s, s)]
    # Sum upper triangle only (each edge counted once)
    weight_sum = np.triu(sub, k=1).sum()
    return float(2.0 * weight_sum / denom)


def global_cluster_correlation(subset: List[int], C: np.ndarray) -> float:
    """
    Mean pairwise intra-cluster correlation, used as tie-breaker in Boost.

    Args:
        subset: vertex indices
        C: correlation matrix

    Returns:
        Mean pairwise correlation; 0 if |subset| < 2
    """
    n = len(subset)
    if n < 2:
        return 0.0
    s = np.array(subset)
    sub = C[np.ix_(s, s)]
    vals = sub[np.triu_indices(n, k=1)]
    return float(np.mean(vals)) if len(vals) > 0 else 0.0
