"""
GBS Boost clustering algorithm.

Recursive cluster extraction:
1. Sample H = floor(0.7 * N * ln(N)) threshold patterns
2. Choose sample with maximum UNWEIGHTED density
   (tie-break: global intra-cluster correlation)
3. Remove those vertices; repeat
4. Stopping rules:
   - Would leave 1 vertex → include it in current cluster (entire residual becomes final)
   - 2 vertices remain → they form the final cluster

Confirmed parameters: budget_factor=0.7, threshold detection, loss=0.
Paper reference: Methodology step 6; sample budget Appendix G.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional

from src.gbs_sampling import (
    gbs_sample_threshold, clicks_to_subsets, filter_valid_samples,
    compute_unweighted_density, global_cluster_correlation,
    compute_n_bar, compute_sample_budget,
)
from src.config import GBS_BOOST_BUDGET


def gbs_boost_cluster(
    A: np.ndarray,
    budget_factor: float = GBS_BOOST_BUDGET,
    loss: float = 0.0,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    GBS Boost recursive clustering on adjacency matrix A.

    Args:
        A: N x N non-negative symmetric adjacency matrix
        budget_factor: F in H = floor(F * N * ln(N)); default 0.7 (confirmed)
        loss: photon loss rate [0, 1); 0 for lossless (exp_1, exp_3-5)
        seed: random seed for reproducibility

    Returns:
        Partition as list of clusters; each cluster is a sorted list of
        original vertex indices in [0, N-1]. Exhaustive, disjoint.
    """
    N = A.shape[0]
    if N <= 0:
        return []
    if N == 1:
        return [[0]]
    if N == 2:
        return [[0, 1]]

    # active: indices in the ORIGINAL matrix that are still unassigned
    active = list(range(N))
    clusters: List[List[int]] = []
    rng_state = np.random.RandomState(seed if seed is not None else 0)

    while len(active) > 2:
        n_active = len(active)
        # Submatrix restricted to active nodes
        A_sub = A[np.ix_(active, active)]
        n_bar = compute_n_bar(n_active)
        H = compute_sample_budget(budget_factor, n_active)

        seed_i = int(rng_state.randint(0, 2**31 - 1))
        raw = gbs_sample_threshold(A_sub, n_bar, H, loss=loss, seed=seed_i)
        subsets_local = clicks_to_subsets(raw)
        valid = filter_valid_samples(subsets_local, min_size=2)

        if not valid:
            # No valid samples → entire residual is the final cluster
            clusters.append(sorted(active))
            active = []
            break

        # Choose max-density sample; tie-break by intra-cluster correlation
        best_local: Optional[List[int]] = None
        best_density = -1.0
        best_corr = -np.inf

        for subset_local in valid:
            d = compute_unweighted_density(subset_local, A_sub)
            if d > best_density + 1e-12:
                best_density = d
                best_local = subset_local
                best_corr = global_cluster_correlation(subset_local, A_sub)
            elif abs(d - best_density) <= 1e-12:
                c = global_cluster_correlation(subset_local, A_sub)
                if c > best_corr:
                    best_local = subset_local
                    best_corr = c

        if best_local is None:
            clusters.append(sorted(active))
            active = []
            break

        # Map local indices back to original
        cluster_orig = [active[i] for i in best_local]
        best_set = set(best_local)
        remaining_orig = [active[i] for i in range(n_active) if i not in best_set]

        # Stopping rule: if removing this cluster would leave exactly 1 vertex,
        # absorb that vertex into the current cluster and stop.
        if len(remaining_orig) == 1:
            cluster_orig.extend(remaining_orig)
            clusters.append(sorted(cluster_orig))
            active = []
            break

        clusters.append(sorted(cluster_orig))
        active = remaining_orig

    # Handle any remaining vertices
    if len(active) == 2:
        clusters.append(sorted(active))
    elif len(active) == 1:
        if clusters:
            clusters[-1] = sorted(clusters[-1] + active)
        else:
            clusters.append(active)

    return clusters
