"""
GBS Roots clustering algorithm.

Recursive cluster extraction using WEIGHTED density:
1. Sample H = floor(0.6 * N * ln(N)) threshold patterns
2. Sort samples by WD(S) descending; put max-WD sample first
3. Greedily build cluster union: start with max-WD sample; add any
   subsequent sample that is DISJOINT and IMPROVES the union's WD
4. Remove resulting cluster; repeat
5. Stopping: 2 vertices remain → final cluster

ASSUMPTION: "Queue order" for subsequent samples is WD-descending.
Algorithm 1 ambiguity resolved: WD improvement is the sole break condition.

Confirmed: budget_factor=0.6, threshold detection. Paper methodology step 7.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional

from src.gbs_sampling import (
    gbs_sample_threshold, clicks_to_subsets, filter_valid_samples,
    compute_weighted_density, compute_n_bar, compute_sample_budget,
)
from src.config import GBS_ROOTS_BUDGET


def gbs_roots_cluster(
    A: np.ndarray,
    budget_factor: float = GBS_ROOTS_BUDGET,
    loss: float = 0.0,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    GBS Roots recursive clustering on adjacency matrix A.

    Args:
        A: N x N non-negative symmetric adjacency matrix
        budget_factor: F in H = floor(F * N * ln(N)); default 0.6 (confirmed)
        loss: photon loss rate [0, 1)
        seed: random seed

    Returns:
        Exhaustive, disjoint partition; each cluster is sorted original indices.
    """
    N = A.shape[0]
    if N <= 0:
        return []
    if N == 1:
        return [[0]]
    if N == 2:
        return [[0, 1]]

    active = list(range(N))
    clusters: List[List[int]] = []
    rng_state = np.random.RandomState(seed if seed is not None else 0)

    while len(active) > 2:
        n_active = len(active)
        A_sub = A[np.ix_(active, active)]
        n_bar = compute_n_bar(n_active)
        H = compute_sample_budget(budget_factor, n_active)

        seed_i = int(rng_state.randint(0, 2**31 - 1))
        raw = gbs_sample_threshold(A_sub, n_bar, H, loss=loss, seed=seed_i)
        subsets_local = clicks_to_subsets(raw)
        valid = filter_valid_samples(subsets_local, min_size=2)

        if not valid:
            clusters.append(sorted(active))
            active = []
            break

        # Sort by WD descending (ASSUMPTION: queue order = WD-descending)
        wd_pairs = [(s, compute_weighted_density(s, A_sub)) for s in valid]
        wd_pairs.sort(key=lambda x: x[1], reverse=True)

        # Start cluster from max-WD sample
        cluster_local = list(wd_pairs[0][0])
        cluster_wd = wd_pairs[0][1]
        cluster_set = set(cluster_local)

        # Greedy union: add disjoint samples that improve WD
        for sample, _wd in wd_pairs[1:]:
            sample_set = set(sample)
            if cluster_set.isdisjoint(sample_set):
                union = cluster_local + list(sample_set)
                union_wd = compute_weighted_density(union, A_sub)
                if union_wd > cluster_wd + 1e-12:
                    cluster_local = union
                    cluster_wd = union_wd
                    cluster_set = set(cluster_local)

        # Map to original indices
        cluster_orig = [active[i] for i in cluster_local]
        remaining_orig = [active[i] for i in range(n_active)
                          if i not in cluster_set]

        # No-singleton rule
        if len(remaining_orig) == 1:
            cluster_orig.extend(remaining_orig)
            clusters.append(sorted(cluster_orig))
            active = []
            break

        clusters.append(sorted(cluster_orig))
        active = remaining_orig

    if len(active) == 2:
        clusters.append(sorted(active))
    elif len(active) == 1:
        if clusters:
            clusters[-1] = sorted(clusters[-1] + active)
        else:
            clusters.append(active)

    return clusters
