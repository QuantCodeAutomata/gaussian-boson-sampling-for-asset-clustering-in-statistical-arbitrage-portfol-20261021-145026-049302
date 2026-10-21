"""
QIC-GBS: Classical independent-pairs-and-singles approximation of GBS.

Uses edge-sampling from adjacency weights (Poisson mechanism) and applies
GBS Boost extraction logic. Budget factor = 1.0 (H = N ln N).

ASSUMPTION: QIC-GBS uses the same fast sampling backend as GBS Boost/Roots
but with F=1.0. This matches the paper's QIC-GBS description as a classical
approximation to GBS sampling.

Confirmed: budget_factor=1.0. Paper methodology step 8.
"""
from __future__ import annotations
import numpy as np
from typing import List, Optional

from src.gbs_sampling import (
    gbs_sample_threshold, clicks_to_subsets, filter_valid_samples,
    compute_unweighted_density, global_cluster_correlation,
    compute_n_bar, compute_sample_budget,
)
from src.config import GBS_QIC_BUDGET


def qic_gbs_cluster(
    A: np.ndarray,
    budget_factor: float = GBS_QIC_BUDGET,
    loss: float = 0.0,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """
    QIC-GBS clustering: classical edge-Poisson sampling + Boost extraction.

    Args:
        A: N x N non-negative symmetric adjacency matrix
        budget_factor: F in H = floor(F * N * ln(N)); default 1.0 (confirmed)
        loss: photon loss rate
        seed: random seed

    Returns:
        Exhaustive disjoint partition (list of sorted vertex lists).
    """
    from src.gbs_boost import gbs_boost_cluster
    # QIC-GBS uses Boost extraction with its own (larger) sample budget
    # Reuse gbs_boost_cluster with F=1.0
    return gbs_boost_cluster(A, budget_factor=budget_factor, loss=loss, seed=seed)
