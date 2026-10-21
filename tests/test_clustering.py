"""Tests for clustering algorithms."""
import numpy as np
import pytest
from src.gbs_boost import gbs_boost_cluster
from src.gbs_roots import gbs_roots_cluster
from src.qic_gbs import qic_gbs_cluster
from src.spectral import spectral_cluster
from src.sponge import sponge_cluster


def make_adj(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.random((n, n)) * 0.5
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def make_corr(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((20, n))
    C = np.corrcoef(X.T)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def check_valid_partition(partition, N):
    """Partition covers all N nodes exactly once, no singletons except edge cases."""
    all_nodes = set()
    for c in partition:
        for node in c:
            assert node not in all_nodes, f"Duplicate node {node}"
            assert 0 <= node < N, f"Node {node} out of range [0, {N})"
            all_nodes.add(node)
    assert all_nodes == set(range(N)), f"Missing nodes: {set(range(N)) - all_nodes}"


def test_gbs_boost_exhaustive():
    """GBS Boost should partition all N vertices."""
    N = 8
    A = make_adj(N)
    part = gbs_boost_cluster(A, seed=0)
    check_valid_partition(part, N)


def test_gbs_boost_no_singletons():
    """Clusters should have size >= 2 (no-singleton rule)."""
    N = 10
    A = make_adj(N)
    part = gbs_boost_cluster(A, seed=0)
    for c in part:
        assert len(c) >= 2, f"Singleton cluster found: {c}"


def test_gbs_boost_two_nodes():
    """Two-node graph returns one cluster of two."""
    A = np.array([[0, 0.5], [0.5, 0]])
    part = gbs_boost_cluster(A, seed=0)
    check_valid_partition(part, 2)


def test_gbs_boost_zero_adj():
    """Zero adjacency should still produce valid partition."""
    N = 6
    A = np.zeros((N, N))
    part = gbs_boost_cluster(A, seed=0)
    check_valid_partition(part, N)


def test_gbs_roots_exhaustive():
    N = 8
    A = make_adj(N)
    part = gbs_roots_cluster(A, seed=0)
    check_valid_partition(part, N)


def test_gbs_roots_no_singletons():
    N = 10
    A = make_adj(N)
    part = gbs_roots_cluster(A, seed=0)
    for c in part:
        assert len(c) >= 2


def test_qic_gbs_exhaustive():
    N = 8
    A = make_adj(N)
    part = qic_gbs_cluster(A, seed=0)
    check_valid_partition(part, N)


def test_spectral_cluster():
    N = 10; K = 3
    C = make_corr(N)
    part = spectral_cluster(C, K, seed=0)
    assert len(part) > 0
    all_nodes = [n for c in part for n in c]
    assert len(all_nodes) == N


def test_sponge_cluster():
    N = 10; K = 3
    C = make_corr(N)
    part = sponge_cluster(C, K, seed=0)
    assert len(part) > 0
    all_nodes = [n for c in part for n in c]
    assert len(all_nodes) == N


def test_all_methods_consistent_N():
    """All methods should produce partitions covering N nodes."""
    N = 12
    A = make_adj(N)
    C = A + np.eye(N); np.fill_diagonal(C, 1.0)
    K = 3

    for method, clusterer in [
        ("GBS Boost", lambda: gbs_boost_cluster(A, seed=0)),
        ("GBS Roots", lambda: gbs_roots_cluster(A, seed=0)),
        ("QIC-GBS", lambda: qic_gbs_cluster(A, seed=0)),
        ("Spectral", lambda: spectral_cluster(C, K, seed=0)),
        ("SPONGE", lambda: sponge_cluster(C, K, seed=0)),
    ]:
        part = clusterer()
        all_n = [n for c in part for n in c]
        assert len(all_n) == N, f"{method}: got {len(all_n)} nodes, expected {N}"
