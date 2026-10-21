"""Tests for GBS sampling and related utilities."""
import numpy as np
import pytest
from src.gbs_sampling import (
    gbs_sample_threshold, compute_unweighted_density,
    compute_weighted_density, global_cluster_correlation,
    compute_n_bar, compute_sample_budget,
)


def make_test_adjacency(n: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.random((n, n))
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)
    return A


def test_gbs_sample_returns_binary():
    """GBS samples should be binary {0, 1}."""
    A = make_test_adjacency(6)
    samples = gbs_sample_threshold(A, n_mean=2.0, n_samples=10, seed=42)
    assert samples.shape == (10, 6)
    assert set(np.unique(samples)).issubset({0, 1})


def test_gbs_sample_zero_matrix():
    """Zero adjacency should return all-zero samples or near-zero click rate."""
    A = np.zeros((5, 5))
    samples = gbs_sample_threshold(A, n_mean=1.0, n_samples=20, seed=0)
    assert samples.shape == (20, 5)
    assert np.all(samples == 0), "Zero adjacency should give zero clicks"


def test_gbs_sample_n_mean_scaling():
    """Higher n_mean should produce more clicks on average."""
    A = make_test_adjacency(10)
    s1 = gbs_sample_threshold(A, n_mean=1.0, n_samples=100, seed=0)
    s2 = gbs_sample_threshold(A, n_mean=5.0, n_samples=100, seed=0)
    assert s2.mean() >= s1.mean(), "Higher n_mean should increase expected clicks"


def test_gbs_sample_single_node():
    """Single-node graph returns empty (no edges)."""
    A = np.zeros((1, 1))
    samples = gbs_sample_threshold(A, n_mean=0.5, n_samples=5)
    assert samples.shape == (5, 1)


def test_n_bar():
    """n_bar = sqrt(N) for N=4 -> 2.0"""
    assert abs(compute_n_bar(4) - 2.0) < 1e-10
    assert abs(compute_n_bar(9) - 3.0) < 1e-10
    assert compute_n_bar(1) == 1.0


def test_sample_budget_floor_rounding():
    """H = floor(F * N * ln(N))."""
    import math
    N, F = 10, 0.7
    H = compute_sample_budget(F, N)
    expected = int(math.floor(F * N * math.log(N)))
    assert H == expected


def test_sample_budget_minimum():
    """Budget is at least 1."""
    assert compute_sample_budget(0.01, 2) >= 1


def test_unweighted_density_complete():
    """Complete graph should have density 1."""
    A = np.ones((4, 4)); np.fill_diagonal(A, 0)
    d = compute_unweighted_density([0, 1, 2, 3], A)
    assert abs(d - 1.0) < 1e-10


def test_unweighted_density_range():
    """Density should be in [0, 1]."""
    A = make_test_adjacency(8)
    subset = [0, 1, 2, 4]
    d = compute_unweighted_density(subset, A)
    assert 0.0 <= d <= 1.0


def test_unweighted_density_single_node():
    """Single-node subset -> density 0."""
    A = make_test_adjacency(5)
    d = compute_unweighted_density([2], A)
    assert d == 0.0


def test_weighted_density_range():
    """WD should be in [0, max_weight] for non-negative A."""
    A = make_test_adjacency(6)
    wd = compute_weighted_density([0, 1, 2, 3], A)
    assert wd >= 0.0


def test_weighted_density_symmetry():
    """WD should be same regardless of vertex ordering."""
    A = make_test_adjacency(5)
    wd1 = compute_weighted_density([0, 1, 2], A)
    wd2 = compute_weighted_density([2, 0, 1], A)
    assert abs(wd1 - wd2) < 1e-12


def test_gbs_reproducible():
    """Same seed should produce identical samples."""
    A = make_test_adjacency(8)
    s1 = gbs_sample_threshold(A, 2.0, 20, seed=99)
    s2 = gbs_sample_threshold(A, 2.0, 20, seed=99)
    np.testing.assert_array_equal(s1, s2)
