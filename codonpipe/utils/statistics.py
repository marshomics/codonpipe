"""Shared statistical utilities for CodonPipe."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction on an array of p-values.

    Args:
        p_values: 1-D array of raw p-values (need not be sorted).

    Returns:
        Array of adjusted p-values (same order as input), capped at 1.0
        and monotonically non-decreasing when sorted by raw p-value.
    """
    p_values = np.asarray(p_values, dtype=float)
    n_total = len(p_values)
    if n_total == 0:
        return p_values.copy()

    # Correct only over finite p-values. NaN/inf entries must NOT count
    # toward the test family size or steal a rank: np.argsort sorts NaN to
    # the end, so including them inflates n and shifts every finite entry's
    # rank, corrupting all adjusted values (verified against statsmodels).
    # NaN p-values are passed through unchanged.
    adjusted = np.full(n_total, np.nan, dtype=float)
    finite_mask = np.isfinite(p_values)
    n = int(finite_mask.sum())
    if n == 0:
        return adjusted

    finite_idx = np.flatnonzero(finite_mask)
    p_finite = p_values[finite_mask]

    # Sort by raw p-value
    order = np.argsort(p_finite)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # Adjust: p_adj_i = p_i * n / rank_i
    adj_finite = np.minimum(p_finite * n / ranks, 1.0)

    # Enforce monotonicity: walk backwards through sorted order so that
    # p_adj[i] <= p_adj[i+1] when sorted by raw p-value.
    sorted_adj = adj_finite[order]
    for i in range(n - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adj_finite[order] = sorted_adj

    adjusted[finite_idx] = adj_finite
    return adjusted
