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
    n = len(p_values)
    if n == 0:
        return p_values.copy()

    # Sort by raw p-value
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # Adjust: p_adj_i = p_i * n / rank_i
    adjusted = np.minimum(p_values * n / ranks, 1.0)

    # Enforce monotonicity: walk backwards through sorted order so that
    # p_adj[i] <= p_adj[i+1] when sorted by raw p-value.
    sorted_adj = adjusted[order]
    for i in range(n - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])
    adjusted[order] = sorted_adj

    return adjusted
