"""RP-cluster boundary regression tests.

Motivated by a real failure (B. subtilis, E. coli dual-anchor COA plots) where
the bootstrap consensus "RP cluster" engulfed 61-83% of the genome. Root cause:
the bootstrap replicate (`_bootstrap_rp_reference`) fit the RP covariance from
ALL resampled RP genes, while the single-shot path fits from a dense core. In a
low-dimensional COA space a few compositionally drifted RP genes inflate the
covariance and the chi-squared ellipse swallows the genome.

These tests lock in:
  1. the bootstrap replicate now performs dense-core selection before fitting,
     so a tight, RP-enriched cluster results even when drifted RP genes exist;
  2. without the drifted genes the boundary is essentially unchanged (no
     regression on clean data).
"""

import numpy as np

from codonpipe.modules import cluster_stability as CS


def _synthetic_genome(n_bulk=2000, n_rp_core=40, n_rp_drift=20, seed=0):
    """A 2-D COA-like space: a tight RP core offset from a large bulk, plus a
    handful of 'drifted' RP genes sitting out in the bulk. Returns
    (X, gene_ids, rp_indices)."""
    rng = np.random.default_rng(seed)
    bulk = rng.normal([0.0, 0.0], 0.30, (n_bulk, 2))
    rp_core = rng.normal([1.2, 1.2], 0.05, (n_rp_core, 2))      # tight, separated
    rp_drift = rng.normal([0.0, 0.0], 0.30, (n_rp_drift, 2))    # drifted into bulk
    X = np.vstack([bulk, rp_core, rp_drift])
    gene_ids = [f"g{i}" for i in range(len(X))]
    rp_indices = np.arange(n_bulk, n_bulk + n_rp_core + n_rp_drift)
    return X, gene_ids, rp_indices


def _consensus(X, gene_ids, rp_indices, n_axes=2, B=60):
    freq = np.zeros(len(gene_ids))
    for b in range(B):
        cl, _ = CS._bootstrap_rp_reference(X, gene_ids, rp_indices, n_axes,
                                           multiplier=0.0, seed=b)
        freq += np.array([1.0 if g in cl else 0.0 for g in gene_ids])
    return freq / B


class TestBootstrapBoundary:
    def test_dense_core_keeps_cluster_compact_with_drifted_rp(self):
        X, gene_ids, rp_idx = _synthetic_genome()
        freq = _consensus(X, gene_ids, rp_idx)
        core = freq >= 0.5
        frac = core.mean()
        # With dense-core fitting the consensus core stays a small slice of the
        # genome even though a third of the RP genes are drifted into the bulk.
        assert frac < 0.10, f"core spans {frac:.1%} of genome (expected < 10%)"

    def test_cluster_is_rp_enriched(self):
        X, gene_ids, rp_idx = _synthetic_genome()
        is_rp = np.zeros(len(gene_ids), bool)
        is_rp[rp_idx] = True
        freq = _consensus(X, gene_ids, rp_idx)
        core = freq >= 0.5
        a = int((core & is_rp).sum()); b = int((core & ~is_rp).sum())
        # The core should be dominated by RP genes, not bulk genes.
        assert a > 0
        assert a >= b, f"core has {a} RP vs {b} non-RP (expected RP-dominated)"

    def test_clean_rp_unaffected(self):
        # No drifted RP genes: the core should still be compact and RP-rich,
        # confirming the dense-core step does not harm the clean case.
        X, gene_ids, rp_idx = _synthetic_genome(n_rp_drift=0)
        is_rp = np.zeros(len(gene_ids), bool); is_rp[rp_idx] = True
        freq = _consensus(X, gene_ids, rp_idx)
        core = freq >= 0.5
        assert core.mean() < 0.10
        assert (core & is_rp).sum() >= (core & ~is_rp).sum()

    def test_select_rp_dense_core_is_importable(self):
        # The bootstrap path depends on the single-shot dense-core helper;
        # guard against the import being dropped.
        assert hasattr(CS, "_select_rp_dense_core")
