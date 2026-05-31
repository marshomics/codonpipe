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


class TestGapBoundary:
    """The RP boundary cuts at the largest gap between the compact cohort and
    the distant outliers, rather than at a fixed chi-squared quantile."""

    def test_cuts_at_cohort_outlier_gap(self):
        # Compact cohort 0.5-3.9, then a clear jump to 5.6+ (the outliers).
        cohort = np.linspace(0.5, 3.9, 40)
        outliers = np.array([5.6, 6.0, 7.1, 8.2, 9.5, 11.0, 12.3])
        d = np.concatenate([cohort, outliers])
        thr, info = CS._gap_boundary(d, n_axes=3)
        # The boundary must sit in the gap (between 3.9 and 5.6), keeping the
        # whole cohort and excluding every outlier.
        assert 3.9 <= thr < 5.6, f"threshold {thr} not in the cohort/outlier gap"
        assert int((d <= thr).sum()) == len(cohort)

    def test_wide_gap_can_override_chi2_ceiling(self):
        import numpy as np
        from codonpipe.modules.cluster_stability import _chi2_threshold
        ceil = _chi2_threshold(3, CS._GAP_CEIL_CHI2_P)
        # Put the cohort/outlier gap ABOVE the chi-squared p999 ceiling so the
        # override path is exercised; the cohort still ends just past the ceiling.
        cohort = np.linspace(0.5, ceil + 0.6, 45)
        outliers = np.array([ceil + 5.0, ceil + 7.0, ceil + 9.0])
        d = np.concatenate([cohort, outliers])
        thr, info = CS._gap_boundary(d, n_axes=3)
        assert thr > ceil, "a wide gap above the ceiling should extend the boundary"
        assert thr <= _chi2_threshold(3, CS._GAP_OVERRIDE_HARD_CEIL_CHI2_P)
        assert int((d <= thr).sum()) == len(cohort)

    def test_no_clean_gap_stays_bounded(self):
        import numpy as np
        from codonpipe.modules.cluster_stability import _chi2_threshold
        # Smoothly decaying distances with no discontinuity: the boundary must
        # not run away; it stays within [floor, ceil].
        d = np.sort(np.abs(np.random.default_rng(0).normal(2.0, 1.0, 60)))
        thr, info = CS._gap_boundary(d, n_axes=3)
        floor = _chi2_threshold(3, CS._GAP_FLOOR_CHI2_P)
        hard = _chi2_threshold(3, CS._GAP_OVERRIDE_HARD_CEIL_CHI2_P)
        assert floor <= thr <= hard

    def test_small_rp_set_falls_back_to_chi2(self):
        d = np.array([1.0, 1.5, 2.0])  # < 4 points
        thr, info = CS._gap_boundary(d, n_axes=3)
        assert info["method"] == "chi2_fallback_small_rp"
        assert np.isfinite(thr) and thr > 0
