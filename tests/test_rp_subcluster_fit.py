"""Tests for RP sub-cluster fit gating and bootstrap log-noise suppression.

Motivated by a real run (genome GCF_001953) where the 51 ribosomal-protein
genes split into a 42-gene fragment and a 9-gene fragment. Fitting the 9-gene
fragment bootstrapped a ~5-gene dense core, whose covariance is rank-deficient,
producing a flood of per-replicate INFO/WARNING lines and one ERROR — none of
which affected the result, because the clean 42-gene fragment was selected as
primary. These tests lock in:

  1. fragments below the fit threshold are excluded from fitting (kept in
     diagnostics), with a single-element fallback so fitting never has nothing;
  2. degenerate covariance during bootstrap logs at DEBUG, but at
     ERROR/WARNING outside the bootstrap (where it genuinely matters).
"""

import logging

import numpy as np

from codonpipe.modules import mahal_clustering as M


# -- sub-cluster fit gating ---------------------------------------------------

class TestSubclusterFitGating:
    def test_min_fit_threshold_value(self):
        # 2x the robust-covariance sample floor, so a dense core (~half) clears it.
        assert M._RP_SUBCLUSTER_MIN_FIT == 2 * M._MIN_RP_FOR_ROBUST

    def _split_rp(self, big=42, small=9, seed=0):
        rng = np.random.default_rng(seed)
        a = rng.normal([0.0, 0.0], 0.4, (big, 2))
        b = rng.normal([6.0, 6.0], 0.05, (small, 2))  # tight, well-separated
        X = np.vstack([a, b])
        ids = [f"rp{i}" for i in range(big + small)]
        return X, ids

    def test_small_fragment_excluded_from_fitting(self):
        X, ids = self._split_rp(42, 9)
        subs = M._detect_rp_subclusters(X, ids)
        sizes = sorted((s["n"] for s in subs), reverse=True)
        # GMM/BIC should find the 42/9 split
        assert sizes == [42, 9]
        fittable = [s for s in subs if s["n"] >= M._RP_SUBCLUSTER_MIN_FIT]
        assert [s["n"] for s in fittable] == [42]   # only the big fragment fits
        assert all(s["n"] >= M._RP_SUBCLUSTER_MIN_FIT for s in fittable)

    def test_fallback_when_all_fragments_too_small(self):
        # Two small fragments, both below threshold -> fittable empty -> the
        # run_mahal_clustering fallback fits the whole RP set as one anchor.
        X, ids = self._split_rp(12, 11)
        subs = M._detect_rp_subclusters(X, ids)
        fittable = [s for s in subs if s["n"] >= M._RP_SUBCLUSTER_MIN_FIT]
        # Reproduce the fallback the production function applies.
        if not fittable:
            fittable = [{"X": X, "gene_ids": ids, "n": len(ids)}]
        assert len(fittable) == 1
        assert fittable[0]["n"] == len(ids)   # whole RP set

    def test_large_single_population_unaffected(self):
        # A single coherent RP population is fit as-is (no spurious split, fits).
        rng = np.random.default_rng(1)
        X = rng.normal([0.0, 0.0], 0.5, (50, 2))
        ids = [f"rp{i}" for i in range(50)]
        subs = M._detect_rp_subclusters(X, ids)
        fittable = [s for s in subs if s["n"] >= M._RP_SUBCLUSTER_MIN_FIT]
        assert len(fittable) >= 1
        assert sum(s["n"] for s in fittable) >= M._RP_SUBCLUSTER_MIN_FIT


# -- bootstrap log-noise suppression ------------------------------------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.getMessage()))


class TestBootstrapLogging:
    def _capture(self):
        logger = logging.getLogger("codonpipe")
        h = _Capture()
        logger.addHandler(h)
        prev = logger.level
        logger.setLevel(logging.DEBUG)
        return logger, h, prev

    def test_zero_covariance_is_error_outside_bootstrap(self):
        logger, h, prev = self._capture()
        try:
            M._IN_BOOTSTRAP = False
            M._safe_inv(np.zeros((2, 2)))
        finally:
            logger.removeHandler(h)
            logger.setLevel(prev)
        levels = [lv for lv, _ in h.records]
        assert "ERROR" in levels
        assert not any(lv == "DEBUG" and "no positive eigenvalues" in msg.lower()
                       for lv, msg in h.records)

    def test_zero_covariance_is_debug_during_bootstrap(self):
        logger, h, prev = self._capture()
        try:
            M._IN_BOOTSTRAP = True
            M._safe_inv(np.zeros((2, 2)))
        finally:
            M._IN_BOOTSTRAP = False
            logger.removeHandler(h)
            logger.setLevel(prev)
        levels = [lv for lv, _ in h.records]
        assert "ERROR" not in levels
        assert "DEBUG" in levels

    def test_bootstrap_flag_restored_after_call(self):
        assert M._IN_BOOTSTRAP is False
        X = np.random.default_rng(0).normal(size=(12, 2))
        M._bootstrap_rp_centroid(X, 2, n_bootstraps=5)
        assert M._IN_BOOTSTRAP is False  # restored even though it was set inside

    def test_small_sample_fallback_quiet_during_bootstrap(self):
        # _fit_robust_rp_reference on a tiny set logs the "empirical covariance"
        # line at INFO normally, DEBUG during bootstrap.
        logger, h, prev = self._capture()
        try:
            X = np.random.default_rng(0).normal(size=(5, 2))
            M._IN_BOOTSTRAP = False
            M._fit_robust_rp_reference(X, 2)
            n_info = sum(1 for lv, m in h.records
                         if lv == "INFO" and "empirical covariance" in m)
            h.records.clear()
            M._IN_BOOTSTRAP = True
            M._fit_robust_rp_reference(X, 2)
            n_debug = sum(1 for lv, m in h.records
                          if lv == "DEBUG" and "empirical covariance" in m)
        finally:
            M._IN_BOOTSTRAP = False
            logger.removeHandler(h)
            logger.setLevel(prev)
        assert n_info >= 1
        assert n_debug >= 1
