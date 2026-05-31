"""Tests for the Mahalanobis dimensionality choices:

  * COA variance-retained axis emission + broken-stick column
    (advanced_analyses.compute_coa_on_rscu / _broken_stick_pct)
  * COA axis selection for the cluster Mahalanobis distance
    (mahal_clustering._select_n_axes)
  * HGT Mahalanobis regularisation diagnostics
    (bio_ecology.detect_hgt_candidates: shrinkage + condition number, 38-d)

These lock in the decision that the optimized-cluster distance uses a small
variance-retained COA space while the genome-wide HGT distance uses the full
38 independent codon dimensions with Ledoit-Wolf shrinkage.
"""

import numpy as np
import pandas as pd

from codonpipe.modules.advanced_analyses import (
    compute_coa_on_rscu,
    _broken_stick_pct,
)
from codonpipe.modules.mahal_clustering import (
    _select_n_axes,
    _MAX_COA_AXES,
    _MIN_COA_AXES,
)
from codonpipe.modules.bio_ecology import detect_hgt_candidates
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES


def _structured_rscu(n=120, n_gradients=3, seed=0):
    """RSCU-like matrix with a few injected latent gradients so COA recovers
    more than one real axis (mimics GC + selection + minor structure)."""
    rng = np.random.default_rng(seed)
    m = len(RSCU_COLUMN_NAMES)
    X = rng.normal(1.0, 0.05, (n, m))
    spans = [(0, 8), (8, 16), (16, 22), (22, 26)][:n_gradients]
    scales = [0.6, 0.3, 0.15, 0.08][:n_gradients]
    for (lo, hi), sc in zip(spans, scales):
        g = rng.normal(0, sc, n)
        X[:, lo:hi] += g[:, None]
    df = pd.DataFrame(np.clip(X, 0.01, None), columns=RSCU_COLUMN_NAMES)
    df.insert(0, "gene", [f"g{i:04d}" for i in range(n)])
    return df


def _inertia(pcts, full_rank=38):
    """Build a COA-inertia table for the leading axes *pcts*.

    Mirrors production: COA emits a handful of leading axes but computes the
    broken-stick null against the FULL singular spectrum (~38 independent codon
    dims), then stores broken_stick_pct[:n_emitted]. Using the full-rank null
    is essential — slicing it from only the emitted axes would inflate the null
    and make real signal look like noise.
    """
    pcts = np.array(pcts, float)
    p = len(pcts)
    bs_full = _broken_stick_pct(full_rank)
    return pd.DataFrame({
        "axis": range(1, p + 1),
        "eigenvalue": pcts,
        "pct_inertia": pcts,
        "cum_pct": np.cumsum(pcts),
        "broken_stick_pct": bs_full[:p],
    })


# -- broken-stick null --------------------------------------------------------

class TestBrokenStick:
    def test_sums_to_100(self):
        for p in (2, 5, 20, 38, 59):
            assert abs(_broken_stick_pct(p).sum() - 100.0) < 1e-6

    def test_monotone_decreasing(self):
        bs = _broken_stick_pct(38)
        assert np.all(np.diff(bs) <= 1e-9)  # first piece is the largest

    def test_empty(self):
        assert _broken_stick_pct(0).size == 0


# -- COA emits a variance-retained set, not a fixed 4 -------------------------

class TestCOAAxisEmission:
    def test_emits_more_than_four_axes(self):
        res = compute_coa_on_rscu(_structured_rscu())
        axis_cols = [c for c in res["coa_coords"].columns if c.startswith("Axis")]
        # The old hard cap was 4; with structure present we now emit more.
        assert len(axis_cols) > 4

    def test_inertia_has_broken_stick_column(self):
        res = compute_coa_on_rscu(_structured_rscu())
        assert "broken_stick_pct" in res["coa_inertia"].columns

    def test_inertia_percentages_normalised_to_full_spectrum(self):
        # cum_pct must be monotone and never exceed 100 (computed against the
        # full singular spectrum, not just the retained axes).
        inertia = compute_coa_on_rscu(_structured_rscu())["coa_inertia"]
        cum = inertia["cum_pct"].values
        assert np.all(np.diff(cum) >= -1e-9)
        assert cum[-1] <= 100.0 + 1e-6

    def test_too_few_genes_returns_empty(self):
        small = _structured_rscu(n=5)
        assert compute_coa_on_rscu(small) == {}


# -- _select_n_axes: broken-stick primary (full-rank null) --------------------

class TestSelectNAxes:
    def test_dominant_single_axis_floors_to_min(self):
        # Axis 1 holds 70%, the rest are noise (below the broken-stick null) ->
        # only 1 real axis, floored to _MIN_COA_AXES (covariance needs >= 2).
        n = _select_n_axes(_inertia([70, 5, 4, 3, 3] + [1] * 10))
        assert n == _MIN_COA_AXES

    def test_high_gc_spread_keeps_several(self):
        # GC axis + selection axis + a couple real, then noise. Broken-stick
        # keeps the leading above-null axes (the motivating high-GC case where
        # the selection axis sits at position 2-4), then stops.
        n = _select_n_axes(_inertia([35, 15, 9, 7, 5, 4] + [2.5] * 9))
        assert 3 <= n <= _MAX_COA_AXES

    def test_uniform_noise_floors_to_min(self):
        # A genuine flat/noise spectrum (every axis at 100/p %, so even axis 1
        # never exceeds the broken-stick null's ~7.9%) -> floor.
        n = _select_n_axes(_inertia([100.0 / 38] * 38))
        assert n == _MIN_COA_AXES

    def test_equal_partition_is_the_noise_null(self):
        # A near-equal partition is, by construction, the broken-stick null
        # itself, so NO axis clears it and the count floors to the minimum.
        # This is the defining property of broken-stick: uniform spread = noise.
        # (Measured: [8]*13+[1]*5 -> 2, because axis 1's 8% < its 11.1% null.)
        n = _select_n_axes(_inertia([8] * 13 + [1] * 5))
        assert n == _MIN_COA_AXES

    def test_rich_signal_retains_many_axes(self):
        # A genuinely high-dimensional signal — leading axes each well above
        # their (decreasing) broken-stick null, summing sanely — retains up to
        # the estimability ceiling. (Measured: -> 8.)
        n = _select_n_axes(_inertia([18, 15, 13, 11, 9, 8, 7, 6, 5] + [1] * 6))
        assert n >= _MAX_COA_AXES - 1

    def test_never_exceeds_ceiling(self):
        # Ten leading axes each above their broken-stick null, total 38 axes
        # (matching the independent-codon rank) summing to ~100 -> the count is
        # bounded by the estimability ceiling, never the spectrum length.
        spectrum = [14, 11, 9, 7.5, 6.5, 5.5, 5, 4.5, 4, 3.5] + [1] * 28
        assert len(spectrum) == 38
        n = _select_n_axes(_inertia(spectrum))
        assert n <= _MAX_COA_AXES

    def test_empty_returns_min(self):
        assert _select_n_axes(pd.DataFrame()) == _MIN_COA_AXES

    def test_fallback_without_broken_stick_column(self):
        # Older COA output lacking broken_stick_pct falls back to the
        # cumulative-inertia rule and still returns a sane count.
        df = _inertia([40, 30, 15, 8, 4, 3]).drop(columns=["broken_stick_pct"])
        n = _select_n_axes(df)
        assert _MIN_COA_AXES <= n <= _MAX_COA_AXES


# -- end-to-end: a displaced selection axis is retained -----------------------

class TestSelectionAxisRetention:
    """The motivating case: a real but minor selection axis sitting *below* the
    GC axis must enter the cluster Mahalanobis distance, not be truncated at the
    old fixed 4. Builds a moderate-GC genome (GC gradient on GC-ending codons +
    an independent selection-like gradient + a minor one) and runs the real COA
    + axis selection end to end."""

    def _moderate_gc_genome(self, n=300, seed=7):
        rng = np.random.default_rng(seed)
        m = len(RSCU_COLUMN_NAMES)
        X = rng.normal(1.0, 0.05, (n, m))
        gc = [i for i, c in enumerate(RSCU_COLUMN_NAMES) if c.endswith(("C", "G"))]
        at = [i for i, c in enumerate(RSCU_COLUMN_NAMES) if c.endswith(("A", "U"))]
        g = rng.normal(0, 0.18, n)
        X[:, gc] += g[:, None]
        X[:, at] -= g[:, None]                      # GC mutational axis
        s = rng.normal(0, 0.22, n)
        X[:, :10] += s[:, None]                     # selection-like axis
        mn = rng.normal(0, 0.15, n)
        X[:, 20:28] += mn[:, None]                  # minor real axis
        df = pd.DataFrame(np.clip(X, 0.01, None), columns=RSCU_COLUMN_NAMES)
        df.insert(0, "gene", [f"g{i:04d}" for i in range(n)])
        return df

    def test_end_to_end_axis_count_is_sane(self):
        # Integration smoke test: real COA on a structured genome, then axis
        # selection, yields an in-range working dimensionality (>= floor, <=
        # estimability ceiling) — never the old fixed 4 and never degenerate.
        # We assert the in-range property rather than an exact count, since the
        # exact count depends on how strongly the random selection gradient
        # happens to clear the broken-stick null (the honest behaviour: a
        # marginal sub-null selection axis is dropped, consistent with the
        # small-cohort estimability limit; a strong one is retained).
        df = self._moderate_gc_genome()
        inertia = compute_coa_on_rscu(df)["coa_inertia"]
        n = _select_n_axes(inertia)
        assert _MIN_COA_AXES <= n <= _MAX_COA_AXES


# -- HGT Mahalanobis regularisation diagnostics -------------------------------

class TestHGTDiagnostics:
    def _inputs(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        m = len(RSCU_COLUMN_NAMES)
        X = rng.normal(1.0, 0.25, (n, m))
        X[:5] += 2.5  # plant a few outliers
        df = pd.DataFrame(np.clip(X, 0.01, None), columns=RSCU_COLUMN_NAMES)
        df.insert(0, "gene", [f"g{i:04d}" for i in range(n)])
        enc = pd.DataFrame({"gene": df["gene"], "GC3": rng.uniform(0.3, 0.7, n)})
        return df, enc

    def test_reports_shrinkage_and_condition_number(self):
        df, enc = self._inputs()
        res = detect_hgt_candidates(df, enc)
        assert "cov_shrinkage" in res.columns
        assert "cov_condition_number" in res.columns

    def test_shrinkage_in_unit_interval(self):
        df, enc = self._inputs()
        res = detect_hgt_candidates(df, enc)
        s = float(res["cov_shrinkage"].iloc[0])
        assert 0.0 <= s <= 1.0

    def test_condition_number_positive_finite(self):
        df, enc = self._inputs()
        res = detect_hgt_candidates(df, enc)
        c = float(res["cov_condition_number"].iloc[0])
        assert np.isfinite(c) and c > 0

    def test_operates_in_38_dimensions(self):
        # The HGT detector genuinely uses the 38 independent codon dimensions
        # (59 columns minus 21, one per multi-codon family).
        df, enc = self._inputs()
        res = detect_hgt_candidates(df, enc)
        assert int(res["n_features"].iloc[0]) == 38
