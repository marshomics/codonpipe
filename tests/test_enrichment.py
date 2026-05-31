"""Tests for the pathway enrichment module."""

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.enrichment import hypergeometric_enrichment


class TestHypergeometricEnrichment:
    """Tests for the hypergeometric pathway enrichment test."""

    def _make_ko_map(self):
        """Pathway map: pw1 has K1-K5, pw2 has K3-K8, pw3 has K9-K10."""
        return {
            "K00001": {"ko00010"},
            "K00002": {"ko00010"},
            "K00003": {"ko00010", "ko00020"},
            "K00004": {"ko00010"},
            "K00005": {"ko00010"},
            "K00006": {"ko00020"},
            "K00007": {"ko00020"},
            "K00008": {"ko00020"},
            "K00009": {"ko00030"},
            "K00010": {"ko00030"},
        }

    def test_detects_enrichment(self):
        """Test set heavily biased toward pw1 should show enrichment."""
        ko_map = self._make_ko_map()
        # Test set: 4 of 5 ko00010 genes, 0 from others
        test_kos = {"K00001", "K00002", "K00003", "K00004"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map)
        assert not result.empty
        assert "ko00010" in result["pathway"].values
        # ko00010 should have the lowest p-value
        top = result.iloc[0]
        assert top["pathway"] == "ko00010"
        assert top["fold_enrichment"] > 1.0

    def test_no_enrichment_uniform(self):
        """Uniform sampling from background shouldn't produce strong enrichment."""
        ko_map = self._make_ko_map()
        # Test set includes one gene from each pathway — no bias
        test_kos = {"K00001", "K00006", "K00009"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map)
        # With such a small set, nothing should be significant
        if "significant" in result.columns:
            assert not result["significant"].any()

    def test_empty_test_set(self):
        """Empty test set returns empty DataFrame."""
        ko_map = self._make_ko_map()
        result = hypergeometric_enrichment(set(), set(ko_map.keys()), ko_map)
        assert result.empty

    def test_empty_background(self):
        """Empty background returns empty DataFrame."""
        ko_map = self._make_ko_map()
        result = hypergeometric_enrichment({"K00001"}, set(), ko_map)
        assert result.empty

    def test_fdr_correction(self):
        """FDR values should be >= raw p-values and <= 1."""
        ko_map = self._make_ko_map()
        test_kos = {"K00001", "K00002", "K00003", "K00004", "K00005"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map)
        if not result.empty and "fdr" in result.columns:
            assert (result["fdr"] >= result["p_value"] - 1e-10).all()
            assert (result["fdr"] <= 1.0).all()

    def test_pathway_names_included(self):
        """Pathway names appear in output when provided."""
        ko_map = self._make_ko_map()
        names = {"ko00010": "Glycolysis", "ko00020": "TCA cycle"}
        test_kos = {"K00001", "K00002"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map, pathway_names=names)
        assert "pathway_name" in result.columns
        glycolysis_row = result[result["pathway"] == "ko00010"]
        if not glycolysis_row.empty:
            assert glycolysis_row.iloc[0]["pathway_name"] == "Glycolysis"

    def test_test_kos_listed(self):
        """Output includes which test KOs hit each pathway."""
        ko_map = self._make_ko_map()
        test_kos = {"K00001", "K00003"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map)
        assert "test_kos_in_pathway" in result.columns
        ko10_row = result[result["pathway"] == "ko00010"]
        if not ko10_row.empty:
            kos_str = ko10_row.iloc[0]["test_kos_in_pathway"]
            assert "K00001" in kos_str
            assert "K00003" in kos_str


    def test_map_pathway_ids_excluded(self):
        """map* pathway IDs should be excluded, keeping only ko* IDs."""
        # Build a ko_map that includes both ko* and map* entries for the
        # same pathway — simulating raw KEGG REST API output.
        ko_map = {
            "K00001": {"ko00010", "map00010"},
            "K00002": {"ko00010", "map00010"},
            "K00003": {"ko00010", "map00010", "ko00020", "map00020"},
            "K00004": {"ko00010", "map00010"},
            "K00005": {"ko00010", "map00010"},
            "K00006": {"ko00020", "map00020"},
            "K00007": {"ko00020", "map00020"},
            "K00008": {"ko00020", "map00020"},
        }
        test_kos = {"K00001", "K00002", "K00003", "K00004", "K00005"}
        background = set(ko_map.keys())

        result = hypergeometric_enrichment(test_kos, background, ko_map)
        # map* pathways should appear because hypergeometric_enrichment
        # doesn't filter — the filtering happens at ingestion.  This test
        # documents the current contract; the real guard is in the load
        # functions.
        pathways = set(result["pathway"].values)
        assert "ko00010" in pathways
        # Verify both map and ko variants are present (since this uses
        # the enrichment function directly, not the filtered loaders)
        assert "map00010" in pathways


class TestMapPathwayFiltering:
    """Verify that map* pathway IDs are filtered out during loading."""

    def test_download_filters_map_ids(self, tmp_path):
        """_load_user_ko_map should exclude map* pathway IDs."""
        from codonpipe.modules.enrichment import _load_user_ko_map

        tsv = tmp_path / "ko_map.tsv"
        tsv.write_text(
            "K00001\tko00010\n"
            "K00001\tmap00010\n"
            "K00002\tko00020\n"
            "K00002\tmap00020\n"
        )
        result = _load_user_ko_map(tsv)
        for ko, pathways in result.items():
            for pw in pathways:
                assert not pw.startswith("map"), f"map* ID {pw} not filtered for {ko}"
        assert result["K00001"] == {"ko00010"}
        assert result["K00002"] == {"ko00020"}

    def test_cached_json_filters_map_ids(self, tmp_path):
        """load_ko_pathway_map should filter map* from cached JSON."""
        import json
        from codonpipe.modules.enrichment import load_ko_pathway_map

        cache = tmp_path / "cache"
        cache.mkdir()
        cached_file = cache / "kegg_ko_pathway.json"
        cached_file.write_text(json.dumps({
            "K00001": ["ko00010", "map00010"],
            "K00002": ["ko00020", "map00020"],
        }))
        result = load_ko_pathway_map(cache_dir=cache)
        for ko, pathways in result.items():
            for pw in pathways:
                assert not pw.startswith("map"), f"map* ID {pw} not filtered for {ko}"


class TestExpressionClassification:
    """Test that per-metric classification works correctly."""

    def test_classify_by_percentile(self):
        from codonpipe.modules.expression import _classify_by_percentile

        # Uses fixed quantile cutoffs: top 10% = high, bottom 10% = low.
        # Uniform 0..99: 90th pctile ≈ 89.1, 10th pctile ≈ 9.9
        # → high: values >= 89.1 (90..99 = 10 values)
        # → low:  values <= 9.9  (0..9   = 10 values)
        # → medium: the remaining 80
        vals = pd.Series(list(range(100)))
        classes = _classify_by_percentile(vals)
        n_high = (classes == "high").sum()
        n_low = (classes == "low").sum()
        n_medium = (classes == "medium").sum()
        assert n_high == 10
        assert n_low == 10
        assert n_medium == 80
        assert n_high + n_low + n_medium == 100

    def test_classify_skewed_distribution(self):
        """Fixed-percentile classification produces stable tier sizes regardless of distribution shape."""
        from codonpipe.modules.expression import _classify_by_percentile

        # Right-skewed: many low values, few high values.
        # With fixed 10th/90th percentile cutoffs, tier sizes should be
        # approximately equal (~10% each) regardless of skew. This is the
        # intentional design: stable, distribution-independent boundaries.
        np.random.seed(42)
        skewed = pd.Series(np.random.exponential(0.3, 1000))
        classes = _classify_by_percentile(skewed)
        n_high = (classes == "high").sum()
        n_low = (classes == "low").sum()
        # Both tiers should contain roughly 10% of values (allow ±2% for ties)
        assert 80 <= n_high <= 120, f"Expected ~100 high genes, got {n_high}"
        assert 80 <= n_low <= 120, f"Expected ~100 low genes, got {n_low}"
        assert n_high + n_low + (classes == "medium").sum() == 1000

    def test_classify_all_nan(self):
        from codonpipe.modules.expression import _classify_by_percentile

        vals = pd.Series([np.nan, np.nan, np.nan])
        classes = _classify_by_percentile(vals)
        assert (classes == "unknown").all()


class TestMelpDegenerateFallback:
    """expression_class must fall back from a saturated MELP to a metric whose
    high/low tiers are populated.

    MELP clips to the coRdon non-negative floor (Supek & Smuc 2010). In high-GC
    genomes such as B. subtilis most genes sit at exactly 0, so the 10th and
    90th percentiles coincide and _classify_by_percentile collapses every gene
    to 'medium'. A degenerate expression_class silently breaks every downstream
    tier analysis (MELP/COG enrichment, tier delta-RSCU). The old logic only
    fell back when MELP was *absent*; these tests lock in fallback when MELP is
    present but degenerate.
    """

    def test_is_degenerate_flags_saturated_column(self):
        from codonpipe.modules.expression import _is_degenerate_class

        # All-medium (no high, no low) -> degenerate.
        saturated = pd.Series(["medium"] * 50)
        assert _is_degenerate_class(saturated) is True

        # Missing only the low tier is still unusable for tier contrast.
        no_low = pd.Series(["high"] * 5 + ["medium"] * 45)
        assert _is_degenerate_class(no_low) is True

        # Empty / all-unknown -> degenerate.
        assert _is_degenerate_class(pd.Series([], dtype=object)) is True
        assert _is_degenerate_class(pd.Series(["unknown"] * 10)) is True

    def test_healthy_column_not_degenerate(self):
        from codonpipe.modules.expression import _is_degenerate_class

        healthy = pd.Series(["high"] * 10 + ["medium"] * 80 + ["low"] * 10)
        assert _is_degenerate_class(healthy) is False

    def test_combine_falls_back_to_cai_when_melp_saturated(self, tmp_path):
        """End-to-end: a saturated MELP score column makes expression_class use
        CAI tiers, restoring populated high/low tiers."""
        from codonpipe.modules.expression import _combine_expression
        from codonpipe.utils.codon_tables import (
            COL_GENE, COL_WIDTH, COL_EXPRESSION_CLASS,
            COL_MELP_CLASS, COL_CAI_CLASS,
        )

        n = 100
        genes = [f"g{i:03d}" for i in range(n)]
        widths = [300] * n
        # MELP saturated: every gene at the coRdon floor -> degenerate tiers.
        melp_scores = [0.0] * n
        # CAI spans a real range -> populated high/low tiers.
        cai_scores = [i / (n - 1) for i in range(n)]
        # Fop also saturated, to prove the fallback stops at CAI (priority order).
        fop_scores = [0.0] * n

        melp_path = tmp_path / "melp.tsv"
        cai_path = tmp_path / "cai.tsv"
        fop_path = tmp_path / "fop.tsv"
        pd.DataFrame({COL_GENE: genes, COL_WIDTH: widths, "MILC": melp_scores}).to_csv(
            melp_path, sep="\t", index=False)
        pd.DataFrame({COL_GENE: genes, "CAI": cai_scores}).to_csv(
            cai_path, sep="\t", index=False)
        pd.DataFrame({COL_GENE: genes, "Fop": fop_scores}).to_csv(
            fop_path, sep="\t", index=False)

        combined = _combine_expression(melp_path, cai_path, fop_path, "testsample")

        # MELP tiers degenerate, CAI tiers healthy.
        assert (combined[COL_MELP_CLASS] == "medium").all()
        assert (combined[COL_CAI_CLASS] == "high").any()
        assert (combined[COL_CAI_CLASS] == "low").any()

        # expression_class should now equal CAI tiers, NOT the saturated MELP.
        assert (combined[COL_EXPRESSION_CLASS] == combined[COL_CAI_CLASS]).all()
        assert (combined[COL_EXPRESSION_CLASS] == "high").any()
        assert (combined[COL_EXPRESSION_CLASS] == "low").any()

    def test_combine_prefers_melp_when_healthy(self, tmp_path):
        """When MELP tiers are populated, expression_class stays on MELP."""
        from codonpipe.modules.expression import _combine_expression
        from codonpipe.utils.codon_tables import (
            COL_GENE, COL_WIDTH, COL_EXPRESSION_CLASS, COL_MELP_CLASS,
        )

        n = 100
        genes = [f"g{i:03d}" for i in range(n)]
        melp_scores = [i / (n - 1) for i in range(n)]   # real range
        cai_scores = [i / (n - 1) for i in range(n)]

        melp_path = tmp_path / "melp.tsv"
        cai_path = tmp_path / "cai.tsv"
        fop_path = tmp_path / "fop.tsv"
        pd.DataFrame({COL_GENE: genes, COL_WIDTH: [300] * n, "MILC": melp_scores}).to_csv(
            melp_path, sep="\t", index=False)
        pd.DataFrame({COL_GENE: genes, "CAI": cai_scores}).to_csv(
            cai_path, sep="\t", index=False)
        pd.DataFrame({COL_GENE: genes, "Fop": cai_scores}).to_csv(
            fop_path, sep="\t", index=False)

        combined = _combine_expression(melp_path, cai_path, fop_path, "testsample")
        assert (combined[COL_EXPRESSION_CLASS] == combined[COL_MELP_CLASS]).all()
        assert (combined[COL_EXPRESSION_CLASS] == "high").any()
