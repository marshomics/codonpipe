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


class TestExpressionClassification:
    """Test that per-metric classification works correctly."""

    def test_classify_by_percentile(self):
        from codonpipe.modules.expression import _classify_by_percentile

        vals = pd.Series(list(range(100)))
        classes = _classify_by_percentile(vals)
        assert (classes == "high").sum() == 10  # >= 90th percentile (90-99)
        assert (classes == "low").sum() == 10  # <= 10th percentile (0-9)
        assert (classes == "medium").sum() == 80

    def test_classify_all_nan(self):
        from codonpipe.modules.expression import _classify_by_percentile

        vals = pd.Series([np.nan, np.nan, np.nan])
        classes = _classify_by_percentile(vals)
        assert (classes == "unknown").all()
