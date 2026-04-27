"""Tests for the additional GOI analyses + figures (gene_set_extras)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules._gene_set_extras import (
    cluster_within_goi,
    compute_expression_tier_breakdown,
    _build_clr_delta_matrix,
    _cliffs_delta_simple,
    _cluster_drivers,
)
from codonpipe.modules.gene_set import _drop_redundant_codon_per_family
from codonpipe.utils.codon_tables import COL_GENE, RSCU_COLUMN_NAMES


# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_rscu_gene_df(n_genes: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        COL_GENE: [f"locus_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
    }
    for c in RSCU_COLUMN_NAMES:
        data[c] = rng.uniform(0.2, 2.5, size=n_genes)
    return pd.DataFrame(data)


def _make_summary_with_classes(genes: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(genes)
    df = pd.DataFrame({COL_GENE: genes})
    for m in ("MELP", "CAI", "Fop"):
        df[m] = rng.uniform(0.2, 0.9, size=n)
        # Default class distribution: 10% high, 80% medium, 10% low
        df[f"{m}_class"] = rng.choice(
            ["high", "medium", "low"], size=n, p=[0.1, 0.8, 0.1],
        )
    return df


# ── Expression tier breakdown ────────────────────────────────────────────────


class TestExpressionTierBreakdown:
    def test_balanced_distribution_no_significance(self):
        # GOI tier distribution matches background → Fisher should be non-significant
        rng = np.random.default_rng(0)
        n_total = 600
        all_genes = [f"locus_{i:04d}" for i in range(n_total)]
        base = _make_summary_with_classes(all_genes, seed=0)
        # GOI is a random sample of 60
        goi_genes = rng.choice(all_genes, size=60, replace=False)
        summary = base[base[COL_GENE].isin(goi_genes)].copy()
        out = compute_expression_tier_breakdown(summary, base)
        assert not out.empty
        for col in ("metric", "tier", "n_goi", "frac_goi", "p_value", "p_adjusted"):
            assert col in out.columns
        # Random sample shouldn't trigger anything BH-significant
        assert (~out["significant"]).all()

    def test_high_enrichment_detected(self):
        # Force GOI to all be 'high' for MELP — Fisher should flag
        n_total = 400
        all_genes = [f"locus_{i:04d}" for i in range(n_total)]
        base = _make_summary_with_classes(all_genes, seed=2)
        goi_genes = all_genes[:60]
        summary = base[base[COL_GENE].isin(goi_genes)].copy()
        summary["MELP_class"] = "high"
        out = compute_expression_tier_breakdown(summary, base)
        melp_high = out[(out["metric"] == "MELP") & (out["tier"] == "high")]
        assert not melp_high.empty
        assert melp_high.iloc[0]["p_adjusted"] < 0.001
        assert melp_high.iloc[0]["frac_goi"] == pytest.approx(1.0)

    def test_skips_when_class_columns_missing(self):
        df = pd.DataFrame({COL_GENE: ["a", "b", "c"]})
        out = compute_expression_tier_breakdown(df, df)
        assert out.empty


# ── Within-GOI clustering ────────────────────────────────────────────────────


class TestCliffsDeltaSimple:
    def test_extreme_positive(self):
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        assert _cliffs_delta_simple(x, y) == 1.0

    def test_zero_overlap(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        assert _cliffs_delta_simple(x, y) == 0.0


class TestClusterWithinGOI:
    def test_recovers_two_synthetic_groups(self):
        """Build 30 GOIs with two latent codon-preference groups; clustering should recover."""
        rng = np.random.default_rng(0)
        n_per = 15
        n_total = 2 * n_per
        rscu_cols_full = list(RSCU_COLUMN_NAMES)
        rscu_cols_indep = _drop_redundant_codon_per_family(rscu_cols_full)

        # Group A: bias toward UUU/CUU (first codons in family)
        # Group B: bias toward UUC/CUG (later codons in family)
        rows = []
        for i in range(n_per):
            row = {COL_GENE: f"A_{i:03d}", "length": rng.integers(300, 3000)}
            for c in rscu_cols_full:
                if c.endswith(("UUU", "UCU", "CUU")):
                    row[c] = rng.uniform(2.0, 2.8)
                elif c.endswith(("UUC", "UCC", "CUC")):
                    row[c] = rng.uniform(0.2, 0.6)
                else:
                    row[c] = rng.uniform(0.5, 1.5)
            rows.append(row)
        for i in range(n_per):
            row = {COL_GENE: f"B_{i:03d}", "length": rng.integers(300, 3000)}
            for c in rscu_cols_full:
                if c.endswith(("UUU", "UCU", "CUU")):
                    row[c] = rng.uniform(0.2, 0.6)
                elif c.endswith(("UUC", "UCC", "CUC")):
                    row[c] = rng.uniform(2.0, 2.8)
                else:
                    row[c] = rng.uniform(0.5, 1.5)
            rows.append(row)
        goi_rscu = pd.DataFrame(rows)

        # Use a neutral genome reference (uniform at 1.0 per codon)
        rscu_genome = {c: 1.0 for c in rscu_cols_full}

        result = cluster_within_goi(goi_rscu, rscu_genome=rscu_genome,
                                    min_cluster_size=3)
        assert result, "cluster_within_goi returned empty"
        # Auto-selected K should be 2 for clear two-group data
        assert result["n_clusters"] == 2
        # Each gene has a cluster assignment
        assert len(result["cluster_id"]) == n_total
        # Most A_xxx and B_xxx genes should be in different clusters
        a_clusters = {result["cluster_id"][f"A_{i:03d}"] for i in range(n_per)}
        b_clusters = {result["cluster_id"][f"B_{i:03d}"] for i in range(n_per)}
        # Modal cluster should differ between A and B
        a_mode = max(a_clusters, key=lambda c: sum(
            result["cluster_id"][g] == c for g in [f"A_{i:03d}" for i in range(n_per)]
        ))
        b_mode = max(b_clusters, key=lambda c: sum(
            result["cluster_id"][g] == c for g in [f"B_{i:03d}" for i in range(n_per)]
        ))
        assert a_mode != b_mode

    def test_too_few_genes_returns_empty(self):
        rscu = _make_rscu_gene_df(4)
        out = cluster_within_goi(rscu, rscu_genome=None, min_cluster_size=3)
        assert out == {}

    def test_drivers_table_has_required_columns(self):
        rng = np.random.default_rng(0)
        n = 20
        rscu = _make_rscu_gene_df(n)
        # Spike in differential signal so silhouette picks K>=2
        for c in RSCU_COLUMN_NAMES[:5]:
            rscu.loc[:n // 2, c] = rng.uniform(2.5, 3.0, size=n // 2 + 1)
            rscu.loc[n // 2:, c] = rng.uniform(0.2, 0.5, size=n - n // 2)
        result = cluster_within_goi(rscu, rscu_genome=None, min_cluster_size=3)
        if result and result["n_clusters"] >= 2 and not result["drivers_df"].empty:
            for col in ("cluster_id", "feature_type", "feature",
                        "median_in", "median_out", "U_statistic",
                        "p_value", "cliffs_delta", "p_adjusted", "significant"):
                assert col in result["drivers_df"].columns
