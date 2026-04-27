"""Tests for the three-way codon-optimization analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.codon_optimization import (
    _find_ffn,
    _w_values_per_family,
    build_recommendation_table,
    build_recommendation_table_vs_genome,
    build_three_way_codon_table,
    compute_optimization_summary,
    compute_optimization_summary_vs_genome,
    compute_per_gene_gain,
    compute_per_gene_three_way_cai,
    compute_top_line_stats,
)
from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS_RSCU,
    RSCU_COLUMN_NAMES,
)


def _uniform_rscu_dict(scale: float = 1.0) -> dict[str, float]:
    return {c: scale for c in RSCU_COLUMN_NAMES}


def _biased_rscu_dict(boost_codon_endings: tuple[str, ...]) -> dict[str, float]:
    """Build an RSCU dict where listed codon-suffixes get RSCU=2 and others=0.5."""
    out = {}
    for c in RSCU_COLUMN_NAMES:
        codon_part = c.split("-")[-1]
        out[c] = 2.0 if codon_part in boost_codon_endings else 0.5
    return out


# ── Helper unit tests ───────────────────────────────────────────────────────


class TestWValuesPerFamily:
    def test_all_uniform_yields_uniform_w(self):
        w = _w_values_per_family(_uniform_rscu_dict())
        # Every codon's w should equal 1.0 when the family is uniform
        assert all(np.isclose(v, 1.0) for v in w.values())

    def test_max_is_one(self):
        w = _w_values_per_family(_biased_rscu_dict(("UUU",)))
        # In Phe family, UUU should be the max with w=1.0
        assert np.isclose(w["Phe-UUU"], 1.0)
        # And the non-boosted codon should have a smaller w
        assert w["Phe-UUC"] < w["Phe-UUU"]


# ── Three-way table ──────────────────────────────────────────────────────────


class TestBuildThreeWayTable:
    def test_table_structure(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        for col in ("amino_acid", "family", "codon", "codon_col",
                    "genome_rscu", "rp_rscu", "mahal_rscu",
                    "rp_w", "mahal_w", "rp_optimal", "mahal_optimal",
                    "family_agree", "delta_w_mahal_minus_rp"):
            assert col in table.columns
        # 38 independent codons (Ser/Leu/Arg split families)
        # Including Met/Trp would push to 41; the project convention treats
        # only synonymous codons in AA_CODON_GROUPS_RSCU.
        n_codons = sum(len(v) for v in AA_CODON_GROUPS_RSCU.values())
        assert len(table) == n_codons

    def test_disagreement_detected(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))   # RP prefers UUU for Phe
        m = _biased_rscu_dict(("UUC",))    # Mahal prefers UUC for Phe
        table = build_three_way_codon_table(g, rp, m)
        phe = table[table["family"] == "Phe"]
        # RP-optimal is UUU, Mahal-optimal is UUC → family_agree should be False
        assert not phe.iloc[0]["family_agree"]
        rp_opt = phe[phe["rp_optimal"]]["codon"].iloc[0]
        m_opt = phe[phe["mahal_optimal"]]["codon"].iloc[0]
        assert rp_opt != m_opt
        # Δw should be non-zero
        assert phe["delta_w_mahal_minus_rp"].abs().max() > 0.5

    def test_full_agreement(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUU",))
        table = build_three_way_codon_table(g, rp, m)
        # Every Phe row should report family_agree = True
        phe_rows = table[table["family"] == "Phe"]
        assert phe_rows["family_agree"].all()


# ── Summary + recommendation ─────────────────────────────────────────────────


class TestSummaryAndRecommendation:
    def test_summary_one_row_per_family(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        summary = compute_optimization_summary(table)
        assert len(summary) == len(AA_CODON_GROUPS_RSCU)
        for col in ("family", "rp_optimal_codon", "mahal_optimal_codon",
                    "agree", "max_codon_w_shift"):
            assert col in summary.columns

    def test_top_line_stats(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        summary = compute_optimization_summary(table)
        stats = compute_top_line_stats(summary, table)
        assert "n_families" in stats
        assert "n_disagreeing" in stats
        # Phe disagrees → at least 1 disagreeing family
        assert stats["n_disagreeing"] >= 1

    def test_recommendation_uses_mahal(self):
        g = _uniform_rscu_dict()
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        summary = compute_optimization_summary(table)
        rec = build_recommendation_table(summary, table)
        # Recommendation for Phe should be UUC (Mahal pick), with UUU as alternative
        phe_rec = rec[rec["family"] == "Phe"].iloc[0]
        assert phe_rec["recommended_codon"] == "UUC"
        assert phe_rec["alternative_codon"] == "UUU"

    def test_empty_input(self):
        assert compute_optimization_summary(pd.DataFrame()).empty
        assert build_recommendation_table(pd.DataFrame(), pd.DataFrame()).empty


# ── Per-gene gain ────────────────────────────────────────────────────────────


class TestPerGeneGain:
    def test_gain_calculation(self):
        cbi_rp = pd.DataFrame({
            "gene": ["g1", "g2", "g3"],
            "cbi_rp": [0.5, 0.3, -0.1],
        })
        cbi_mahal = pd.DataFrame({
            "gene": ["g1", "g2", "g3"],
            "cbi_mahal": [0.6, 0.2, 0.1],
        })
        gain = compute_per_gene_gain(cbi_rp, cbi_mahal)
        assert "gain_mahal_minus_rp" in gain.columns
        assert np.isclose(
            gain.set_index("gene").loc["g1", "gain_mahal_minus_rp"], 0.1,
        )
        assert np.isclose(
            gain.set_index("gene").loc["g3", "gain_mahal_minus_rp"], 0.2,
        )

    def test_handles_missing_inputs(self):
        assert compute_per_gene_gain(None, None).empty
        assert compute_per_gene_gain(pd.DataFrame(), pd.DataFrame()).empty


# ── Mahal-vs-genome variants ────────────────────────────────────────────────


class TestSummaryVsGenome:
    def test_detects_disagreement(self):
        # Genome biased toward UUU, Mahal biased toward UUC
        g = _biased_rscu_dict(("UUU",))
        rp = _biased_rscu_dict(("UUC",))   # any
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        summary_g = compute_optimization_summary_vs_genome(table)
        phe = summary_g[summary_g["family"] == "Phe"].iloc[0]
        assert not phe["agree"]
        assert phe["genome_optimal_codon"] == "UUU"
        assert phe["mahal_optimal_codon"] == "UUC"
        assert phe["max_codon_w_shift_vs_genome"] > 0.5

    def test_recommendation_uses_mahal(self):
        g = _biased_rscu_dict(("UUU",))
        rp = _biased_rscu_dict(("UUU",))
        m = _biased_rscu_dict(("UUC",))
        table = build_three_way_codon_table(g, rp, m)
        summary_g = compute_optimization_summary_vs_genome(table)
        rec_g = build_recommendation_table_vs_genome(summary_g, table)
        phe = rec_g[rec_g["family"] == "Phe"].iloc[0]
        assert phe["recommended_codon"] == "UUC"
        assert phe["alternative_codon"] == "UUU"
        assert "differs from genome-most-frequent" in phe["rationale"]


class TestPerGeneThreeWayCAI:
    def test_synthetic_genome(self, tmp_path):
        """Build a tiny .ffn and verify three-way CAI runs end-to-end."""
        ffn = tmp_path / "test.ffn"
        # Two short genes, ~250 nt each (above MIN_GENE_LENGTH of 240)
        # Simple alternating codons covering each AA family
        seq1 = ("ATG" + "GCT" * 30 + "GCC" * 30 + "GCA" * 30 + "TAA")  # mostly Ala variants
        seq2 = ("ATG" + "GCG" * 90 + "TAA")
        with open(ffn, "w") as fh:
            fh.write(f">gene1\n{seq1}\n>gene2\n{seq2}\n")

        # All-uniform reference frames → every CAI should equal 1.0
        g = _uniform_rscu_dict()
        rp = _uniform_rscu_dict()
        m = _uniform_rscu_dict()
        cai = compute_per_gene_three_way_cai(ffn, g, rp, m, min_length=240)
        assert not cai.empty
        for col in ("cai_genome", "cai_rp", "cai_mahal",
                    "gain_mahal_vs_genome", "gain_mahal_vs_rp", "gain_rp_vs_genome"):
            assert col in cai.columns
        # Uniform w means every codon has w=1.0, so CAI = 1.0 for every gene
        assert np.allclose(cai["cai_genome"].dropna(), 1.0, atol=1e-6)
        assert np.allclose(cai["cai_mahal"].dropna(), 1.0, atol=1e-6)
        # Gains should be ~0
        assert (cai["gain_mahal_vs_genome"].abs() < 1e-6).all()

    def test_ffn_finder_role_based_layout(self, tmp_path):
        """Production layout: <sample>/annotation/prokka/<sid>.ffn."""
        sample_dir = tmp_path / "sample"
        prokka_dir = sample_dir / "annotation" / "prokka"
        prokka_dir.mkdir(parents=True)
        target = prokka_dir / "S.ffn"
        target.write_text(">g\nATG" + "GCC" * 10 + "\n")
        found = _find_ffn(sample_dir, "S")
        assert found is not None
        assert found.resolve() == target.resolve()

    def test_ffn_finder_legacy_flat_layout(self, tmp_path):
        sample_dir = tmp_path / "sample"
        prokka_dir = sample_dir / "prokka"
        prokka_dir.mkdir(parents=True)
        target = prokka_dir / "S.ffn"
        target.write_text(">g\nATG\n")
        found = _find_ffn(sample_dir, "S")
        assert found is not None
        assert found.resolve() == target.resolve()

    def test_ffn_finder_glob_fallback(self, tmp_path):
        """Externally-supplied .ffn dropped under prokka/ without a {sid} prefix."""
        sample_dir = tmp_path / "sample"
        prokka_dir = sample_dir / "prokka"
        prokka_dir.mkdir(parents=True)
        target = prokka_dir / "external_genes.ffn"
        target.write_text(">g\nATG\n")
        found = _find_ffn(sample_dir, "different_sid")
        assert found is not None
        assert found.resolve() == target.resolve()

    def test_ffn_finder_explicit_override(self, tmp_path):
        external = tmp_path / "external" / "genes.ffn"
        external.parent.mkdir()
        external.write_text(">g\nATG\n")
        # No prokka subdir at all; only the explicit override
        found = _find_ffn(tmp_path / "empty_sample", "S", override=external)
        assert found is not None
        assert found.resolve() == external.resolve()

    def test_ffn_finder_returns_none_when_missing(self, tmp_path):
        assert _find_ffn(tmp_path / "no_sample", "S") is None

    def test_biased_references_give_different_cais(self, tmp_path):
        """When references differ, the per-gene CAIs should diverge."""
        ffn = tmp_path / "test.ffn"
        # A gene that uses GCC (Ala) heavily
        seq = "ATG" + "GCC" * 90 + "TAA"
        with open(ffn, "w") as fh:
            fh.write(f">gene1\n{seq}\n")

        g = _biased_rscu_dict(("GCC",))   # genome favours GCC (matches gene)
        rp = _biased_rscu_dict(("GCG",))   # RP favours GCG (penalises gene)
        m = _biased_rscu_dict(("GCC",))    # Mahal favours GCC (matches gene)

        cai = compute_per_gene_three_way_cai(ffn, g, rp, m, min_length=240)
        assert not cai.empty
        row = cai.iloc[0]
        # Gene matches genome and Mahal; should score much higher there than RP
        assert row["cai_genome"] > row["cai_rp"]
        assert row["cai_mahal"] > row["cai_rp"]
        assert row["gain_mahal_vs_rp"] > 0
        # Gain over genome should be ~0 since both favour GCC
        assert abs(row["gain_mahal_vs_genome"]) < 0.05
