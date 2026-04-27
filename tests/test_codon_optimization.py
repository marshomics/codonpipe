"""Tests for the three-way codon-optimization analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.codon_optimization import (
    _w_values_per_family,
    build_recommendation_table,
    build_three_way_codon_table,
    compute_optimization_summary,
    compute_per_gene_gain,
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
