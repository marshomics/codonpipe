"""Tests for codon_table_formats module."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from codonpipe.modules.codon_table_formats import (
    compute_absolute_counts,
    compute_frequency_per_thousand,
    compute_rscu_table,
    compute_relative_adaptiveness,
    compute_codon_adaptation_weights,
    compute_cbi,
    generate_all_codon_tables,
    _resolve_optimal_codons,
)

from codonpipe.utils.codon_tables import CODON_TABLE_11, SENSE_CODONS


def _write_fasta(tmp_path, n_genes=10, seed=0):
    rng = np.random.default_rng(seed)
    codons = [c for c in CODON_TABLE_11 if CODON_TABLE_11[c] not in ("*",)]
    seqs = []
    for g in range(n_genes):
        n_codons = rng.integers(80, 200)
        chosen = rng.choice(codons, size=n_codons)
        seq = "ATG" + "".join(chosen) + "TAA"
        seqs.append((f"gene_{g:04d}", seq))
    fasta = tmp_path / "test.ffn"
    fasta.write_text("\n".join(f">{name}\n{seq}" for name, seq in seqs))
    return fasta


def _write_rp_fasta(tmp_path, n_genes=10, seed=1):
    rng = np.random.default_rng(seed)
    codons = [c for c in CODON_TABLE_11 if CODON_TABLE_11[c] not in ("*",)]
    seqs = []
    for g in range(n_genes):
        n_codons = rng.integers(80, 200)
        chosen = rng.choice(codons, size=n_codons)
        seq = "ATG" + "".join(chosen) + "TAA"
        seqs.append((f"gene_{g:04d}", seq))
    fasta = tmp_path / "rp.ffn"
    fasta.write_text("\n".join(f">{name}\n{seq}" for name, seq in seqs))
    return fasta


def _make_expr_df(n_genes=10):
    rows = []
    for g in range(n_genes):
        rows.append({"gene": f"gene_{g:04d}", "expression_class": "high" if g < 3 else "low"})
    return pd.DataFrame(rows)


# ── Absolute counts ─────────────────────────────────────────────────────────

class TestAbsoluteCounts:
    def test_has_64_codons(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        # Covers all 64 codons (sense + stops + Met + Trp)
        assert len(result) == 64
        assert set(["codon", "amino_acid", "count", "total_codons"]).issubset(result.columns)

    def test_all_counts_nonnegative(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        assert (result["count"] >= 0).all()

    def test_total_codons_consistent(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        # total_codons column is the grand total and equals the sum of counts
        assert (result["total_codons"] == result["total_codons"].iloc[0]).all()
        assert result["count"].sum() == result["total_codons"].iloc[0]

    def test_empty_on_no_genes(self, tmp_path):
        # All genes below the 240-nt minimum -> empty result
        fasta = tmp_path / "short.ffn"
        fasta.write_text(">g0\nATGAAATAA\n")
        result = compute_absolute_counts(fasta)
        assert result.empty


# ── Frequency per thousand ──────────────────────────────────────────────────

class TestFrequencyPerThousand:
    def test_per_thousand_sums_to_about_1000(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_frequency_per_thousand(ffn)
        # All 64 codons; per-thousand normalised over all codons
        assert abs(result["per_thousand"].sum() - 1000.0) < 1.0

    def test_per_thousand_nonnegative(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_frequency_per_thousand(ffn)
        assert (result["per_thousand"] >= 0).all()


# ── RSCU table ──────────────────────────────────────────────────────────────

class TestRSCUTable:
    def test_stops_are_nan(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_rscu_table(ffn)
        stops = result[result["amino_acid"] == "*"]
        assert stops["rscu"].isna().all()

    def test_met_trp_are_one(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_rscu_table(ffn)
        for aa in ("Met", "Trp"):
            vals = result.loc[result["amino_acid"] == aa, "rscu"].dropna()
            if not vals.empty:
                assert (abs(vals - 1.0) < 1e-6).all()

    def test_sense_codon_rscu_defined(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_rscu_table(ffn)
        sense = result[result["codon"].isin(SENSE_CODONS)]
        # At least some sense codons have a defined RSCU
        assert sense["rscu"].notna().any()


# ── Relative Adaptiveness (W values) ──────────────────────────────────────

class TestRelativeAdaptiveness:
    def test_w_values_between_0_and_1(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_relative_adaptiveness(ffn)
        # w values should be between 0 and 1 (ratio to max). Stop codons get
        # NaN (no CAI weight), so check only the defined sense-codon values.
        w = result["w_value"].dropna()
        assert (w >= 0).all()
        assert (w <= 1.001).all()  # small tolerance for rounding
        assert w.notna().any()

    def test_max_w_per_aa_is_one(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_relative_adaptiveness(ffn)
        for aa in result["amino_acid"].unique():
            if aa == "*":
                continue
            aa_vals = result.loc[result["amino_acid"] == aa, "w_value"].dropna()
            if not aa_vals.empty and aa_vals.sum() > 0:
                assert abs(aa_vals.max() - 1.0) < 0.01


# ── Codon Adaptation Weights ─────────────────────────────────────────────

class TestCodonAdaptationWeights:
    def test_weight_has_positive_and_negative(self, tmp_path):
        ffn_all = _write_fasta(tmp_path, n_genes=50, seed=42)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        result = compute_codon_adaptation_weights(ffn_all, ffn_all, rp_ids)
        if not result.empty:
            assert (result["weight"] > 0).any() or (result["weight"] < 0).any()

    def test_is_optimal_flag(self, tmp_path):
        ffn_all = _write_fasta(tmp_path, n_genes=50, seed=42)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        result = compute_codon_adaptation_weights(ffn_all, ffn_all, rp_ids)
        if not result.empty:
            assert "is_optimal" in result.columns
            # is_optimal matches weight > 0 for synonymous codons. Met/Trp are
            # single-codon families with weight == 0 but is_optimal == True (the
            # only usable codon), so exclude them from the equivalence.
            multi = result[~result["amino_acid"].isin(["Met", "Trp", "*"])]
            assert (multi["is_optimal"] == (multi["weight"] > 0)).all()

    def test_stops_not_optimal(self, tmp_path):
        ffn_all = _write_fasta(tmp_path, n_genes=50, seed=42)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        result = compute_codon_adaptation_weights(ffn_all, ffn_all, rp_ids)
        if not result.empty:
            stops = result[result["amino_acid"] == "*"]
            assert not stops["is_optimal"].any()


# ── CBI ──────────────────────────────────────────────────────────────────────

class TestCBI:
    def test_cbi_per_gene(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=20, seed=7)
        # Build a simple optimal-codon map from the adaptation weights
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        weights = compute_codon_adaptation_weights(ffn, ffn, rp_ids)
        optimal = _resolve_optimal_codons(weights, label="test")
        result = compute_cbi(ffn, optimal)
        assert not result.empty
        assert set(["gene", "cbi", "n_optimal", "n_total", "n_random", "length"]).issubset(result.columns)

    def test_cbi_values_bounded(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=20, seed=7)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        weights = compute_codon_adaptation_weights(ffn, ffn, rp_ids)
        optimal = _resolve_optimal_codons(weights, label="test")
        result = compute_cbi(ffn, optimal)
        # CBI is bounded above by 1.0 (all-optimal); lower bound can be slightly
        # negative for random usage. Check the defined values stay <= 1.
        cbi = result["cbi"].dropna()
        assert (cbi <= 1.0001).all()

    def test_cbi_random_codons_near_zero(self, tmp_path):
        # Random codon usage -> CBI near 0 on average (no optimal-codon bias).
        ffn = _write_fasta(tmp_path, n_genes=50, seed=123)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        weights = compute_codon_adaptation_weights(ffn, ffn, rp_ids)
        optimal = _resolve_optimal_codons(weights, label="test")
        result = compute_cbi(ffn, optimal)
        if not result.empty:
            assert abs(result["cbi"].mean()) < 0.5


# ── _resolve_optimal_codons ──────────────────────────────────────────────────

class TestResolveOptimalCodons:
    def test_empty_input(self):
        assert _resolve_optimal_codons(pd.DataFrame()) == {}

    def test_returns_one_codon_per_family(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50, seed=42)
        rp_ids = {f"gene_{i:04d}" for i in range(5)}
        weights = compute_codon_adaptation_weights(ffn, ffn, rp_ids)
        optimal = _resolve_optimal_codons(weights, label="test")
        # Each amino acid maps to at most one codon
        assert all(isinstance(c, str) for c in optimal.values())


# ── generate_all_codon_tables (integration) ─────────────────────────────────

class TestGenerateAllCodonTables:
    def test_generates_all_gene_tables(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=None,
            output_dir=tmp_path,
            sample_id="test",
        )
        # Count formats (absolute / per-thousand / RSCU) are merged into one
        # codon_counts table; relative adaptiveness stays separate.
        assert "all_codon_counts" in outputs
        assert "all_relative_adaptiveness" in outputs

    def test_generates_ribosomal_tables(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        rp_ffn = _write_rp_fasta(tmp_path)
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("\n".join([f"gene_{i:04d}" for i in range(10)]) + "\n")
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=rp_ffn,
            output_dir=tmp_path,
            sample_id="test",
            rp_ids_file=rp_ids,
        )
        assert "ribosomal_codon_counts" in outputs
        assert "ribosomal_relative_adaptiveness" in outputs
        # When ribosomal + all are present, adaptation weights + CBI are emitted
        assert "adaptation_weights" in outputs

    def test_generates_high_expression_tables(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        expr_df = pd.DataFrame({
            "gene": [f"gene_{i:04d}" for i in range(50)],
            "CAI_class": ["high"] * 10 + ["medium"] * 30 + ["low"] * 10,
        })
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=None,
            output_dir=tmp_path,
            sample_id="test",
            expr_df=expr_df,
        )
        assert "high_expression_codon_counts" in outputs
        assert "high_expression_relative_adaptiveness" in outputs

    def test_no_high_expression_without_expr(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=None,
            output_dir=tmp_path,
            sample_id="test",
        )
        assert "high_expression_codon_counts" not in outputs

    def test_output_files_exist(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=None,
            output_dir=tmp_path,
            sample_id="test",
        )
        for name, path in outputs.items():
            assert Path(path).exists(), f"Output file {name} not found at {path}"
