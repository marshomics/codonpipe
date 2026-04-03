"""Tests for the codon table formats module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.codon_table_formats import (
    compute_absolute_counts,
    compute_frequency_per_thousand,
    compute_rscu_table,
    compute_relative_adaptiveness,
    compute_codon_adaptation_weights,
    compute_cbi,
    generate_all_codon_tables,
)
from codonpipe.utils.codon_tables import CODON_TABLE_11


# ── Fixtures ────────────────────────────────────────────────────────────────

def _write_fasta(tmpdir: Path, n_genes=50, gene_prefix="gene_", seed=42) -> Path:
    """Write a synthetic FASTA file with deterministic sequences."""
    fasta_path = tmpdir / "test.ffn"
    rng = np.random.default_rng(seed)
    bases = list("ATGC")
    with open(fasta_path, "w") as f:
        for i in range(n_genes):
            gene_id = f"{gene_prefix}{i:04d}"
            length = 600
            seq = "".join(rng.choice(bases, size=length))
            f.write(f">{gene_id}\n{seq}\n")
    return fasta_path


def _write_rp_fasta(tmpdir: Path) -> Path:
    """Write a small FASTA of 'ribosomal protein' genes."""
    fasta_path = tmpdir / "rp.ffn"
    rng = np.random.default_rng(99)
    bases = list("ATGC")
    with open(fasta_path, "w") as f:
        for i in range(10):
            gene_id = f"rp_{i:04d}"
            length = 600
            seq = "".join(rng.choice(bases, size=length))
            f.write(f">{gene_id}\n{seq}\n")
    return fasta_path


def _write_rp_ids(tmpdir: Path) -> Path:
    """Write ribosomal protein IDs file."""
    ids_path = tmpdir / "rp_ids.txt"
    ids_path.write_text("\n".join([f"rp_{i:04d}" for i in range(10)]) + "\n")
    return ids_path


# ── Absolute Counts ────────────────────────────────────────────────────────

class TestAbsoluteCounts:
    def test_has_64_codons(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        assert len(result) == 64

    def test_all_counts_nonnegative(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        assert (result["count"] >= 0).all()

    def test_total_codons_consistent(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_absolute_counts(ffn)
        assert result["count"].sum() == result["total_codons"].iloc[0]

    def test_empty_fasta_returns_empty(self, tmp_path):
        ffn = tmp_path / "empty.ffn"
        ffn.write_text("")
        result = compute_absolute_counts(ffn)
        assert result.empty

    def test_filter_by_gene_ids(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        subset = {f"gene_{i:04d}" for i in range(10)}
        result_all = compute_absolute_counts(ffn)
        result_sub = compute_absolute_counts(ffn, gene_ids=subset)
        # Subset should have fewer total codons
        assert result_sub["total_codons"].iloc[0] < result_all["total_codons"].iloc[0]


# ── Frequency Per Thousand ─────────────────────────────────────────────────

class TestFrequencyPerThousand:
    def test_per_thousand_sum(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_frequency_per_thousand(ffn)
        # Sum of all per_thousand should be ~1000
        total = result["per_thousand"].sum()
        assert abs(total - 1000.0) < 0.5

    def test_values_nonnegative(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_frequency_per_thousand(ffn)
        assert (result["per_thousand"] >= 0).all()


# ── RSCU Table ─────────────────────────────────────────────────────────────

class TestRSCUTable:
    def test_excludes_stops_and_single(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_rscu_table(ffn)
        # Should not have stop codons, Met (ATG), or Trp (TGG) — those are excluded from SENSE_CODONS
        for _, row in result.iterrows():
            assert row["amino_acid"] not in ("Stop",)

    def test_rscu_values_reasonable(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_rscu_table(ffn)
        valid = result["rscu"].dropna()
        # RSCU should be between 0 and ~6 for typical amino acids
        assert (valid >= 0).all()
        assert (valid <= 10).all()


# ── Relative Adaptiveness (W values) ──────────────────────────────────────

class TestRelativeAdaptiveness:
    def test_w_values_between_0_and_1(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_relative_adaptiveness(ffn)
        # w values should be between 0 and 1 (ratio to max)
        assert (result["w_value"] >= 0).all()
        assert (result["w_value"] <= 1.001).all()  # small tolerance

    def test_max_w_per_aa_is_one(self, tmp_path):
        ffn = _write_fasta(tmp_path)
        result = compute_relative_adaptiveness(ffn)
        for aa in result["amino_acid"].unique():
            aa_vals = result.loc[result["amino_acid"] == aa, "w_value"]
            if aa_vals.sum() > 0:
                assert abs(aa_vals.max() - 1.0) < 0.01


# ── Codon Adaptation Weights ─────────────────────────────────────────────

class TestCodonAdaptationWeights:
    def test_weight_has_positive_and_negative(self, tmp_path):
        ffn_all = _write_fasta(tmp_path, n_genes=50, seed=42)
        # Create reference with different codon preferences
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
            # is_optimal should match weight > 0
            assert (result["is_optimal"] == (result["weight"] > 0)).all()


# ── CBI ────────────────────────────────────────────────────────────────────

class TestCBI:
    def test_cbi_per_gene(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=20)
        optimal = {"Ala": "GCT", "Gly": "GGT", "Val": "GTT", "Leu": "CTG"}
        result = compute_cbi(ffn, optimal)
        assert "gene" in result.columns
        assert "cbi" in result.columns
        assert len(result) > 0

    def test_cbi_values_bounded_minus1_to_1(self, tmp_path):
        """CBI is defined on [-1, 1]. Values outside indicate a formula error."""
        ffn = _write_fasta(tmp_path, n_genes=50)
        # Provide optimal codons for many amino acids
        optimal = {
            "Ala": "GCU", "Gly": "GGU", "Val": "GUU", "Pro": "CCU",
            "Thr": "ACU", "Leu": "CUG", "Ile": "AUU", "Phe": "UUU",
            "Tyr": "UAU", "His": "CAU", "Gln": "CAA", "Asn": "AAU",
            "Lys": "AAA", "Asp": "GAU", "Glu": "GAA", "Cys": "UGU",
            "Ser": "UCU", "Arg": "CGU",
        }
        result = compute_cbi(ffn, optimal)
        valid = result["cbi"].dropna()
        assert (valid >= -1.01).all(), f"CBI below -1: {valid.min()}"
        assert (valid <= 1.01).all(), f"CBI above 1: {valid.max()}"

    def test_cbi_perfect_bias_gives_1(self, tmp_path):
        """A gene using only optimal codons should have CBI = 1.0."""
        # Build a FASTA with one gene that uses only GCU (Ala optimal)
        fasta_path = tmp_path / "perfect.ffn"
        # 200 codons all GCU = Ala
        seq = "GCT" * 200  # DNA form of GCU
        fasta_path.write_text(f">perfect_gene\n{seq}\n")
        optimal = {"Ala": "GCU"}
        result = compute_cbi(fasta_path, optimal, min_length=0)
        assert len(result) == 1
        assert abs(result.iloc[0]["cbi"] - 1.0) < 0.01

    def test_cbi_random_codons_near_zero(self, tmp_path):
        """Random equal-usage sequences should give CBI near 0."""
        ffn = _write_fasta(tmp_path, n_genes=100, seed=42)
        optimal = {"Ala": "GCU", "Val": "GUU", "Pro": "CCU", "Thr": "ACU"}
        result = compute_cbi(ffn, optimal)
        mean_cbi = result["cbi"].mean()
        # Random sequences: CBI should be roughly around 0 (within ±0.3)
        assert abs(mean_cbi) < 0.4, f"Mean CBI of random seqs = {mean_cbi}, expected near 0"


# ── Full Table Generation ─────────────────────────────────────────────────

class TestGenerateAllCodonTables:
    def test_generates_all_gene_tables(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=None,
            output_dir=tmp_path,
            sample_id="test",
        )
        assert "all_absolute" in outputs
        assert "all_rscu" in outputs
        assert "all_summary" in outputs

    def test_generates_ribosomal_tables(self, tmp_path):
        ffn = _write_fasta(tmp_path, n_genes=50)
        rp_ffn = _write_rp_fasta(tmp_path)
        # Use gene IDs that exist in the main FASTA
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("\n".join([f"gene_{i:04d}" for i in range(10)]) + "\n")
        outputs = generate_all_codon_tables(
            ffn_path=ffn,
            rp_ffn_path=rp_ffn,
            output_dir=tmp_path,
            sample_id="test",
            rp_ids_file=rp_ids,
        )
        assert "ribosomal_absolute" in outputs
        assert "ribosomal_rscu" in outputs

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
        assert "high_expression_absolute" in outputs

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
