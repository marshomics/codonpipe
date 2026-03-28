"""Tests for the advanced codon usage analyses module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.advanced_analyses import (
    compute_coa_on_rscu,
    compute_delta_rscu,
    compute_enc_diff,
    compute_gc12_gc3,
    compute_gene_length_bias,
    compute_pr2,
    compute_s_value,
    compute_trna_codon_correlation,
    extract_trna_counts_from_gff,
    _reverse_complement,
)
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_rscu_gene_df(n_genes=100) -> pd.DataFrame:
    """Create a synthetic per-gene RSCU DataFrame."""
    rng = np.random.default_rng(42)
    data = {"gene": [f"gene_{i:04d}" for i in range(n_genes)],
            "length": rng.integers(300, 3000, size=n_genes)}
    for col in RSCU_COLUMN_NAMES:
        data[col] = rng.uniform(0.2, 3.0, size=n_genes)
    return pd.DataFrame(data)


def _make_enc_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
        "ENC": rng.uniform(25, 60, size=n_genes),
        "GC3": rng.uniform(0.2, 0.8, size=n_genes),
    })


def _make_expr_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    classes = ["low"] * 5 + ["medium"] * 90 + ["high"] * 5
    return pd.DataFrame({
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "CAI": rng.uniform(0.1, 0.9, size=n_genes),
        "MELP": rng.uniform(0, 1, size=n_genes),
        "Fop": rng.uniform(0.2, 0.8, size=n_genes),
        "CAI_class": classes[:n_genes],
        "MELP_class": classes[:n_genes],
        "Fop_class": classes[:n_genes],
    })


def _make_encprime_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "ENCprime": rng.uniform(25, 60, size=n_genes),
        "width": rng.integers(300, 3000, size=n_genes),
    })


def _write_test_fasta(path: Path, n_seqs=50) -> None:
    """Write a minimal test FASTA with valid CDS sequences."""
    rng = np.random.default_rng(42)
    with open(path, "w") as f:
        for i in range(n_seqs):
            length = rng.choice([300, 600, 900, 1200])
            seq = "".join(rng.choice(list("ACGT"), size=length))
            # Ensure starts with ATG and length divisible by 3
            seq = "ATG" + seq[3:]
            seq = seq[:len(seq) - len(seq) % 3]
            f.write(f">gene_{i:04d}\n{seq}\n")


def _write_test_gff(path: Path) -> None:
    """Write a minimal GFF3 with tRNA features."""
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        f.write("contig_1\tProkka\ttRNA\t100\t175\t.\t+\t.\tID=tRNA1;product=tRNA-Ala(TGC)\n")
        f.write("contig_1\tProkka\ttRNA\t200\t275\t.\t+\t.\tID=tRNA2;product=tRNA-Ala(TGC)\n")
        f.write("contig_1\tProkka\ttRNA\t300\t375\t.\t-\t.\tID=tRNA3;product=tRNA-Gly(GCC)\n")
        f.write("contig_1\tProkka\ttRNA\t400\t475\t.\t+\t.\tID=tRNA4;product=tRNA-Leu(CAA)\n")
        f.write("contig_1\tProkka\tCDS\t500\t800\t.\t+\t.\tID=cds1;product=some protein\n")


# ── Tests ────────────────────────────────────────────────────────────────────

class TestCOA:
    def test_coa_returns_expected_keys(self):
        rscu_df = _make_rscu_gene_df(50)
        result = compute_coa_on_rscu(rscu_df)
        assert "coa_coords" in result
        assert "coa_codon_coords" in result
        assert "coa_inertia" in result

    def test_coa_coords_shape(self):
        rscu_df = _make_rscu_gene_df(50)
        result = compute_coa_on_rscu(rscu_df)
        coords = result["coa_coords"]
        assert "gene" in coords.columns
        assert "Axis1" in coords.columns
        assert "Axis2" in coords.columns
        assert len(coords) == 50

    def test_coa_inertia_sums_approximately(self):
        rscu_df = _make_rscu_gene_df(100)
        result = compute_coa_on_rscu(rscu_df)
        inertia = result["coa_inertia"]
        # Cumulative percentage should increase
        assert inertia["cum_pct"].is_monotonic_increasing

    def test_coa_merges_expression_tiers(self):
        rscu_df = _make_rscu_gene_df(100)
        expr_df = _make_expr_df(100)
        result = compute_coa_on_rscu(rscu_df, expr_df)
        assert "CAI_class" in result["coa_coords"].columns

    def test_coa_too_few_genes(self):
        rscu_df = _make_rscu_gene_df(5)
        result = compute_coa_on_rscu(rscu_df)
        assert result == {}


class TestSValue:
    def test_s_value_computed(self):
        rscu_df = _make_rscu_gene_df(50)
        # Use first gene as pseudo-reference
        rscu_rp = {col: 1.0 for col in RSCU_COLUMN_NAMES}
        result = compute_s_value(rscu_df, rscu_rp)
        assert "S_value" in result.columns
        assert len(result) == 50
        assert (result["S_value"] >= 0).all()

    def test_s_value_no_reference(self):
        rscu_df = _make_rscu_gene_df(50)
        result = compute_s_value(rscu_df, None)
        assert result.empty

    def test_s_value_identical_to_ref_is_zero(self):
        """A gene with identical RSCU to reference should have S=0."""
        ref = {col: 1.5 for col in RSCU_COLUMN_NAMES}
        data = {"gene": ["test_gene"], "length": [600]}
        for col in RSCU_COLUMN_NAMES:
            data[col] = [1.5]
        rscu_df = pd.DataFrame(data)
        result = compute_s_value(rscu_df, ref)
        assert abs(result.iloc[0]["S_value"]) < 1e-10


class TestENCDiff:
    def test_enc_diff_computed(self):
        enc_df = _make_enc_df(50)
        encprime_df = _make_encprime_df(50)
        result = compute_enc_diff(enc_df, encprime_df)
        assert "ENC_diff" in result.columns
        assert len(result) == 50

    def test_enc_diff_equals_enc_minus_encprime(self):
        enc_df = _make_enc_df(20)
        encprime_df = _make_encprime_df(20)
        result = compute_enc_diff(enc_df, encprime_df)
        for _, row in result.iterrows():
            assert abs(row["ENC_diff"] - (row["ENC"] - row["ENCprime"])) < 1e-10


class TestGC12GC3:
    def test_gc12_gc3_from_fasta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ffn = Path(tmpdir) / "test.ffn"
            _write_test_fasta(ffn, 30)
            result = compute_gc12_gc3(ffn)
            assert not result.empty
            assert "GC12" in result.columns
            assert "GC3" in result.columns
            assert (result["GC12"] >= 0).all() and (result["GC12"] <= 1).all()
            assert (result["GC3"] >= 0).all() and (result["GC3"] <= 1).all()


class TestPR2:
    def test_pr2_from_fasta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ffn = Path(tmpdir) / "test.ffn"
            _write_test_fasta(ffn, 30)
            result = compute_pr2(ffn)
            assert not result.empty
            assert "A3_ratio" in result.columns
            assert "G3_ratio" in result.columns
            assert (result["A3_ratio"] >= 0).all() and (result["A3_ratio"] <= 1).all()
            assert (result["G3_ratio"] >= 0).all() and (result["G3_ratio"] <= 1).all()


class TestDeltaRSCU:
    def test_delta_rscu_computed(self):
        rscu_df = _make_rscu_gene_df(100)
        expr_df = _make_expr_df(100)
        result = compute_delta_rscu(rscu_df, expr_df, "CAI_class")
        assert not result.empty
        assert "delta_rscu" in result.columns
        assert "genome_avg_rscu" in result.columns
        assert "high_expr_rscu" in result.columns

    def test_delta_rscu_missing_class(self):
        rscu_df = _make_rscu_gene_df(50)
        expr_df = _make_expr_df(50)
        result = compute_delta_rscu(rscu_df, expr_df, "nonexistent_class")
        assert result.empty


class TestTRNA:
    def test_extract_trna_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff = Path(tmpdir) / "test.gff"
            _write_test_gff(gff)
            result = extract_trna_counts_from_gff(gff)
            assert not result.empty
            assert "tRNA_copy_number" in result.columns
            # We wrote 2 Ala(TGC), 1 Gly(GCC), 1 Leu(CAA)
            tgc_row = result[result["anticodon"] == "TGC"]
            assert len(tgc_row) == 1
            assert tgc_row.iloc[0]["tRNA_copy_number"] == 2

    def test_trna_codon_correlation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff = Path(tmpdir) / "test.gff"
            _write_test_gff(gff)
            trna_df = extract_trna_counts_from_gff(gff)
            rscu_df = _make_rscu_gene_df(50)
            result = compute_trna_codon_correlation(trna_df, rscu_df)
            assert not result.empty
            assert "tRNA_copy_number" in result.columns
            assert "rscu_all_genes" in result.columns


class TestReverseComplement:
    def test_basic(self):
        assert _reverse_complement("ATG") == "CAT"
        assert _reverse_complement("AACGT") == "ACGTT"
        assert _reverse_complement("GCC") == "GGC"


class TestGeneLengthBias:
    def test_basic(self):
        enc_df = _make_enc_df(50)
        result = compute_gene_length_bias(enc_df)
        assert "length" in result.columns
        assert "ENC" in result.columns

    def test_merges_expression(self):
        enc_df = _make_enc_df(50)
        expr_df = _make_expr_df(50)
        result = compute_gene_length_bias(enc_df, expr_df)
        assert "CAI" in result.columns
        assert "MELP" in result.columns
