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
    compute_rscu_distance,
    compute_trna_codon_correlation,
    extract_trna_counts_from_gff,
    _explicit_valid_anticodon,
    _infer_anticodon_from_sequence,
    _normalize_trna_aa,
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


class TestRSCUDistance:
    def test_rscu_distance_computed(self):
        rscu_df = _make_rscu_gene_df(50)
        rscu_rp = {col: 1.0 for col in RSCU_COLUMN_NAMES}
        result = compute_rscu_distance(rscu_df, rscu_rp)
        assert "RSCU_distance" in result.columns
        assert len(result) == 50
        assert (result["RSCU_distance"] >= 0).all()

    def test_rscu_distance_no_reference(self):
        rscu_df = _make_rscu_gene_df(50)
        result = compute_rscu_distance(rscu_df, None)
        assert result.empty

    def test_rscu_distance_identical_to_ref_is_zero(self):
        """A gene with identical RSCU to reference should have distance=0."""
        ref = {col: 1.5 for col in RSCU_COLUMN_NAMES}
        data = {"gene": ["test_gene"], "length": [600]}
        for col in RSCU_COLUMN_NAMES:
            data[col] = [1.5]
        rscu_df = pd.DataFrame(data)
        result = compute_rscu_distance(rscu_df, ref)
        assert abs(result.iloc[0]["RSCU_distance"]) < 1e-10


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
            # New long-format schema (one row per (anticodon, decoded codon)
            # under dos Reis 2004 wobble decoding) preserves the original
            # tRNA_copy_number and adds wobble_weight + effective_tRNA.
            for col in ("tRNA_copy_number", "wobble_weight", "effective_tRNA"):
                assert col in result.columns
            # We wrote 2 Ala(TGC), 1 Gly(GCC), 1 Leu(CAA).
            # TGC anticodon: 5' base T (=U in RNA) wobbles → reads codons GCA
            # (Watson-Crick, weight 1.0) and GCG (wobble, weight 0.561).
            tgc_rows = result[result["anticodon"] == "TGC"]
            assert len(tgc_rows) == 2
            # Copy number is unchanged across the rows (still 2 tRNA genes).
            assert (tgc_rows["tRNA_copy_number"] == 2).all()
            decoded = set(tgc_rows["codon"])
            assert decoded == {"GCA", "GCG"}

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


def _trna_gene_with_anticodon(anticodon_dna: str, length: int = 76) -> str:
    """Synthetic tRNA gene: the anticodon at the canonical index 33, padded with
    C so no other amino-acid-consistent triplet falls in the search window."""
    pad = "C" * 33
    tail = "C" * (length - 33 - 3)
    return pad + anticodon_dna + tail


class TestTRNANcbiAnnotations:
    """NCBI/RefSeq GFFs put only the amino acid in product= (e.g.
    product=tRNA-Ile) and sometimes a wrong anticodon in the Note. The parser
    must recover the anticodon from the genome sequence and override bad Notes,
    while still trusting Prokka's inline anticodons."""

    def test_normalize_trna_aa(self):
        assert _normalize_trna_aa("Ile") == "Ile"
        assert _normalize_trna_aa("fMet") == "Met"
        assert _normalize_trna_aa("Ile2") == "Ile"   # strip isoacceptor suffix
        assert _normalize_trna_aa("OTHER") is None
        assert _normalize_trna_aa("Xaa") is None
        assert _normalize_trna_aa("") is None

    def test_explicit_valid_anticodon_accepts_consistent(self):
        # TGC decodes Ala (Watson-Crick GCA) -> accepted.
        assert _explicit_valid_anticodon("product=tRNA-Ala(TGC)", "Ala") == "TGC"
        # anticodon= attribute form.
        assert _explicit_valid_anticodon("anticodon=tgc;x=1", "Ala") == "TGC"

    def test_explicit_valid_anticodon_rejects_inconsistent(self):
        # AGC decodes Ala, not Arg -> rejected (real NCBI annotation error).
        assert _explicit_valid_anticodon("product=tRNA-Arg(AGC)", "Arg") is None
        # GAU/GAT does not decode Tyr -> rejected.
        assert _explicit_valid_anticodon("transfer RNA-Tyr(GAU)", "Tyr") is None

    def test_infer_anticodon_from_sequence(self):
        gene = _trna_gene_with_anticodon("TGC")        # Ala anticodon at idx 33
        assert _infer_anticodon_from_sequence(gene, "Ala") == "TGC"
        # Amino-acid constraint: Trp's only anticodon (CCA) is absent from this
        # poly-C + TGC gene, so no spurious match is returned.
        assert _infer_anticodon_from_sequence(gene, "Trp") is None

    def test_amino_acid_only_recovered_from_genome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gene = _trna_gene_with_anticodon("TGC")
            (tmp / "genome.fasta").write_text(f">chr1\n{gene}\n")
            gff = tmp / "x.gff"
            gff.write_text(
                "##gff-version 3\n"
                f"chr1\tRefSeq\ttRNA\t1\t{len(gene)}\t.\t+\t.\t"
                "ID=t1;product=tRNA-Ala\n"   # amino acid only, no anticodon
            )
            df = extract_trna_counts_from_gff(gff, genome_fasta=tmp / "genome.fasta")
            assert not df.empty
            assert set(df["anticodon"]) == {"TGC"}
            assert (df["amino_acid"] == "Ala").all()

    def test_wrong_note_overridden_by_genome(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gene = _trna_gene_with_anticodon("ACG")    # true Arg anticodon
            (tmp / "g.fasta").write_text(f">chr1\n{gene}\n")
            gff = tmp / "x.gff"
            gff.write_text(
                "##gff-version 3\n"
                f"chr1\tRefSeq\ttRNA\t1\t{len(gene)}\t.\t+\t.\t"
                "ID=t1;product=tRNA-Arg;Note=transfer RNA-Arg(AGC)\n"  # AGC is wrong
            )
            df = extract_trna_counts_from_gff(gff, genome_fasta=tmp / "g.fasta")
            anticodons = set(df["anticodon"])
            assert "ACG" in anticodons       # genome-inferred, correct
            assert "AGC" not in anticodons   # bad Note rejected

    def test_embedded_fasta_used_when_no_genome_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gene = _trna_gene_with_anticodon("TGC")
            gff = tmp / "x.gff"
            gff.write_text(
                "##gff-version 3\n"
                f"chr1\tProkka\ttRNA\t1\t{len(gene)}\t.\t+\t.\t"
                "ID=t1;product=tRNA-Ala\n"
                "##FASTA\n"
                f">chr1\n{gene}\n"
            )
            df = extract_trna_counts_from_gff(gff)   # no genome_fasta arg
            assert set(df["anticodon"]) == {"TGC"}

    def test_minus_strand_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            gene = _trna_gene_with_anticodon("TGC")      # sense strand
            contig = _reverse_complement(gene)           # store on the - strand
            (tmp / "g.fasta").write_text(f">chr1\n{contig}\n")
            gff = tmp / "x.gff"
            gff.write_text(
                "##gff-version 3\n"
                f"chr1\tRefSeq\ttRNA\t1\t{len(contig)}\t.\t-\t.\t"
                "ID=t1;product=tRNA-Ala\n"
            )
            df = extract_trna_counts_from_gff(gff, genome_fasta=tmp / "g.fasta")
            assert set(df["anticodon"]) == {"TGC"}


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
