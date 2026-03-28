"""Tests for the RSCU computation module."""

from collections import Counter

import numpy as np
import pytest

from codonpipe.modules.rscu import (
    compute_rscu_from_counts,
    count_codons,
    _calculate_enc,
    _calculate_gc3,
    _codon_to_col_name,
)


class TestCountCodons:
    def test_simple_sequence(self):
        # ATG = AUG (Met), AAA (Lys), TTT = UUU (Phe), TAA = UAA (stop)
        seq = "ATGAAATTTTAA"
        counts = count_codons(seq)
        assert counts["AUG"] == 1
        assert counts["AAA"] == 1
        assert counts["UUU"] == 1
        assert counts["UAA"] == 1

    def test_rna_input(self):
        seq = "AUGAAAUUUUAA"
        counts = count_codons(seq)
        assert counts["AUG"] == 1

    def test_ignores_incomplete_codons(self):
        seq = "ATGAAATT"  # last "TT" is incomplete
        counts = count_codons(seq)
        assert sum(counts.values()) == 2

    def test_empty_sequence(self):
        counts = count_codons("")
        assert sum(counts.values()) == 0


class TestComputeRSCU:
    def test_equal_usage(self):
        """When all synonymous codons are used equally, RSCU = 1.0 for each."""
        # Phe has 2 codons: UUU, UUC
        counts = Counter({"UUU": 50, "UUC": 50})
        rscu = compute_rscu_from_counts(counts)
        assert abs(rscu["Phe-UUU"] - 1.0) < 0.01
        assert abs(rscu["Phe-UUC"] - 1.0) < 0.01

    def test_extreme_bias(self):
        """When only one codon is used, RSCU = n_synonymous."""
        counts = Counter({"UUU": 100, "UUC": 0})
        rscu = compute_rscu_from_counts(counts)
        assert abs(rscu["Phe-UUU"] - 2.0) < 0.01
        assert abs(rscu["Phe-UUC"] - 0.0) < 0.01

    def test_four_fold_degenerate(self):
        """Val has 4 codons; equal usage → RSCU = 1.0 each."""
        counts = Counter({"GUU": 25, "GUC": 25, "GUA": 25, "GUG": 25})
        rscu = compute_rscu_from_counts(counts)
        for codon_col in ["Val-GUU", "Val-GUC", "Val-GUA", "Val-GUG"]:
            assert abs(rscu[codon_col] - 1.0) < 0.01

    def test_zero_amino_acid(self):
        """If an amino acid has zero total counts, RSCU should be 0."""
        counts = Counter()  # empty
        rscu = compute_rscu_from_counts(counts)
        assert rscu["Phe-UUU"] == 0.0


class TestCodonToColName:
    def test_serine_split(self):
        assert _codon_to_col_name("UCU", "Ser") == "Ser4-UCU"
        assert _codon_to_col_name("AGU", "Ser") == "Ser2-AGU"

    def test_leucine_split(self):
        assert _codon_to_col_name("CUU", "Leu") == "Leu4-CUU"
        assert _codon_to_col_name("UUA", "Leu") == "Leu2-UUA"

    def test_arginine_split(self):
        assert _codon_to_col_name("CGU", "Arg") == "Arg4-CGU"
        assert _codon_to_col_name("AGA", "Arg") == "Arg2-AGA"

    def test_simple_amino_acid(self):
        assert _codon_to_col_name("UUU", "Phe") == "Phe-UUU"
        assert _codon_to_col_name("GCU", "Ala") == "Ala-GCU"


class TestGC3:
    def test_all_gc(self):
        # Codons: GCC GCG GCC → GC3 positions: C, G, C → 100%
        assert abs(_calculate_gc3("GCCGCGGCC") - 1.0) < 0.01

    def test_all_at(self):
        # Codons: AAA TTT → GC3 positions: A, T → 0%
        assert abs(_calculate_gc3("AAATTT") - 0.0) < 0.01

    def test_mixed(self):
        # Codons: AAG (G) AAT (T) → 50%
        assert abs(_calculate_gc3("AAGAAT") - 0.5) < 0.01


class TestENC:
    def test_extreme_bias_low_enc(self):
        """Extreme bias should give low ENC (close to 20)."""
        # Use only one codon per amino acid
        counts = Counter({
            "UUU": 100, "UUC": 0,  # Phe
            "UUA": 100, "UUG": 0, "CUU": 0, "CUC": 0, "CUA": 0, "CUG": 0,  # Leu
            "AUU": 100, "AUC": 0, "AUA": 0,  # Ile
            "GUU": 100, "GUC": 0, "GUA": 0, "GUG": 0,  # Val
        })
        enc = _calculate_enc(counts)
        # With extreme bias, ENC should be relatively low
        assert enc < 40

    def test_no_bias_high_enc(self):
        """Equal codon usage should give high ENC (close to 61)."""
        counts = Counter()
        from codonpipe.utils.codon_tables import AA_CODON_GROUPS
        for aa, codons in AA_CODON_GROUPS.items():
            for codon in codons:
                counts[codon] = 100
        enc = _calculate_enc(counts)
        assert enc > 50
