"""Tests for the RSCU computation module."""

from collections import Counter

import numpy as np
import pytest

from codonpipe.modules.rscu import (
    compute_rscu_from_counts,
    count_codons,
    _calculate_enc,
    _calculate_gc3,
)
from codonpipe.utils.codon_tables import codon_to_col_name as _codon_to_col_name


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
        """If an amino acid has zero total counts, RSCU should be NaN (unobserved)."""
        counts = Counter()  # empty
        rscu = compute_rscu_from_counts(counts)
        import math
        assert math.isnan(rscu["Phe-UUU"])


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


class TestComputeRSCUSplitFamilies:
    """Verify that split families (Ser4/Ser2, Leu4/Leu2, Arg4/Arg2) are computed independently."""

    def test_serine_split_independence(self):
        """Ser4 (UCN) and Ser2 (AGY) RSCU should be computed from their own subfamily totals."""
        # All Ser4 usage in UCU, zero in UCC/UCA/UCG → Ser4-UCU RSCU = 4.0
        # All Ser2 usage in AGU, zero in AGC → Ser2-AGU RSCU = 2.0
        counts = Counter({
            "UCU": 100, "UCC": 0, "UCA": 0, "UCG": 0,
            "AGU": 50, "AGC": 0,
        })
        rscu = compute_rscu_from_counts(counts)
        assert abs(rscu["Ser4-UCU"] - 4.0) < 0.01
        assert abs(rscu["Ser4-UCC"] - 0.0) < 0.01
        assert abs(rscu["Ser2-AGU"] - 2.0) < 0.01
        assert abs(rscu["Ser2-AGC"] - 0.0) < 0.01

    def test_serine_equal_usage(self):
        """Equal usage within each subfamily → RSCU = 1.0 for all."""
        counts = Counter({
            "UCU": 25, "UCC": 25, "UCA": 25, "UCG": 25,
            "AGU": 50, "AGC": 50,
        })
        rscu = compute_rscu_from_counts(counts)
        for col in ["Ser4-UCU", "Ser4-UCC", "Ser4-UCA", "Ser4-UCG"]:
            assert abs(rscu[col] - 1.0) < 0.01
        for col in ["Ser2-AGU", "Ser2-AGC"]:
            assert abs(rscu[col] - 1.0) < 0.01

    def test_leucine_split_independence(self):
        """Leu4 and Leu2 computed from their own subfamily totals."""
        counts = Counter({
            "CUU": 100, "CUC": 100, "CUA": 0, "CUG": 0,  # Leu4
            "UUA": 80, "UUG": 20,  # Leu2
        })
        rscu = compute_rscu_from_counts(counts)
        # Leu4: CUU and CUC each have 100/200 * 4 = 2.0
        assert abs(rscu["Leu4-CUU"] - 2.0) < 0.01
        assert abs(rscu["Leu4-CUA"] - 0.0) < 0.01
        # Leu2: UUA has 80/100 * 2 = 1.6
        assert abs(rscu["Leu2-UUA"] - 1.6) < 0.01
        assert abs(rscu["Leu2-UUG"] - 0.4) < 0.01

    def test_arginine_split_independence(self):
        """Arg4 and Arg2 computed from their own subfamily totals."""
        counts = Counter({
            "CGU": 25, "CGC": 25, "CGA": 25, "CGG": 25,  # Arg4
            "AGA": 100, "AGG": 0,  # Arg2
        })
        rscu = compute_rscu_from_counts(counts)
        for col in ["Arg4-CGU", "Arg4-CGC", "Arg4-CGA", "Arg4-CGG"]:
            assert abs(rscu[col] - 1.0) < 0.01
        assert abs(rscu["Arg2-AGA"] - 2.0) < 0.01
        assert abs(rscu["Arg2-AGG"] - 0.0) < 0.01


class TestENC:
    def test_extreme_bias_low_enc(self):
        """Extreme bias should give low ENC (close to 20)."""
        counts = Counter()
        from codonpipe.utils.codon_tables import AA_CODON_GROUPS
        # Use only the first codon per amino acid, 100 counts each
        for aa, codons in AA_CODON_GROUPS.items():
            counts[codons[0]] = 100
            for c in codons[1:]:
                counts[c] = 0
        enc, had_f_zero = _calculate_enc(counts)
        assert enc < 30, f"Extreme bias should give ENC < 30, got {enc}"
        assert enc >= 20, f"ENC cannot be below 20, got {enc}"

    def test_no_bias_high_enc(self):
        """Equal codon usage should give high ENC (close to 61)."""
        counts = Counter()
        from codonpipe.utils.codon_tables import AA_CODON_GROUPS
        for aa, codons in AA_CODON_GROUPS.items():
            for codon in codons:
                counts[codon] = 100
        enc, had_f_zero = _calculate_enc(counts)
        assert enc > 58, f"No bias should give ENC > 58, got {enc}"
        assert enc <= 61, f"ENC cannot exceed 61, got {enc}"

    def test_empty_counts_gives_no_bias(self):
        """Empty codon counts (no data) should assume no bias → ENC near 61."""
        counts = Counter()
        enc, had_f_zero = _calculate_enc(counts)
        # With no observed amino acids, fallback assumes no bias: F=1/k
        # This should give ENC = 2 + 9*2 + 1*3 + 5*4 + 3*6 = 2+18+3+20+18 = 61
        assert abs(enc - 61.0) < 0.01, f"Empty counts should give ENC=61, got {enc}"

    def test_enc_bounded(self):
        """ENC should always be in [20, 61]."""
        from codonpipe.utils.codon_tables import AA_CODON_GROUPS
        rng = np.random.RandomState(123)
        for _ in range(20):
            counts = Counter()
            for aa, codons in AA_CODON_GROUPS.items():
                for codon in codons:
                    counts[codon] = rng.randint(0, 200)
            enc, had_f_zero = _calculate_enc(counts)
            assert 20 <= enc <= 61, f"ENC {enc} outside valid range [20, 61]"
