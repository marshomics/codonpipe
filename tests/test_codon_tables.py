"""Tests for the codon tables module."""

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    AA_CODON_GROUPS_RSCU,
    AMINO_ACID_FAMILIES,
    CODON_TABLE_11,
    RSCU_COLUMN_NAMES,
    SENSE_CODONS,
    dna_to_rna,
)


def test_codon_table_has_64_codons():
    assert len(CODON_TABLE_11) == 64


def test_sense_codons_exclude_stops_met_trp():
    for codon, aa in SENSE_CODONS.items():
        assert aa not in ("*", "Met", "Trp")
    # Should be 59 sense codons (64 - 3 stops - 1 Met - 1 Trp)
    assert len(SENSE_CODONS) == 59


def test_aa_codon_groups_cover_all_sense():
    all_codons = set()
    for codons in AA_CODON_GROUPS.values():
        all_codons.update(codons)
    assert all_codons == set(SENSE_CODONS.keys())


def test_rscu_column_names_count():
    assert len(RSCU_COLUMN_NAMES) == 59


def test_amino_acid_families_serine_has_6():
    assert len(AMINO_ACID_FAMILIES["Ser"]) == 6


def test_amino_acid_families_leucine_has_6():
    assert len(AMINO_ACID_FAMILIES["Leu"]) == 6


def test_dna_to_rna():
    assert dna_to_rna("ATGCTT") == "AUGCUU"
    assert dna_to_rna("atgctt") == "AUGCUU"


# ── AA_CODON_GROUPS_RSCU split-family tests ──────────────────────────────

def test_rscu_groups_split_serine():
    """Ser must be split into Ser4 (UCN, 4 codons) and Ser2 (AGY, 2 codons)."""
    assert "Ser4" in AA_CODON_GROUPS_RSCU
    assert "Ser2" in AA_CODON_GROUPS_RSCU
    assert "Ser" not in AA_CODON_GROUPS_RSCU
    assert len(AA_CODON_GROUPS_RSCU["Ser4"]) == 4
    assert len(AA_CODON_GROUPS_RSCU["Ser2"]) == 2
    assert all(c.startswith("UC") for c in AA_CODON_GROUPS_RSCU["Ser4"])
    assert all(c.startswith("AG") for c in AA_CODON_GROUPS_RSCU["Ser2"])


def test_rscu_groups_split_leucine():
    """Leu must be split into Leu4 (CUN, 4 codons) and Leu2 (UUN, 2 codons)."""
    assert "Leu4" in AA_CODON_GROUPS_RSCU
    assert "Leu2" in AA_CODON_GROUPS_RSCU
    assert "Leu" not in AA_CODON_GROUPS_RSCU
    assert len(AA_CODON_GROUPS_RSCU["Leu4"]) == 4
    assert len(AA_CODON_GROUPS_RSCU["Leu2"]) == 2


def test_rscu_groups_split_arginine():
    """Arg must be split into Arg4 (CGN, 4 codons) and Arg2 (AGR, 2 codons)."""
    assert "Arg4" in AA_CODON_GROUPS_RSCU
    assert "Arg2" in AA_CODON_GROUPS_RSCU
    assert "Arg" not in AA_CODON_GROUPS_RSCU
    assert len(AA_CODON_GROUPS_RSCU["Arg4"]) == 4
    assert len(AA_CODON_GROUPS_RSCU["Arg2"]) == 2


def test_rscu_groups_cover_all_sense_codons():
    """AA_CODON_GROUPS_RSCU must cover exactly the same 59 sense codons as AA_CODON_GROUPS."""
    rscu_codons = set()
    for codons in AA_CODON_GROUPS_RSCU.values():
        rscu_codons.update(codons)
    original_codons = set()
    for codons in AA_CODON_GROUPS.values():
        original_codons.update(codons)
    assert rscu_codons == original_codons


def test_rscu_groups_non_split_aas_unchanged():
    """Amino acids other than Ser/Leu/Arg should appear identically in both dicts."""
    for aa in ("Phe", "Tyr", "His", "Gln", "Asn", "Lys", "Asp", "Glu",
               "Cys", "Pro", "Thr", "Ala", "Val", "Gly", "Ile"):
        assert aa in AA_CODON_GROUPS_RSCU
        assert set(AA_CODON_GROUPS_RSCU[aa]) == set(AA_CODON_GROUPS[aa])
