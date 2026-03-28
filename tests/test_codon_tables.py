"""Tests for the codon tables module."""

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
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
