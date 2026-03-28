"""Codon and amino acid reference tables for CodonPipe."""

from __future__ import annotations

# Standard genetic code (NCBI translation table 11 for bacteria/archaea)
CODON_TABLE_11 = {
    "UUU": "Phe", "UUC": "Phe",
    "UUA": "Leu", "UUG": "Leu", "CUU": "Leu", "CUC": "Leu", "CUA": "Leu", "CUG": "Leu",
    "AUU": "Ile", "AUC": "Ile", "AUA": "Ile",
    "AUG": "Met",
    "GUU": "Val", "GUC": "Val", "GUA": "Val", "GUG": "Val",
    "UCU": "Ser", "UCC": "Ser", "UCA": "Ser", "UCG": "Ser", "AGU": "Ser", "AGC": "Ser",
    "CCU": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACU": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCU": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "UAU": "Tyr", "UAC": "Tyr",
    "UAA": "*", "UAG": "*", "UGA": "*",
    "CAU": "His", "CAC": "His",
    "CAA": "Gln", "CAG": "Gln",
    "AAU": "Asn", "AAC": "Asn",
    "AAA": "Lys", "AAG": "Lys",
    "GAU": "Asp", "GAC": "Asp",
    "GAA": "Glu", "GAG": "Glu",
    "UGU": "Cys", "UGC": "Cys",
    "UGG": "Trp",
    "CGU": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg", "AGA": "Arg", "AGG": "Arg",
    "GGU": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}

# Sense codons only (exclude stops, Met, Trp as they have no synonymous codons)
SENSE_CODONS = {k: v for k, v in CODON_TABLE_11.items() if v not in ("*", "Met", "Trp")}

# Group codons by amino acid
AA_CODON_GROUPS: dict[str, list[str]] = {}
for codon, aa in SENSE_CODONS.items():
    AA_CODON_GROUPS.setdefault(aa, []).append(codon)

# RSCU column names matching the original notebook convention
RSCU_COLUMN_NAMES = [
    "Phe-UUU", "Phe-UUC",
    "Ser4-UCU", "Ser4-UCC", "Ser4-UCA", "Ser4-UCG", "Ser2-AGU", "Ser2-AGC",
    "Leu4-CUU", "Leu4-CUC", "Leu4-CUA", "Leu4-CUG", "Leu2-UUA", "Leu2-UUG",
    "Tyr-UAU", "Tyr-UAC",
    "Cys-UGU", "Cys-UGC",
    "Arg4-CGU", "Arg4-CGC", "Arg4-CGA", "Arg4-CGG", "Arg2-AGA", "Arg2-AGG",
    "Pro-CCU", "Pro-CCC", "Pro-CCA", "Pro-CCG",
    "His-CAU", "His-CAC",
    "Gln-CAA", "Gln-CAG",
    "Ile-AUU", "Ile-AUC", "Ile-AUA",
    "Thr-ACU", "Thr-ACC", "Thr-ACA", "Thr-ACG",
    "Asn-AAU", "Asn-AAC",
    "Lys-AAA", "Lys-AAG",
    "Val-GUU", "Val-GUC", "Val-GUA", "Val-GUG",
    "Ala-GCU", "Ala-GCC", "Ala-GCA", "Ala-GCG",
    "Asp-GAU", "Asp-GAC",
    "Glu-GAA", "Glu-GAG",
    "Gly-GGU", "Gly-GGC", "Gly-GGA", "Gly-GGG",
]

# Mapping from RSCU column name back to codon triplet (RNA)
RSCU_COL_TO_CODON = {}
for col in RSCU_COLUMN_NAMES:
    parts = col.split("-")
    RSCU_COL_TO_CODON[col] = parts[-1]

# Amino acids with their synonymous codon families (for per-amino-acid analyses)
AMINO_ACID_FAMILIES = {
    "Phe": ["Phe-UUU", "Phe-UUC"],
    "Ser": ["Ser4-UCU", "Ser4-UCC", "Ser4-UCA", "Ser4-UCG", "Ser2-AGU", "Ser2-AGC"],
    "Leu": ["Leu4-CUU", "Leu4-CUC", "Leu4-CUA", "Leu4-CUG", "Leu2-UUA", "Leu2-UUG"],
    "Tyr": ["Tyr-UAU", "Tyr-UAC"],
    "Cys": ["Cys-UGU", "Cys-UGC"],
    "Arg": ["Arg4-CGU", "Arg4-CGC", "Arg4-CGA", "Arg4-CGG", "Arg2-AGA", "Arg2-AGG"],
    "Pro": ["Pro-CCU", "Pro-CCC", "Pro-CCA", "Pro-CCG"],
    "His": ["His-CAU", "His-CAC"],
    "Gln": ["Gln-CAA", "Gln-CAG"],
    "Ile": ["Ile-AUU", "Ile-AUC", "Ile-AUA"],
    "Thr": ["Thr-ACU", "Thr-ACC", "Thr-ACA", "Thr-ACG"],
    "Asn": ["Asn-AAU", "Asn-AAC"],
    "Lys": ["Lys-AAA", "Lys-AAG"],
    "Val": ["Val-GUU", "Val-GUC", "Val-GUA", "Val-GUG"],
    "Ala": ["Ala-GCU", "Ala-GCC", "Ala-GCA", "Ala-GCG"],
    "Asp": ["Asp-GAU", "Asp-GAC"],
    "Glu": ["Glu-GAA", "Glu-GAG"],
    "Gly": ["Gly-GGU", "Gly-GGC", "Gly-GGA", "Gly-GGG"],
}

# DNA to RNA codon conversion
def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence to RNA."""
    return seq.upper().replace("T", "U")
