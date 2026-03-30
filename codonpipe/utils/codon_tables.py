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

# Synonymous codon families for RSCU computation.
# Ser, Leu, and Arg are SPLIT into 4-fold and 2-fold subfamilies because
# the two groups occupy different codon boxes and cannot interconvert via
# single nucleotide substitutions.  Pooling all 6 codons inflates/deflates
# RSCU values across subfamilies (see Shields et al. 1988; Sharp et al. 1986).
# ENC computation (Wright 1990) should still treat them as 6-fold —
# use AA_CODON_GROUPS for that purpose.
AA_CODON_GROUPS_RSCU: dict[str, list[str]] = {}
for _aa, _codons in AA_CODON_GROUPS.items():
    if _aa == "Ser":
        AA_CODON_GROUPS_RSCU["Ser4"] = [c for c in _codons if c.startswith("UC")]
        AA_CODON_GROUPS_RSCU["Ser2"] = [c for c in _codons if not c.startswith("UC")]
    elif _aa == "Leu":
        AA_CODON_GROUPS_RSCU["Leu4"] = [c for c in _codons if c.startswith("CU")]
        AA_CODON_GROUPS_RSCU["Leu2"] = [c for c in _codons if not c.startswith("CU")]
    elif _aa == "Arg":
        AA_CODON_GROUPS_RSCU["Arg4"] = [c for c in _codons if c.startswith("CG")]
        AA_CODON_GROUPS_RSCU["Arg2"] = [c for c in _codons if not c.startswith("CG")]
    else:
        AA_CODON_GROUPS_RSCU[_aa] = _codons


# DNA to RNA codon conversion
# Minimum gene length (nucleotides) to include in codon analyses.
# A single shared constant used by all analysis modules to ensure consistent
# gene filtering.  240 nt ≈ 80 codons, the minimum for stable RSCU estimates.
MIN_GENE_LENGTH = 240


# ── Canonical column names shared across modules ──────────────────────────
# Defining these once prevents silent breakage when modules are updated
# independently.  Import these constants instead of hardcoding strings.

# Core DataFrame key columns
COL_GENE = "gene"
COL_WIDTH = "width"
COL_SAMPLE_ID = "sample_id"

# Expression metric score columns (produced by expression.py, consumed everywhere)
COL_MELP = "MELP"
COL_CAI = "CAI"
COL_FOP = "Fop"
EXPRESSION_METRICS = (COL_MELP, COL_CAI, COL_FOP)

# Expression classification columns
COL_MELP_CLASS = "MELP_class"
COL_CAI_CLASS = "CAI_class"
COL_FOP_CLASS = "Fop_class"
COL_EXPRESSION_CLASS = "expression_class"
EXPRESSION_CLASS_COLS = (COL_MELP_CLASS, COL_CAI_CLASS, COL_FOP_CLASS)

# ENC-related columns (produced by rscu.py and advanced_analyses.py)
COL_ENC = "ENC"
COL_ENCPRIME = "ENCprime"
COL_ENC_DIFF = "ENC_diff"
COL_ENCPRIME_RESIDUAL = "ENCprime_residual"

# GC content columns
COL_GC3 = "GC3"
COL_GC12 = "GC12"

# Mahalanobis cluster membership flag
COL_IN_MAHAL_CLUSTER = "in_mahal_cluster"

# RP-prefixed columns for RP-based scores preserved alongside Mahalanobis-based
RP_PREFIX = "rp_"


def dna_to_rna(seq: str) -> str:
    """Convert a DNA sequence to RNA."""
    return seq.upper().replace("T", "U")


def codon_to_col_name(codon: str, aa: str) -> str:
    """Convert a codon and amino acid to the RSCU column name convention.

    Serine, Leucine, and Arginine are split into two families:
        Ser4 (UCN) vs Ser2 (AGY), Leu4 (CUN) vs Leu2 (UUN),
        Arg4 (CGN) vs Arg2 (AGR).
    """
    if aa == "Ser":
        return f"Ser4-{codon}" if codon.startswith("UC") else f"Ser2-{codon}"
    elif aa == "Leu":
        return f"Leu4-{codon}" if codon.startswith("CU") else f"Leu2-{codon}"
    elif aa == "Arg":
        return f"Arg4-{codon}" if codon.startswith("CG") else f"Arg2-{codon}"
    else:
        return f"{aa}-{codon}"
