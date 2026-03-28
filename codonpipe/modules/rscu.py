"""Module for computing Relative Synonymous Codon Usage (RSCU) and codon usage tables."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    AA_CODON_GROUPS_RSCU,
    CODON_TABLE_11,
    RSCU_COLUMN_NAMES,
    SENSE_CODONS,
    dna_to_rna,
)

logger = logging.getLogger("codonpipe")

# Minimum gene length (nucleotides) to include in codon analysis
MIN_GENE_LENGTH = 240


def count_codons(sequence: str) -> Counter:
    """Count codons in a nucleotide sequence (DNA or RNA).

    Converts to RNA internally. Ignores incomplete codons at the end.
    """
    rna = dna_to_rna(sequence)
    counts = Counter()
    for i in range(0, len(rna) - 2, 3):
        codon = rna[i : i + 3]
        if codon in CODON_TABLE_11:
            counts[codon] += 1
    return counts


def compute_rscu_from_counts(codon_counts: Counter) -> dict[str, float]:
    """Compute RSCU values from codon counts.

    RSCU = (observed frequency of codon) / (expected frequency if all synonymous
    codons were used equally) = (count_i * n_synonymous) / sum(counts for that AA)

    Ser, Leu, and Arg are split into 4-fold and 2-fold subfamilies
    (Ser4/Ser2, Leu4/Leu2, Arg4/Arg2) because the two groups occupy
    different codon boxes and should not be pooled (Sharp et al. 1986).

    Returns:
        Dict mapping RSCU column names (e.g., "Phe-UUU") to RSCU values.
    """
    rscu = {}
    for family_name, codons in AA_CODON_GROUPS_RSCU.items():
        total = sum(codon_counts.get(c, 0) for c in codons)
        n_syn = len(codons)
        for codon in codons:
            count = codon_counts.get(codon, 0)
            if total > 0:
                rscu_val = (count * n_syn) / total
            else:
                rscu_val = 0.0

            # Build the column name matching our convention
            col_name = f"{family_name}-{codon}"
            rscu[col_name] = rscu_val

    return rscu


def compute_rscu_per_gene(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> pd.DataFrame:
    """Compute per-gene RSCU values from a nucleotide CDS FASTA file.

    Args:
        ffn_path: Path to nucleotide CDS FASTA (e.g., Prokka .ffn or filtered file).
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with gene IDs as rows and RSCU column names as columns.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        counts = count_codons(seq)
        rscu_vals = compute_rscu_from_counts(counts)
        rscu_vals["gene"] = rec.id
        rscu_vals["length"] = len(seq)
        rows.append(rscu_vals)

    if not rows:
        logger.warning("No genes >= %d nt in %s", min_length, ffn_path)
        return pd.DataFrame(columns=["gene", "length"] + RSCU_COLUMN_NAMES)

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["gene", "length"] + [c for c in RSCU_COLUMN_NAMES if c in df.columns]
    df = df[cols]
    return df


def compute_rscu_genome_summary(
    ffn_path: Path, min_length: int = MIN_GENE_LENGTH
) -> dict[str, float]:
    """Compute genome-level median RSCU across all genes.

    Args:
        ffn_path: Path to nucleotide CDS FASTA.
        min_length: Minimum gene length.

    Returns:
        Dict of median RSCU values per codon.
    """
    gene_df = compute_rscu_per_gene(ffn_path, min_length)
    if gene_df.empty:
        return {col: np.nan for col in RSCU_COLUMN_NAMES}

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in gene_df.columns]
    medians = gene_df[rscu_cols].median().to_dict()
    return medians


def compute_codon_frequency_table(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> pd.DataFrame:
    """Compute absolute and relative codon frequency tables.

    Returns DataFrame with columns:
        codon, amino_acid, count, frequency, rscu, per_thousand
    """
    total_counts = Counter()
    n_genes = 0
    total_codons = 0

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        n_genes += 1
        gene_counts = count_codons(seq)
        total_counts += gene_counts
        total_codons += sum(gene_counts.values())

    if total_codons == 0:
        return pd.DataFrame(columns=["codon", "amino_acid", "count", "frequency", "rscu", "per_thousand"])

    rows = []
    rscu_all = compute_rscu_from_counts(total_counts)

    for codon, aa in sorted(CODON_TABLE_11.items()):
        count = total_counts.get(codon, 0)
        freq = count / total_codons if total_codons > 0 else 0
        per_thousand = freq * 1000
        col_name = _codon_to_col_name(codon, aa) if aa not in ("*", "Met", "Trp") else f"{aa}-{codon}"
        rscu_val = rscu_all.get(col_name, np.nan)

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "count": count,
            "frequency": round(freq, 6),
            "rscu": round(rscu_val, 4) if not np.isnan(rscu_val) else np.nan,
            "per_thousand": round(per_thousand, 2),
        })

    return pd.DataFrame(rows)


def compute_concatenated_rscu(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> dict[str, float]:
    """Compute RSCU from a concatenated set of sequences (e.g., all ribosomal proteins).

    Instead of per-gene RSCU followed by median, this pools all codons first,
    then computes RSCU. More appropriate for a reference set.
    """
    total_counts = Counter()
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        total_counts += count_codons(seq)

    return compute_rscu_from_counts(total_counts)


def compute_enc(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> pd.DataFrame:
    """Compute Effective Number of Codons (ENC/Nc) per gene.

    ENC ranges from 20 (extreme bias, one codon per amino acid)
    to 61 (no bias, all synonymous codons used equally).

    Uses the Wright (1990) formula.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        counts = count_codons(seq)
        enc = _calculate_enc(counts)
        gc3 = _calculate_gc3(seq)
        rows.append({"gene": rec.id, "length": len(seq), "ENC": enc, "GC3": gc3})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["gene", "length", "ENC", "GC3"])


def _calculate_enc(codon_counts: Counter) -> float:
    """Calculate ENC using the Wright (1990) method."""
    # Group by number of synonymous codons (degeneracy class)
    # k=1: Met, Trp (skip)
    # k=2: Phe, Tyr, His, Gln, Asn, Lys, Asp, Glu, Cys
    # k=3: Ile
    # k=4: Val, Pro, Thr, Ala, Gly
    # k=6: Leu, Arg, Ser

    f_values = {2: [], 3: [], 4: [], 6: []}

    for aa, codons in AA_CODON_GROUPS.items():
        k = len(codons)
        if k == 1:
            continue

        n_aa = sum(codon_counts.get(c, 0) for c in codons)
        if n_aa <= 1:
            continue

        # Homozygosity F = (n * sum(p_i^2) - 1) / (n - 1)
        p_sum_sq = sum((codon_counts.get(c, 0) / n_aa) ** 2 for c in codons)
        f_hat = (n_aa * p_sum_sq - 1) / (n_aa - 1)

        if k in f_values:
            f_values[k].append(f_hat)
        elif k == 6:
            f_values[6].append(f_hat)

    # ENC = number of aa families + K/F_avg for each degeneracy class
    # Wright (1990): Nc = 2 + 9/F̄₂ + 1/F̄₃ + 5/F̄₄ + 3/F̄₆
    enc = 2.0  # Met + Trp always contribute 1 each
    n_families = {2: 9, 3: 1, 4: 5, 6: 3}

    for k, f_list in f_values.items():
        if f_list:
            f_avg = np.mean(f_list)
            if f_avg > 0:
                enc += n_families[k] / f_avg
            else:
                # F_hat = 0 shouldn't happen with n > 1, but if it does
                # treat as no bias: each family contributes k codons
                enc += n_families[k] * k
        else:
            # No amino acids observed for this degeneracy class.
            # Assume no bias: F = 1/k → contribution = n_families * k
            enc += n_families[k] * k

    return min(enc, 61.0)


def _calculate_gc3(sequence: str) -> float:
    """Calculate GC content at the third codon position."""
    gc3_count = 0
    total = 0
    seq_upper = sequence.upper()
    for i in range(2, len(seq_upper), 3):
        base = seq_upper[i]
        if base in "ACGT":
            total += 1
            if base in "GC":
                gc3_count += 1
    return gc3_count / total if total > 0 else 0.0


def _codon_to_col_name(codon: str, aa: str) -> str:
    """Convert a codon and amino acid to the RSCU column name convention.

    Serine, Leucine, and Arginine are split into two families:
        Ser4 (UCN) vs Ser2 (AGY), Leu4 (CUN) vs Leu2 (UUN),
        Arg4 (CGN) vs Arg2 (AGR).
    """
    if aa == "Ser":
        if codon.startswith("UC"):
            return f"Ser4-{codon}"
        else:
            return f"Ser2-{codon}"
    elif aa == "Leu":
        if codon.startswith("CU"):
            return f"Leu4-{codon}"
        else:
            return f"Leu2-{codon}"
    elif aa == "Arg":
        if codon.startswith("CG"):
            return f"Arg4-{codon}"
        else:
            return f"Arg2-{codon}"
    else:
        return f"{aa}-{codon}"


def run_rscu_analysis(
    ffn_path: Path,
    rp_ffn_path: Path | None,
    output_dir: Path,
    sample_id: str,
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, Path]:
    """Run complete RSCU analysis for a genome.

    Args:
        ffn_path: Path to all CDS nucleotide sequences.
        rp_ffn_path: Path to ribosomal protein nucleotide sequences (or None).
        output_dir: Base output directory.
        sample_id: Sample identifier.
        min_length: Minimum gene length in nt.

    Returns:
        Dict of output file paths.
    """
    rscu_dir = output_dir / "rscu"
    rscu_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Per-gene RSCU for all CDS
    logger.info("Computing per-gene RSCU for all CDS in %s", sample_id)
    all_rscu = compute_rscu_per_gene(ffn_path, min_length)
    all_rscu_path = rscu_dir / f"{sample_id}_rscu_all_genes.tsv"
    all_rscu.to_csv(all_rscu_path, sep="\t", index=False)
    outputs["rscu_all_genes"] = all_rscu_path

    # Genome-level median RSCU
    median_rscu = compute_rscu_genome_summary(ffn_path, min_length)
    median_df = pd.DataFrame([{"sample_id": sample_id, **median_rscu}])
    median_path = rscu_dir / f"{sample_id}_rscu_median.tsv"
    median_df.to_csv(median_path, sep="\t", index=False)
    outputs["rscu_median"] = median_path

    # Codon frequency table (absolute counts + frequencies)
    logger.info("Computing codon frequency table for %s", sample_id)
    freq_table = compute_codon_frequency_table(ffn_path, min_length)
    freq_path = rscu_dir / f"{sample_id}_codon_frequency.tsv"
    freq_table.to_csv(freq_path, sep="\t", index=False)
    outputs["codon_frequency"] = freq_path

    # Ribosomal protein RSCU (concatenated)
    if rp_ffn_path is not None and rp_ffn_path.exists():
        logger.info("Computing concatenated RSCU for ribosomal proteins in %s", sample_id)
        rp_rscu = compute_concatenated_rscu(rp_ffn_path, min_length=0)
        rp_rscu_df = pd.DataFrame([{"sample_id": sample_id, **rp_rscu}])
        rp_rscu_path = rscu_dir / f"{sample_id}_rscu_ribosomal.tsv"
        rp_rscu_df.to_csv(rp_rscu_path, sep="\t", index=False)
        outputs["rscu_ribosomal"] = rp_rscu_path

        # Per-gene RSCU for ribosomal proteins
        rp_gene_rscu = compute_rscu_per_gene(rp_ffn_path, min_length=0)
        rp_gene_path = rscu_dir / f"{sample_id}_rscu_rp_genes.tsv"
        rp_gene_rscu.to_csv(rp_gene_path, sep="\t", index=False)
        outputs["rscu_rp_genes"] = rp_gene_path

    # ENC analysis
    logger.info("Computing ENC for all genes in %s", sample_id)
    enc_df = compute_enc(ffn_path, min_length)
    enc_path = rscu_dir / f"{sample_id}_enc.tsv"
    enc_df.to_csv(enc_path, sep="\t", index=False)
    outputs["enc"] = enc_path

    return outputs
