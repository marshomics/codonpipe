"""Module for generating codon usage tables in all standard formats.

Provides functions to compute and export codon usage statistics including:
- Absolute codon counts
- Frequency per 1000 codons
- RSCU (Relative Synonymous Codon Usage)
- Relative adaptiveness (w values)
- Codon adaptation weights
- CBI (Codon Bias Index)

Supports generation of comprehensive tables for different gene sets:
- All genes (genome-wide)
- Ribosomal protein genes
- High-expression genes
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    CODON_TABLE_11,
    SENSE_CODONS,
    dna_to_rna,
)
from codonpipe.modules.rscu import count_codons, compute_rscu_from_counts

logger = logging.getLogger("codonpipe")

# Minimum gene length (nucleotides) to include in codon analysis
MIN_GENE_LENGTH = 240


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


def _get_sequence_ids_from_file(ffn_path: Path) -> set[str]:
    """Extract all sequence IDs from a FASTA file."""
    ids = set()
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        ids.add(rec.id)
    return ids


def _filter_sequences_by_ids(ffn_path: Path, gene_ids: set[str] | None, min_length: int = MIN_GENE_LENGTH) -> Counter:
    """Count codons from sequences, optionally filtered by gene IDs.

    Returns:
        Counter of all codons found in filtered sequences.
    """
    total_counts = Counter()

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)

        # Filter by gene IDs if specified
        if gene_ids is not None and rec.id not in gene_ids:
            continue

        # Filter by minimum length
        if len(seq) < min_length:
            continue

        gene_counts = count_codons(seq)
        total_counts += gene_counts

    return total_counts


def compute_absolute_counts(
    ffn_path: Path, gene_ids: set[str] | None = None, min_length: int = MIN_GENE_LENGTH
) -> pd.DataFrame:
    """Count codons from sequences in ffn_path, optionally filtered to gene_ids.

    Args:
        ffn_path: Path to nucleotide CDS FASTA file.
        gene_ids: Optional set of gene IDs to include. If None, uses all genes.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: codon, amino_acid, count, total_codons
        Includes all 64 codons (including stops, Met, Trp).
    """
    total_counts = _filter_sequences_by_ids(ffn_path, gene_ids, min_length)

    if not total_counts:
        return pd.DataFrame(columns=["codon", "amino_acid", "count", "total_codons"])

    total_codons = sum(total_counts.values())
    rows = []

    for codon, aa in sorted(CODON_TABLE_11.items()):
        count = total_counts.get(codon, 0)
        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "count": count,
            "total_codons": total_codons,
        })

    return pd.DataFrame(rows)


def compute_frequency_per_thousand(
    ffn_path: Path, gene_ids: set[str] | None = None, min_length: int = MIN_GENE_LENGTH
) -> pd.DataFrame:
    """Compute frequency per 1000 codons from sequences.

    Args:
        ffn_path: Path to nucleotide CDS FASTA file.
        gene_ids: Optional set of gene IDs to include.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: codon, amino_acid, count, per_thousand
    """
    total_counts = _filter_sequences_by_ids(ffn_path, gene_ids, min_length)

    if not total_counts:
        return pd.DataFrame(columns=["codon", "amino_acid", "count", "per_thousand"])

    total_codons = sum(total_counts.values())
    rows = []

    for codon, aa in sorted(CODON_TABLE_11.items()):
        count = total_counts.get(codon, 0)
        per_thousand = (count / total_codons * 1000) if total_codons > 0 else 0.0
        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "count": count,
            "per_thousand": round(per_thousand, 2),
        })

    return pd.DataFrame(rows)


def compute_rscu_table(
    ffn_path: Path, gene_ids: set[str] | None = None, min_length: int = MIN_GENE_LENGTH
) -> pd.DataFrame:
    """Compute RSCU from concatenated sequences.

    RSCU = (observed frequency of codon) / (expected frequency if all synonymous
    codons were used equally) = (count_i * n_synonymous) / sum(counts for that AA)

    Args:
        ffn_path: Path to nucleotide CDS FASTA file.
        gene_ids: Optional set of gene IDs to include.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: codon, amino_acid, count, rscu
        Only includes sense codons (excludes stops, Met, Trp).
    """
    total_counts = _filter_sequences_by_ids(ffn_path, gene_ids, min_length)

    if not total_counts:
        return pd.DataFrame(columns=["codon", "amino_acid", "count", "rscu"])

    rscu_vals = compute_rscu_from_counts(total_counts)
    rows = []

    for codon in sorted(SENSE_CODONS.keys()):
        aa = CODON_TABLE_11[codon]
        count = total_counts.get(codon, 0)
        col_name = _codon_to_col_name(codon, aa)
        rscu_val = rscu_vals.get(col_name, np.nan)

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "count": count,
            "rscu": round(rscu_val, 4) if not np.isnan(rscu_val) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_relative_adaptiveness(
    ffn_path: Path, gene_ids: set[str] | None = None, min_length: int = MIN_GENE_LENGTH
) -> pd.DataFrame:
    """Compute relative adaptiveness (w values) for codon adaptation index (CAI).

    w_ij = RSCU_ij / RSCU_max_j where RSCU_max_j is the maximum RSCU among
    synonymous codons for amino acid j.

    Used for CAI calculation: CAI = geometric mean of w values

    Args:
        ffn_path: Path to nucleotide CDS FASTA file.
        gene_ids: Optional set of gene IDs to include.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: codon, amino_acid, rscu, w_value
    """
    total_counts = _filter_sequences_by_ids(ffn_path, gene_ids, min_length)

    if not total_counts:
        return pd.DataFrame(columns=["codon", "amino_acid", "rscu", "w_value"])

    rscu_vals = compute_rscu_from_counts(total_counts)

    # Compute maximum RSCU per amino acid
    max_rscu_per_aa = {}
    for aa, codons in AA_CODON_GROUPS.items():
        rscu_list = []
        for codon in codons:
            col_name = _codon_to_col_name(codon, aa)
            rscu_val = rscu_vals.get(col_name, 0.0)
            rscu_list.append(rscu_val)
        max_rscu_per_aa[aa] = max(rscu_list) if rscu_list else 1.0

    rows = []
    for codon in sorted(SENSE_CODONS.keys()):
        aa = CODON_TABLE_11[codon]
        col_name = _codon_to_col_name(codon, aa)
        rscu_val = rscu_vals.get(col_name, 0.0)
        max_rscu = max_rscu_per_aa[aa]
        w_value = rscu_val / max_rscu if max_rscu > 0 else 0.0

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "rscu": round(rscu_val, 4),
            "w_value": round(w_value, 4),
        })

    return pd.DataFrame(rows)


def compute_codon_adaptation_weights(
    ffn_ref_path: Path,
    ffn_all_path: Path,
    ref_gene_ids: set[str] | None = None,
    min_length: int = MIN_GENE_LENGTH,
) -> pd.DataFrame:
    """Compute codon adaptation weights comparing reference set to all genes.

    Weight = ln(RSCU_ref / RSCU_all) for each codon.
    Positive weight = codon preferred in reference (highly expressed) genes.

    Args:
        ffn_ref_path: Path to reference gene sequences (e.g., ribosomal proteins).
        ffn_all_path: Path to all genes.
        ref_gene_ids: Optional set of gene IDs to include from reference file.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: codon, amino_acid, rscu_ref, rscu_all, weight, is_optimal
        is_optimal: True if weight > 0 (preferred in reference set)
    """
    # Compute RSCU for reference set
    ref_counts = _filter_sequences_by_ids(ffn_ref_path, ref_gene_ids, min_length)
    if not ref_counts:
        logger.warning("No reference sequences found in %s", ffn_ref_path)
        return pd.DataFrame(
            columns=["codon", "amino_acid", "rscu_ref", "rscu_all", "weight", "is_optimal"]
        )

    ref_rscu = compute_rscu_from_counts(ref_counts)

    # Compute RSCU for all genes
    all_counts = _filter_sequences_by_ids(ffn_all_path, None, min_length)
    if not all_counts:
        logger.warning("No sequences found in %s", ffn_all_path)
        return pd.DataFrame(
            columns=["codon", "amino_acid", "rscu_ref", "rscu_all", "weight", "is_optimal"]
        )

    all_rscu = compute_rscu_from_counts(all_counts)

    rows = []
    for codon in sorted(SENSE_CODONS.keys()):
        aa = CODON_TABLE_11[codon]
        col_name = _codon_to_col_name(codon, aa)
        rscu_ref = ref_rscu.get(col_name, 0.0)
        rscu_all = all_rscu.get(col_name, 0.0)

        # Compute weight: ln(rscu_ref / rscu_all)
        # Add small pseudocount to avoid log(0)
        pseudocount = 0.01
        weight = np.log((rscu_ref + pseudocount) / (rscu_all + pseudocount))
        is_optimal = weight > 0

        rows.append({
            "codon": codon,
            "amino_acid": aa,
            "rscu_ref": round(rscu_ref, 4),
            "rscu_all": round(rscu_all, 4),
            "weight": round(weight, 4),
            "is_optimal": is_optimal,
        })

    return pd.DataFrame(rows)


def compute_cbi(
    ffn_path: Path,
    optimal_codons: dict[str, str],
    gene_ids: set[str] | None = None,
    min_length: int = MIN_GENE_LENGTH,
) -> pd.DataFrame:
    """Compute Codon Bias Index (CBI) per gene.

    CBI = (N_opt - N_rand) / (N_total - N_rand)
    where:
        N_opt = count of optimal codons
        N_rand = expected count under equal usage
        N_total = total synonymous codons

    Reference: Bennetzen & Hall (1982)

    Args:
        ffn_path: Path to nucleotide CDS FASTA file.
        optimal_codons: Dict mapping amino acid -> optimal codon (from compute_codon_adaptation_weights).
        gene_ids: Optional set of gene IDs to include.
        min_length: Minimum sequence length in nucleotides to include.

    Returns:
        DataFrame with columns: gene, cbi, n_optimal, n_total, length
    """
    rows = []

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)

        # Filter by gene IDs if specified
        if gene_ids is not None and rec.id not in gene_ids:
            continue

        # Filter by minimum length
        if len(seq) < min_length:
            continue

        gene_counts = count_codons(seq)

        # Count optimal codons and compute CBI per amino acid
        n_opt_total = 0
        n_total = 0
        n_aa_total = 0

        for aa, codons in AA_CODON_GROUPS.items():
            n_aa = sum(gene_counts.get(c, 0) for c in codons)
            if n_aa == 0:
                continue

            n_aa_total += 1
            optimal_codon = optimal_codons.get(aa)

            # Count optimal codons
            n_opt = gene_counts.get(optimal_codon, 0) if optimal_codon else 0

            # Expected count under equal usage
            n_expected = n_aa / len(codons)

            # Per-amino-acid contribution
            if n_aa > n_expected:
                n_opt_total += n_opt
                n_total += n_aa

        # Compute CBI
        if n_total > 0:
            n_expected_total = n_total / 20  # Average if equal usage
            cbi = (n_opt_total - n_expected_total) / (n_total - n_expected_total) if (
                n_total - n_expected_total
            ) != 0 else 0.0
        else:
            cbi = np.nan

        rows.append({
            "gene": rec.id,
            "cbi": round(cbi, 4) if not np.isnan(cbi) else np.nan,
            "n_optimal": n_opt_total,
            "n_total": n_total,
            "length": len(seq),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gene", "cbi", "n_optimal", "n_total", "length"]
    )


def generate_all_codon_tables(
    ffn_path: Path,
    rp_ffn_path: Path | None,
    output_dir: Path,
    sample_id: str,
    expr_df: pd.DataFrame | None = None,
    rp_ids_file: Path | None = None,
) -> dict[str, Path]:
    """Generate comprehensive codon usage tables for multiple gene sets.

    Generates all table formats for three gene sets:
    a) All genes (genome-wide)
    b) Ribosomal protein genes (from rp_ffn_path if provided)
    c) High-expression genes (if expr_df available)

    Saves tables as TSVs to output_dir/codon_tables/
    Naming: {sample_id}_{geneset}_{format}.tsv

    Args:
        ffn_path: Path to all CDS nucleotide sequences.
        rp_ffn_path: Path to ribosomal protein sequences (or None).
        output_dir: Base output directory.
        sample_id: Sample identifier.
        expr_df: Optional DataFrame with expression data. Should have columns
                 'gene_id' and either 'expression_class' or 'CAI_class' with values like 'high'.
        rp_ids_file: Optional path to file with ribosomal protein gene IDs (one per line).

    Returns:
        Dict mapping output descriptions to file paths.
    """
    output_dir = Path(output_dir)
    codon_dir = output_dir / "codon_tables"
    codon_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Define gene sets
    gene_sets = {
        "all": {"ffn": ffn_path, "gene_ids": None, "desc": "All genes"},
    }

    # Add ribosomal proteins if available
    if rp_ffn_path and rp_ffn_path.exists():
        if rp_ids_file and rp_ids_file.exists():
            # Load ribosomal protein IDs from file
            rp_ids = set()
            with open(rp_ids_file) as f:
                for line in f:
                    gene_id = line.strip()
                    if gene_id:
                        rp_ids.add(gene_id)
            gene_sets["ribosomal"] = {
                "ffn": ffn_path,
                "gene_ids": rp_ids,
                "desc": "Ribosomal proteins",
            }
        else:
            # Use sequences from rp_ffn_path directly
            gene_sets["ribosomal"] = {
                "ffn": rp_ffn_path,
                "gene_ids": None,
                "desc": "Ribosomal proteins",
            }

    # Add high-expression genes if available
    if expr_df is not None and not expr_df.empty:
        # Try to find high-expression genes
        high_expr_ids = None
        for col in ["expression_class", "CAI_class"]:
            if col in expr_df.columns:
                mask = expr_df[col] == "high"
                if mask.any():
                    gene_col = next(
                        (c for c in ("gene", "gene_id", "gene_name") if c in expr_df.columns),
                        expr_df.columns[0],
                    )
                    high_expr_ids = set(expr_df.loc[mask, gene_col])
                    break

        if high_expr_ids:
            gene_sets["high_expression"] = {
                "ffn": ffn_path,
                "gene_ids": high_expr_ids,
                "desc": "High-expression genes",
            }

    # Generate tables for each gene set
    for geneset_name, geneset_info in gene_sets.items():
        ffn = geneset_info["ffn"]
        gene_ids = geneset_info["gene_ids"]
        desc = geneset_info["desc"]

        logger.info("Generating codon tables for %s (%s)", desc, sample_id)

        # Compute all formats
        abs_counts = compute_absolute_counts(ffn, gene_ids)
        freq_per_1k = compute_frequency_per_thousand(ffn, gene_ids)
        rscu_table = compute_rscu_table(ffn, gene_ids)
        w_values = compute_relative_adaptiveness(ffn, gene_ids)

        # Save individual format tables
        formats = {
            "absolute": abs_counts,
            "frequency_per_thousand": freq_per_1k,
            "rscu": rscu_table,
            "relative_adaptiveness": w_values,
        }

        for fmt_name, df in formats.items():
            if not df.empty:
                out_path = codon_dir / f"{sample_id}_{geneset_name}_{fmt_name}.tsv"
                df.to_csv(out_path, sep="\t", index=False)
                outputs[f"{geneset_name}_{fmt_name}"] = out_path
                logger.info("Saved %s to %s", fmt_name, out_path)

        # Create combined summary table (side-by-side all formats)
        if not rscu_table.empty:
            summary = rscu_table[["codon", "amino_acid"]].copy()
            summary["abs_count"] = abs_counts.set_index("codon")["count"]
            summary["freq_per_1000"] = freq_per_1k.set_index("codon")["per_thousand"]
            summary["rscu"] = rscu_table.set_index("codon")["rscu"]
            summary["w_value"] = w_values.set_index("codon")["w_value"]

            summary_path = codon_dir / f"{sample_id}_{geneset_name}_summary.tsv"
            summary.to_csv(summary_path, sep="\t", index=False)
            outputs[f"{geneset_name}_summary"] = summary_path
            logger.info("Saved combined summary to %s", summary_path)

    # Compute codon adaptation weights (reference vs all)
    if "ribosomal" in gene_sets and "all" in gene_sets:
        logger.info("Computing codon adaptation weights (ribosomal vs all genes)")
        rp_ids = gene_sets["ribosomal"]["gene_ids"]
        weights_df = compute_codon_adaptation_weights(ffn_path, ffn_path, rp_ids)

        if not weights_df.empty:
            weights_path = codon_dir / f"{sample_id}_adaptation_weights.tsv"
            weights_df.to_csv(weights_path, sep="\t", index=False)
            outputs["adaptation_weights"] = weights_path
            logger.info("Saved adaptation weights to %s", weights_path)

            # Extract optimal codons (weight > 0)
            optimal_codons = dict(
                weights_df[weights_df["is_optimal"]][["amino_acid", "codon"]]
                .drop_duplicates(subset=["amino_acid"])
                .set_index("amino_acid")["codon"]
            )

            # Compute CBI for all genes
            if optimal_codons:
                logger.info("Computing CBI for all genes")
                cbi_df = compute_cbi(ffn_path, optimal_codons)

                if not cbi_df.empty:
                    cbi_path = codon_dir / f"{sample_id}_cbi.tsv"
                    cbi_df.to_csv(cbi_path, sep="\t", index=False)
                    outputs["cbi"] = cbi_path
                    logger.info("Saved CBI to %s", cbi_path)

    logger.info("Codon table generation complete. Output directory: %s", codon_dir)
    return outputs
