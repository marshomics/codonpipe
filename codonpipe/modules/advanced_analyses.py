"""Advanced codon usage analyses: COA, S-value, neutrality, PR2, delta RSCU,
tRNA-codon correlation, COG enrichment, gene length vs bias, and ENC-ENC' difference.

All analyses are computed from data already available in the pipeline
(per-gene RSCU, ENC, ENCprime, expression tiers, GFF annotations, COG assignments).
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    CODON_TABLE_11,
    RSCU_COLUMN_NAMES,
    RSCU_COL_TO_CODON,
    SENSE_CODONS,
    dna_to_rna,
)

logger = logging.getLogger("codonpipe")


# ─── Correspondence Analysis on RSCU ────────────────────────────────────────


def compute_coa_on_rscu(
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Correspondence Analysis on the per-gene RSCU matrix.

    Projects genes into a low-dimensional space where the major axes
    capture the dominant sources of codon usage variation. In most
    bacterial genomes, axis 1 correlates with expression level.

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        expr_df: Optional expression table with gene and *_class columns.

    Returns:
        Dict with:
            - 'coa_coords': DataFrame (gene, Axis1, Axis2, Axis3, Axis4,
              inertia_pct_1..4, plus expression tier columns if expr_df given)
            - 'coa_codon_coords': DataFrame (codon, Axis1, Axis2)
            - 'coa_inertia': DataFrame (axis, eigenvalue, pct_inertia, cum_pct)
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if len(rscu_cols) < 4 or len(rscu_gene_df) < 10:
        logger.warning("Too few genes/codons for COA (%d genes, %d codons)",
                       len(rscu_gene_df), len(rscu_cols))
        return {}

    X = rscu_gene_df[rscu_cols].values.copy()
    genes = rscu_gene_df["gene"].values

    # Replace NaN with 0 (genuine absence); zeros are meaningful RSCU values
    X = np.where(np.isnan(X), 0.0, X)
    # For chi-squared standardization, add small constant to avoid zero row/column sums
    epsilon = 1e-8

    # Correspondence analysis via SVD on standardized residuals
    row_sums = X.sum(axis=1, keepdims=True) + epsilon
    col_sums = X.sum(axis=0, keepdims=True) + epsilon
    grand_total = X.sum() + epsilon

    # Expected frequencies
    E = (row_sums @ col_sums) / grand_total

    # Standardized residuals (chi-squared metric)
    row_masses = row_sums.ravel() / grand_total
    col_masses = col_sums.ravel() / grand_total

    # Residual matrix
    S = (X / grand_total - E / grand_total) / np.sqrt(E / grand_total)

    # SVD
    n_axes = min(4, min(S.shape) - 1)
    if n_axes < 2:
        return {}

    U, sigma, Vt = np.linalg.svd(S, full_matrices=False)
    # Save all singular values before truncation
    all_sigma = sigma.copy()
    # Keep top axes
    U = U[:, :n_axes]
    sigma = sigma[:n_axes]
    V = Vt[:n_axes, :].T

    eigenvalues = sigma ** 2
    total_inertia = np.sum(all_sigma ** 2)
    pct_inertia = eigenvalues / total_inertia * 100 if total_inertia > 0 else np.zeros(n_axes)

    # Row (gene) coordinates: scale by sqrt(row_masses)
    row_coords = np.diag(1.0 / np.sqrt(row_masses + 1e-30)) @ U * sigma[np.newaxis, :]
    # Column (codon) coordinates
    col_coords = np.diag(1.0 / np.sqrt(col_masses + 1e-30)) @ V * sigma[np.newaxis, :]

    # Build gene coordinate DataFrame
    axis_names = [f"Axis{i+1}" for i in range(n_axes)]
    coa_df = pd.DataFrame(row_coords, columns=axis_names)
    coa_df.insert(0, "gene", genes)

    # Merge expression tiers if available
    if expr_df is not None and "gene" in expr_df.columns:
        class_cols = [c for c in expr_df.columns if c.endswith("_class") and c != "expression_class"]
        if class_cols:
            merge_cols = ["gene"] + class_cols
            available = [c for c in merge_cols if c in expr_df.columns]
            coa_df = coa_df.merge(expr_df[available], on="gene", how="left")

    # Codon coordinates
    codon_df = pd.DataFrame(col_coords[:, :2], columns=["Axis1", "Axis2"])
    codon_df.insert(0, "codon", rscu_cols)

    # Inertia summary
    inertia_df = pd.DataFrame({
        "axis": list(range(1, n_axes + 1)),
        "eigenvalue": eigenvalues,
        "pct_inertia": pct_inertia,
        "cum_pct": np.cumsum(pct_inertia),
    })

    return {
        "coa_coords": coa_df,
        "coa_codon_coords": codon_df,
        "coa_inertia": inertia_df,
    }


# ─── S-value (RSCU distance to ribosomal proteins) ──────────────────────────


def compute_s_value(
    rscu_gene_df: pd.DataFrame,
    rscu_rp: dict[str, float] | None,
    metric: str = "euclidean",
    rscu_ace: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute per-gene RSCU distance to a reference codon usage profile.

    When *rscu_ace* is provided (ACE consensus RSCU), it is used as the
    reference instead of ribosomal proteins.  The ACE consensus is
    genome-specific and composition-independent, making S-values more
    comparable across organisms with different GC content.

    Genes with low S-values have codon usage similar to the reference
    (i.e. adapted toward translational optimization).

    Args:
        rscu_gene_df: Per-gene RSCU table.
        rscu_rp: Concatenated RSCU for ribosomal proteins (fallback reference).
        metric: 'euclidean' or 'chi_squared'.
        rscu_ace: ACE consensus RSCU dict (preferred reference when available).

    Returns:
        DataFrame with gene, S_value, S_reference columns.
    """
    # Prefer ACE consensus over RP-based reference
    ref = rscu_ace if rscu_ace is not None else rscu_rp
    ref_label = "ace" if rscu_ace is not None else "rp"

    if ref is None:
        return pd.DataFrame(columns=["gene", "S_value", "S_reference"])

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns and c in ref]
    if not rscu_cols:
        return pd.DataFrame(columns=["gene", "S_value", "S_reference"])

    ref_vec = np.array([ref[c] for c in rscu_cols])
    gene_mat = rscu_gene_df[rscu_cols].values

    if metric == "chi_squared":
        # Chi-squared distance: sum((obs_i - exp_i)^2 / exp_i)
        denom = np.where(ref_vec == 0, 1e-10, ref_vec)
        dists = np.sqrt(np.sum((gene_mat - ref_vec[np.newaxis, :]) ** 2 / denom[np.newaxis, :], axis=1))
    else:
        # Euclidean distance
        dists = np.sqrt(np.sum((gene_mat - ref_vec[np.newaxis, :]) ** 2, axis=1))

    return pd.DataFrame({
        "gene": rscu_gene_df["gene"].values,
        "S_value": dists,
        "S_reference": ref_label,
    })


# ─── ENC - ENC' difference ──────────────────────────────────────────────────


def compute_enc_diff(
    enc_df: pd.DataFrame,
    encprime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ENC - ENC' for each gene.

    Genes where ENC << ENC' are under codon selection pressure beyond
    what background nucleotide composition predicts.

    Args:
        enc_df: ENC table with gene, ENC, GC3.
        encprime_df: ENCprime table with gene and a score column.

    Returns:
        DataFrame with gene, ENC, ENCprime, ENC_diff, GC3.
    """
    if enc_df.empty or encprime_df.empty:
        return pd.DataFrame(columns=["gene", "ENC", "ENCprime", "ENC_diff", "GC3"])

    # Identify the ENCprime score column
    score_candidates = [c for c in encprime_df.columns if c not in ("gene", "width")]
    if not score_candidates:
        logger.warning("ENCprime DataFrame has no score column; skipping ENC diff")
        return pd.DataFrame(columns=["gene", "ENC", "ENCprime", "ENC_diff", "GC3"])
    score_col = score_candidates[0]

    merged = enc_df[["gene", "ENC", "GC3"]].merge(
        encprime_df[["gene", score_col]].rename(columns={score_col: "ENCprime"}),
        on="gene",
        how="inner",
    )
    merged["ENC_diff"] = merged["ENC"] - merged["ENCprime"]
    return merged


# ─── Neutrality plot (GC12 vs GC3) ──────────────────────────────────────────


def compute_gc12_gc3(ffn_path: Path, min_length: int = 240) -> pd.DataFrame:
    """Compute per-gene GC content at 1st+2nd vs 3rd codon positions.

    The Sueoka (1988) neutrality test: GC12 vs GC3 regression slope
    indicates relative contribution of mutation (slope ~1) vs selection
    (slope ~0).

    Returns:
        DataFrame with gene, GC1, GC2, GC12, GC3, length.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < min_length:
            continue

        gc1 = gc2 = gc3 = total1 = total2 = total3 = 0
        for i in range(0, len(seq) - 2, 3):
            for pos_offset, counters in [(0, "1"), (1, "2"), (2, "3")]:
                base = seq[i + pos_offset]
                if base in "ACGT":
                    if counters == "1":
                        total1 += 1
                        if base in "GC":
                            gc1 += 1
                    elif counters == "2":
                        total2 += 1
                        if base in "GC":
                            gc2 += 1
                    else:
                        total3 += 1
                        if base in "GC":
                            gc3 += 1

        if total1 > 0 and total2 > 0 and total3 > 0:
            gc1_frac = gc1 / total1
            gc2_frac = gc2 / total2
            gc12_frac = (gc1 + gc2) / (total1 + total2)
            gc3_frac = gc3 / total3
            rows.append({
                "gene": rec.id,
                "GC1": gc1_frac,
                "GC2": gc2_frac,
                "GC12": gc12_frac,
                "GC3": gc3_frac,
                "length": len(seq),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gene", "GC1", "GC2", "GC12", "GC3", "length"]
    )


# ─── PR2 (Parity Rule 2) plot ───────────────────────────────────────────────


def compute_pr2(ffn_path: Path, min_length: int = 240) -> pd.DataFrame:
    """Compute PR2 bias statistics: A3/(A3+T3) vs G3/(G3+C3) per gene.

    Under no strand bias and no selection, both ratios cluster at 0.5.
    Deviations reveal replication-associated mutational asymmetry and
    translational selection.

    Returns:
        DataFrame with gene, A3_ratio (A3/(A3+T3)), G3_ratio (G3/(G3+C3)), length.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < min_length:
            continue

        base3_counts = Counter()
        for i in range(2, len(seq), 3):
            base = seq[i]
            if base in "ACGT":
                base3_counts[base] += 1

        a3 = base3_counts.get("A", 0)
        t3 = base3_counts.get("T", 0)
        g3 = base3_counts.get("G", 0)
        c3 = base3_counts.get("C", 0)

        at_total = a3 + t3
        gc_total = g3 + c3

        if at_total > 0 and gc_total > 0:
            rows.append({
                "gene": rec.id,
                "A3_ratio": a3 / at_total,
                "G3_ratio": g3 / gc_total,
                "length": len(seq),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gene", "A3_ratio", "G3_ratio", "length"]
    )


# ─── Delta RSCU between expression tiers ─────────────────────────────────────


def compute_delta_rscu(
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    class_col: str = "expression_class",
) -> pd.DataFrame:
    """Compute delta RSCU (high-expression genes vs genome average) per codon.

    Positive delta = codon favored in highly expressed genes.
    Negative delta = codon avoided in highly expressed genes.

    The default *class_col* is ``expression_class``, which resolves to
    ACE-MELP tiers when ACE has run, or RP-MELP tiers otherwise.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        expr_df: Expression table with gene and *_class columns.
        class_col: Which classification column to use.

    Returns:
        DataFrame with codon, amino_acid, genome_avg_rscu, high_expr_rscu,
        delta_rscu columns.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols or class_col not in expr_df.columns:
        return pd.DataFrame()

    # Merge RSCU with expression tiers
    merged = rscu_gene_df.merge(
        expr_df[["gene", class_col]], on="gene", how="inner"
    )

    genome_avg = merged[rscu_cols].mean()
    high_mask = merged[class_col] == "high"

    if high_mask.sum() < 3:
        logger.warning("Fewer than 3 high-expression genes for delta RSCU")
        return pd.DataFrame()

    high_avg = merged.loc[high_mask, rscu_cols].mean()

    rows = []
    for col in rscu_cols:
        parts = col.split("-")
        aa = parts[0]
        codon = parts[-1]
        rows.append({
            "codon_col": col,
            "codon": codon,
            "amino_acid": aa,
            "genome_avg_rscu": round(genome_avg[col], 4),
            "high_expr_rscu": round(high_avg[col], 4),
            "delta_rscu": round(high_avg[col] - genome_avg[col], 4),
        })

    return pd.DataFrame(rows)


# ─── tRNA gene count vs codon frequency ──────────────────────────────────────


def extract_trna_counts_from_gff(gff_path: Path) -> pd.DataFrame:
    """Extract tRNA gene counts per anticodon from a GFF3 file (e.g. Prokka output).

    Parses tRNA features and extracts anticodon information. Maps anticodon
    back to the codon it decodes (reverse complement + RNA conversion).

    Args:
        gff_path: Path to GFF3 annotation file.

    Returns:
        DataFrame with anticodon, codon (RNA), amino_acid, tRNA_copy_number.
    """
    anticodon_pattern = re.compile(r"(?:anticodon|product)=.*?([ACGT]{3})", re.IGNORECASE)
    product_aa_pattern = re.compile(r"product=tRNA-(\w{3})", re.IGNORECASE)

    trna_counts: Counter = Counter()
    trna_aa: dict[str, str] = {}

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            # GFF3 has ## directives and FASTA section
            if line.startswith(">"):
                break

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type != "tRNA":
                continue

            attrs = parts[8]

            # Try to extract anticodon
            ac_match = anticodon_pattern.search(attrs)
            if ac_match:
                anticodon_dna = ac_match.group(1).upper()
                trna_counts[anticodon_dna] += 1

                # Try to get amino acid from product field
                aa_match = product_aa_pattern.search(attrs)
                if aa_match:
                    trna_aa[anticodon_dna] = aa_match.group(1)

    if not trna_counts:
        # Fallback: try aragorn-style annotation where product=tRNA-Xxx(anticodon)
        trna_counts, trna_aa = _parse_trna_fallback(gff_path)

    if not trna_counts:
        return pd.DataFrame(columns=["anticodon", "codon", "amino_acid", "tRNA_copy_number"])

    rows = []
    for anticodon_dna, count in trna_counts.items():
        # Reverse complement the anticodon to get the codon it decodes
        codon_dna = _reverse_complement(anticodon_dna)
        codon_rna = dna_to_rna(codon_dna)
        aa = trna_aa.get(anticodon_dna, CODON_TABLE_11.get(codon_rna, "?"))
        rows.append({
            "anticodon": anticodon_dna,
            "codon": codon_rna,
            "amino_acid": aa,
            "tRNA_copy_number": count,
        })

    return pd.DataFrame(rows)


def _parse_trna_fallback(gff_path: Path) -> tuple[Counter, dict[str, str]]:
    """Fallback tRNA parser for various annotation formats."""
    # Match patterns like tRNA-Ala(TGC) or tRNA-Ala(tgc)
    pattern = re.compile(r"tRNA-(\w{3,4})\(([ACGTacgt]{3})\)")
    counts: Counter = Counter()
    aa_map: dict[str, str] = {}

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith(("#", ">")):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            if parts[2] not in ("tRNA", "gene"):
                continue
            m = pattern.search(parts[8])
            if m:
                aa_abbr = m.group(1)
                anticodon = m.group(2).upper()
                counts[anticodon] += 1
                aa_map[anticodon] = aa_abbr

    return counts, aa_map


def compute_trna_codon_correlation(
    trna_df: pd.DataFrame,
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
    class_col: str = "expression_class",
) -> pd.DataFrame:
    """Correlate tRNA gene copy number with codon frequency in highly expressed genes.

    Strong positive correlation = evidence of co-adaptation between tRNA pools
    and codon usage in highly expressed genes.

    Args:
        trna_df: tRNA count table (from extract_trna_counts_from_gff).
        rscu_gene_df: Per-gene RSCU table.
        expr_df: Expression table for subsetting high-expression genes.
        class_col: Expression classification column (defaults to
            ``expression_class``, which is ACE-MELP when available).

    Returns:
        DataFrame with codon, amino_acid, tRNA_copy_number, codon_freq_all,
        codon_freq_high, codon_freq_low.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if trna_df.empty or not rscu_cols:
        return pd.DataFrame()

    # Build codon -> tRNA copy number mapping (RNA codons)
    trna_map = dict(zip(trna_df["codon"], trna_df["tRNA_copy_number"]))

    # Genome-wide average RSCU per codon
    genome_avg = rscu_gene_df[rscu_cols].mean()

    # High/low expression averages
    high_avg = low_avg = None
    if expr_df is not None and class_col in expr_df.columns:
        merged = rscu_gene_df.merge(expr_df[["gene", class_col]], on="gene", how="inner")
        high_mask = merged[class_col] == "high"
        low_mask = merged[class_col] == "low"
        if high_mask.sum() >= 3:
            high_avg = merged.loc[high_mask, rscu_cols].mean()
        if low_mask.sum() >= 3:
            low_avg = merged.loc[low_mask, rscu_cols].mean()

    rows = []
    for col in rscu_cols:
        codon = RSCU_COL_TO_CODON[col]
        aa = col.split("-")[0]
        trna_n = trna_map.get(codon, 0)

        row = {
            "codon_col": col,
            "codon": codon,
            "amino_acid": aa,
            "tRNA_copy_number": trna_n,
            "rscu_all_genes": round(genome_avg[col], 4),
        }
        if high_avg is not None:
            row["rscu_high_expr"] = round(high_avg[col], 4)
        if low_avg is not None:
            row["rscu_low_expr"] = round(low_avg[col], 4)
        rows.append(row)

    return pd.DataFrame(rows)


# ─── COG category enrichment in expression tiers ────────────────────────────


def compute_cog_enrichment(
    cog_result_tsv: Path,
    expr_df: pd.DataFrame,
    class_col: str = "expression_class",
) -> pd.DataFrame:
    """Test COG functional category enrichment in expression tiers.

    Uses Fisher's exact test per COG functional category to ask whether
    that category is enriched among high-expression (or low-expression) genes.

    Args:
        cog_result_tsv: COGclassifier result.tsv.
        expr_df: Expression table with gene and *_class columns.
        class_col: Expression classification column to use (defaults to
            ``expression_class``, which is ACE-MELP when available).

    Returns:
        DataFrame with COG_category, description, tier, n_tier, n_background,
        odds_ratio, p_value, fdr columns.
    """
    if not cog_result_tsv.exists() or expr_df.empty or class_col not in expr_df.columns:
        return pd.DataFrame()

    cog_df = pd.read_csv(cog_result_tsv, sep="\t")

    # Find COG functional category column
    func_col = None
    for candidate in ["FUNCTIONAL_CATEGORY", "functional_category", "COG_category",
                       "cog_category", "LETTER", "letter"]:
        if candidate in cog_df.columns:
            func_col = candidate
            break
    # Fallback: look for single-letter columns
    if func_col is None:
        for col in cog_df.columns:
            vals = cog_df[col].dropna().astype(str)
            if vals.str.match(r"^[A-Z]$").mean() > 0.5:
                func_col = col
                break

    if func_col is None:
        logger.warning("Cannot find COG functional category column")
        return pd.DataFrame()

    # Find query/gene ID column
    query_col = None
    for candidate in ["QUERY_ID", "query_id", "Query", "query", "gene_id", "protein_id"]:
        if candidate in cog_df.columns:
            query_col = candidate
            break
    if query_col is None:
        query_col = cog_df.columns[0]

    # COG category descriptions
    cog_descriptions = {
        "J": "Translation, ribosomal structure and biogenesis",
        "A": "RNA processing and modification",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "B": "Chromatin structure and dynamics",
        "D": "Cell cycle control, cell division, chromosome partitioning",
        "Y": "Nuclear structure",
        "V": "Defense mechanisms",
        "T": "Signal transduction mechanisms",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "Z": "Cytoskeleton",
        "W": "Extracellular structures",
        "U": "Intracellular trafficking, secretion, and vesicular transport",
        "O": "Posttranslational modification, protein turnover, chaperones",
        "X": "Mobilome: prophages, transposons",
        "C": "Energy production and conversion",
        "G": "Carbohydrate transport and metabolism",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis, transport and catabolism",
        "R": "General function prediction only",
        "S": "Function unknown",
    }

    # Merge COG with expression
    # Some genes may have multi-letter categories; explode them
    cog_slim = cog_df[[query_col, func_col]].copy()
    cog_slim.columns = ["gene", "cog_cat"]
    cog_slim["cog_cat"] = cog_slim["cog_cat"].astype(str)

    # Explode multi-letter categories (e.g. "KL" -> "K", "L")
    exploded = []
    for _, row in cog_slim.iterrows():
        for letter in row["cog_cat"]:
            if letter.isalpha():
                exploded.append({"gene": row["gene"], "cog_cat": letter})
    cog_exploded = pd.DataFrame(exploded)

    merged = cog_exploded.merge(expr_df[["gene", class_col]], on="gene", how="inner")

    # All genes with COG annotations
    all_categories = sorted(merged["cog_cat"].unique())
    total_genes = len(merged["gene"].unique())

    rows = []
    for tier in ["high", "low"]:
        tier_genes = set(merged.loc[merged[class_col] == tier, "gene"].unique())
        non_tier_genes = set(merged["gene"].unique()) - tier_genes

        if len(tier_genes) < 3:
            continue

        for cat in all_categories:
            cat_genes = set(merged.loc[merged["cog_cat"] == cat, "gene"].unique())
            # 2x2 table:
            # tier & cat, tier & !cat
            # !tier & cat, !tier & !cat
            a = len(cat_genes & tier_genes)
            b = len(tier_genes - cat_genes)
            c = len(cat_genes - tier_genes)
            d = len(non_tier_genes - cat_genes)

            if a + c == 0:
                continue

            odds_ratio, pval = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")

            rows.append({
                "COG_category": cat,
                "description": cog_descriptions.get(cat, ""),
                "tier": tier,
                "n_in_tier": a,
                "n_total": a + c,
                "pct_in_tier": round(100 * a / len(tier_genes), 1) if len(tier_genes) > 0 else 0,
                "pct_in_genome": round(100 * (a + c) / total_genes, 1) if total_genes > 0 else 0,
                "odds_ratio": round(odds_ratio, 3),
                "p_value": pval,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # BH FDR correction
    n_tests = len(result)
    if n_tests > 0:
        pvals = result["p_value"].values
        fdr = np.zeros(n_tests)
        for i in range(n_tests):
            fdr[i] = pvals[i] * n_tests / (i + 1)
        for i in range(n_tests - 2, -1, -1):
            fdr[i] = min(fdr[i], fdr[i + 1])
        fdr = np.minimum(fdr, 1.0)
        result["fdr"] = fdr
        result["significant"] = result["fdr"] <= 0.05

    return result


# ─── Gene length vs codon bias ──────────────────────────────────────────────


def compute_gene_length_bias(
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge gene length with codon bias metrics for scatter analysis.

    Short genes tend to have weaker codon bias due to stochastic effects.

    Args:
        enc_df: ENC table with gene, length, ENC, GC3.
        expr_df: Optional expression table to add CAI/MELP/Fop scores.

    Returns:
        DataFrame with gene, length, ENC, GC3, and optionally CAI/MELP/Fop.
    """
    if enc_df.empty:
        return pd.DataFrame()

    result = enc_df[["gene", "length", "ENC", "GC3"]].copy()

    if expr_df is not None and "gene" in expr_df.columns:
        score_cols = [c for c in ["MELP", "CAI", "Fop"] if c in expr_df.columns]
        if score_cols:
            result = result.merge(
                expr_df[["gene"] + score_cols], on="gene", how="left"
            )

    return result


# ─── Helper functions ────────────────────────────────────────────────────────


def _reverse_complement(dna: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "g": "c", "c": "g"}
    return "".join(comp.get(b, b) for b in reversed(dna))


# ─── Master runner ───────────────────────────────────────────────────────────


def run_advanced_analyses(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    rscu_rp: dict[str, float] | None = None,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    gff_path: Path | None = None,
    cog_result_tsv: Path | None = None,
    rscu_ace: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run all advanced codon usage analyses.

    Args:
        ffn_path: Path to CDS nucleotide FASTA.
        output_dir: Base output directory.
        sample_id: Sample identifier.
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC table with gene, ENC, GC3.
        rscu_rp: Concatenated RSCU for ribosomal proteins.
        expr_df: Expression table with gene, scores, *_class columns.
        encprime_df: ENCprime table.
        gff_path: GFF3 annotation file (for tRNA extraction).
        cog_result_tsv: COGclassifier result.tsv (for COG enrichment).
        rscu_ace: ACE consensus RSCU dict. When provided, S-value uses
            this genome-specific reference instead of ribosomal proteins.

    Returns:
        Dict of output DataFrames and file paths.
    """
    adv_dir = output_dir / "advanced"
    adv_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame | Path] = {}

    # 1. COA on RSCU
    logger.info("Computing Correspondence Analysis on RSCU for %s", sample_id)
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df)
    if coa_results:
        for key, df in coa_results.items():
            out_path = adv_dir / f"{sample_id}_{key}.tsv"
            df.to_csv(out_path, sep="\t", index=False)
            outputs[key] = df
            outputs[f"{key}_path"] = out_path

    # 2. S-value (prefer ACE consensus reference when available)
    ref_label = "ACE consensus" if rscu_ace is not None else "ribosomal proteins"
    logger.info("Computing S-value (RSCU distance to %s) for %s", ref_label, sample_id)
    s_val_df = compute_s_value(rscu_gene_df, rscu_rp, rscu_ace=rscu_ace)
    if not s_val_df.empty:
        out_path = adv_dir / f"{sample_id}_s_value.tsv"
        s_val_df.to_csv(out_path, sep="\t", index=False)
        outputs["s_value"] = s_val_df
        outputs["s_value_path"] = out_path

    # 3. ENC - ENC' difference
    if encprime_df is not None and not encprime_df.empty:
        logger.info("Computing ENC - ENC' difference for %s", sample_id)
        enc_diff_df = compute_enc_diff(enc_df, encprime_df)
        if not enc_diff_df.empty:
            out_path = adv_dir / f"{sample_id}_enc_diff.tsv"
            enc_diff_df.to_csv(out_path, sep="\t", index=False)
            outputs["enc_diff"] = enc_diff_df
            outputs["enc_diff_path"] = out_path

    # 4. Neutrality plot data (GC12 vs GC3)
    logger.info("Computing GC12 vs GC3 (neutrality) for %s", sample_id)
    gc12_gc3_df = compute_gc12_gc3(ffn_path)
    if not gc12_gc3_df.empty:
        out_path = adv_dir / f"{sample_id}_gc12_gc3.tsv"
        gc12_gc3_df.to_csv(out_path, sep="\t", index=False)
        outputs["gc12_gc3"] = gc12_gc3_df
        outputs["gc12_gc3_path"] = out_path

    # 5. PR2 data
    logger.info("Computing PR2 bias statistics for %s", sample_id)
    pr2_df = compute_pr2(ffn_path)
    if not pr2_df.empty:
        out_path = adv_dir / f"{sample_id}_pr2.tsv"
        pr2_df.to_csv(out_path, sep="\t", index=False)
        outputs["pr2"] = pr2_df
        outputs["pr2_path"] = out_path

    # 6. Delta RSCU
    if expr_df is not None:
        logger.info("Computing delta RSCU (high-expression vs genome avg) for %s", sample_id)
        delta_class_cols = ["expression_class",
                           "CAI_class", "MELP_class", "Fop_class"]
        for class_col in delta_class_cols:
            if class_col in expr_df.columns:
                delta_df = compute_delta_rscu(rscu_gene_df, expr_df, class_col)
                if not delta_df.empty:
                    metric = class_col.replace("_class", "")
                    out_path = adv_dir / f"{sample_id}_delta_rscu_{metric}.tsv"
                    delta_df.to_csv(out_path, sep="\t", index=False)
                    outputs[f"delta_rscu_{metric}"] = delta_df
                    outputs[f"delta_rscu_{metric}_path"] = out_path

    # 7. tRNA gene copy number correlation
    if gff_path is not None and gff_path.exists():
        logger.info("Extracting tRNA gene counts from GFF for %s", sample_id)
        trna_df = extract_trna_counts_from_gff(gff_path)
        if not trna_df.empty:
            trna_out = adv_dir / f"{sample_id}_trna_counts.tsv"
            trna_df.to_csv(trna_out, sep="\t", index=False)
            outputs["trna_counts"] = trna_df
            outputs["trna_counts_path"] = trna_out

            trna_corr_df = compute_trna_codon_correlation(
                trna_df, rscu_gene_df, expr_df,
            )
            if not trna_corr_df.empty:
                corr_out = adv_dir / f"{sample_id}_trna_codon_correlation.tsv"
                trna_corr_df.to_csv(corr_out, sep="\t", index=False)
                outputs["trna_codon_correlation"] = trna_corr_df
                outputs["trna_codon_correlation_path"] = corr_out
        else:
            logger.info("No tRNA features found in GFF for %s", sample_id)
    else:
        logger.info("No GFF file available for %s; skipping tRNA analysis", sample_id)

    # 8. COG category enrichment
    if cog_result_tsv is not None and cog_result_tsv.exists() and expr_df is not None:
        logger.info("Computing COG category enrichment by expression tier for %s", sample_id)
        cog_enrich = compute_cog_enrichment(cog_result_tsv, expr_df)
        if not cog_enrich.empty:
            out_path = adv_dir / f"{sample_id}_cog_enrichment.tsv"
            cog_enrich.to_csv(out_path, sep="\t", index=False)
            outputs["cog_enrichment"] = cog_enrich
            outputs["cog_enrichment_path"] = out_path

    # 9. Gene length vs codon bias
    logger.info("Computing gene length vs codon bias for %s", sample_id)
    length_bias_df = compute_gene_length_bias(enc_df, expr_df)
    if not length_bias_df.empty:
        out_path = adv_dir / f"{sample_id}_gene_length_bias.tsv"
        length_bias_df.to_csv(out_path, sep="\t", index=False)
        outputs["gene_length_bias"] = length_bias_df
        outputs["gene_length_bias_path"] = out_path

    n_analyses = sum(1 for k in outputs if not k.endswith("_path"))
    logger.info("Advanced analyses complete for %s: %d datasets produced", sample_id, n_analyses)
    return outputs
