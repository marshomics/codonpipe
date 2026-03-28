"""Biological and ecological analyses for codon usage in CodonPipe.

Includes HGT detection via Mahalanobis distance, growth rate prediction from
ribosomal protein adaptation, translational selection quantification, phage/mobile
element detection, strand asymmetry analysis, and operon-level codon coadaptation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats
from scipy.spatial.distance import mahalanobis, euclidean
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from codonpipe.modules.rscu import count_codons, compute_rscu_from_counts
from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    CODON_TABLE_11,
    RSCU_COLUMN_NAMES,
    RSCU_COL_TO_CODON,
    SENSE_CODONS,
    dna_to_rna,
)

logger = logging.getLogger("codonpipe")

# Minimum gene length (nucleotides) for analyses
MIN_GENE_LENGTH = 240

# Prefixes commonly prepended to gene IDs in GFF3 ID= attributes
_GFF_ID_PREFIXES = ("cds-", "gene-", "rna-", "mrna-", "CDS:", "gene:", "cds_", "gene_")


def _extract_gene_ids_from_attrs(attrs: str) -> list[str]:
    """Extract candidate gene IDs from a GFF3 attributes string.

    Tries ID=, Name=, locus_tag=, protein_id=, and Parent= fields.
    For each raw value also generates prefix-stripped variants
    (e.g. ``cds-WP_123`` → ``WP_123``).

    Returns a list of candidate IDs, ordered from most to least
    specific (Name/locus_tag first, then ID/Parent).
    """
    candidates: list[str] = []
    # Fields in priority order — Name and locus_tag are the most
    # likely to match FASTA record.id values produced by Prokka / PGAP.
    for field in ("Name", "locus_tag", "protein_id", "ID", "Parent"):
        m = re.search(rf"{field}=([^;\n]+)", attrs)
        if m:
            raw = m.group(1)
            candidates.append(raw)
            # Strip common prefixes
            for pfx in _GFF_ID_PREFIXES:
                if raw.startswith(pfx):
                    candidates.append(raw[len(pfx):])
    return candidates


def _parse_gff_gene_map(
    gff_path: Path,
    known_genes: set[str],
    feature_types: tuple[str, ...] = ("gene", "CDS"),
) -> dict[str, tuple[int, int, str]]:
    """Parse a GFF3 file and return ``{gene_id: (start, end, strand)}``
    where *gene_id* is resolved against *known_genes* (RSCU gene names).

    When the GFF ``ID=`` value doesn't match any known gene, the function
    tries alternative attribute fields and prefix-stripped forms.
    """
    gene_coords: dict[str, tuple[int, int, str]] = {}
    unresolved = 0

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith(">"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] not in feature_types:
                continue
            start, end, strand = int(parts[3]), int(parts[4]), parts[6]
            attrs = parts[8]

            candidates = _extract_gene_ids_from_attrs(attrs)
            resolved = None
            for cid in candidates:
                if cid in known_genes:
                    resolved = cid
                    break
            if resolved is None:
                unresolved += 1
                continue
            # Keep first occurrence (don't overwrite gene with CDS for
            # same locus).
            if resolved not in gene_coords:
                gene_coords[resolved] = (start, end, strand)

    total_features = len(gene_coords) + unresolved
    if unresolved and total_features:
        pct = 100 * unresolved / total_features
        logger.debug(
            "GFF ID resolution: %d/%d features matched known genes "
            "(%.0f%% unresolved)",
            len(gene_coords), total_features, pct,
        )
        if not gene_coords:
            # Log some examples of what we saw vs what we expected
            logger.warning(
                "No GFF features could be matched to RSCU gene names. "
                "Example RSCU genes: %s",
                list(known_genes)[:5],
            )

    return gene_coords


def detect_hgt_candidates(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect horizontal gene transfer (HGT) candidates via Mahalanobis distance.

    Computes Mahalanobis distance of each gene's RSCU vector from the genome mean
    using robust covariance estimation (LedoitWolf shrinkage). Also computes per-gene
    GC3 deviation from genome mean.

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        enc_df: ENC table with GC3 column.
        expr_df: Optional expression table with gene and expression_class columns.

    Returns:
        DataFrame with columns: gene, mahalanobis_dist, gc3_deviation, p_value,
        hgt_flag, and optionally expression_class.
    """
    if rscu_gene_df.empty or enc_df.empty:
        logger.warning("Empty RSCU or ENC DataFrame; skipping HGT detection")
        return pd.DataFrame(columns=[
            "gene", "mahalanobis_dist", "gc3_deviation", "p_value", "hgt_flag"
        ])

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        logger.warning("No RSCU columns found; skipping HGT detection")
        return pd.DataFrame(columns=[
            "gene", "mahalanobis_dist", "gc3_deviation", "p_value", "hgt_flag"
        ])

    # Extract RSCU matrix
    X = rscu_gene_df[rscu_cols].values.copy()
    genes = rscu_gene_df["gene"].values
    n_genes, n_features = X.shape

    # Handle NaN/inf
    X = np.where(np.isnan(X) | np.isinf(X), 0, X)

    # If fewer genes than features, use PCA reduction first
    if n_genes < n_features:
        logger.info("Fewer genes (%d) than RSCU features (%d); applying PCA reduction",
                    n_genes, n_features)
        pca = PCA(n_components=min(n_genes - 1, n_features))
        X = pca.fit_transform(X)
        n_features = X.shape[1]

    # Compute robust covariance via LedoitWolf
    try:
        lw = LedoitWolf()
        cov, _ = lw.fit(X).covariance_, lw.shrinkage_
        cov_inv = np.linalg.pinv(cov)
    except Exception as e:
        logger.warning("Covariance estimation failed (%s); using standard covariance", e)
        cov = np.cov(X.T)
        cov_inv = np.linalg.pinv(cov)

    # Genome mean RSCU
    mean_rscu = X.mean(axis=0)

    # Compute Mahalanobis distance for each gene
    mahal_dists = []
    for i, x in enumerate(X):
        diff = x - mean_rscu
        try:
            d = np.sqrt(diff @ cov_inv @ diff.T)
        except (np.linalg.LinAlgError, ValueError):
            d = np.linalg.norm(diff)
        mahal_dists.append(d)

    mahal_dists = np.array(mahal_dists)

    # Chi-squared p-value from Mahalanobis distance (df = n_features)
    p_values = 1 - stats.chi2.cdf(mahal_dists ** 2, df=n_features)

    # GC3 deviation from genome mean
    merged_gc3 = enc_df[["gene", "GC3"]].copy()
    merged_gc3.set_index("gene", inplace=True)
    gc3_mean = enc_df["GC3"].mean()
    gc3_deviations = []
    for gene in genes:
        if gene in merged_gc3.index:
            gc3_deviations.append(merged_gc3.loc[gene, "GC3"] - gc3_mean)
        else:
            gc3_deviations.append(np.nan)
    gc3_deviations = np.array(gc3_deviations)

    # Flag putative HGT: p < 0.001
    hgt_flags = p_values < 0.001

    # Build result DataFrame
    result = pd.DataFrame({
        "gene": genes,
        "mahalanobis_dist": mahal_dists,
        "gc3_deviation": gc3_deviations,
        "p_value": p_values,
        "hgt_flag": hgt_flags,
    })

    # Merge expression class if available
    if expr_df is not None and "gene" in expr_df.columns:
        if "expression_class" in expr_df.columns:
            result = result.merge(
                expr_df[["gene", "expression_class"]], on="gene", how="left"
            )

    logger.info("HGT detection: flagged %d/%d genes as putative HGT",
                hgt_flags.sum(), len(genes))

    return result


def predict_growth_rate(
    expr_df: pd.DataFrame,
    rp_ids_file: Path | str | None = None,
) -> dict | None:
    """Predict minimum doubling time from ribosomal protein codon adaptation.

    Based on Vieira-Silva & Rocha (2010): predicted doubling time = exp(a + b * mean_CAI_rp)
    where a=7.15 and b=-7.38 (empirical coefficients from their paper, result in hours).

    Args:
        expr_df: Expression table with gene and CAI column.
        rp_ids_file: File with ribosomal protein gene IDs (one per line).
                     If None, uses genes with "ribosomal" or "rp" in name.

    Returns:
        Dict with mean_cai_rp, predicted_doubling_time_hours, n_rp_genes, growth_class.
        Returns None if no CAI data or RP genes found.
    """
    if expr_df.empty or "CAI" not in expr_df.columns:
        logger.warning("No CAI column in expression data; cannot predict growth rate")
        return None

    # Identify RP genes
    rp_genes = set()
    if rp_ids_file is not None:
        rp_path = Path(rp_ids_file)
        if rp_path.exists():
            with open(rp_path) as f:
                rp_genes = set(line.strip() for line in f if line.strip())
    else:
        # Heuristic: match genes with "ribosomal" or "rp" in name
        gene_names = expr_df["gene"].str.lower()
        rp_genes = set(expr_df.loc[
            gene_names.str.contains("ribosomal|rp", na=False), "gene"
        ])

    if not rp_genes:
        logger.warning("No ribosomal protein genes identified")
        return None

    # Extract CAI for RP genes
    rp_mask = expr_df["gene"].isin(rp_genes)
    rp_cai = expr_df.loc[rp_mask, "CAI"].dropna()

    if len(rp_cai) < 1:
        logger.warning("No CAI values for ribosomal proteins")
        return None

    mean_cai_rp = rp_cai.mean()

    # Empirical coefficients from Vieira-Silva & Rocha (2010)
    a = 7.15
    b = -7.38
    predicted_doubling_time = np.exp(a + b * mean_cai_rp)

    # Growth class
    if predicted_doubling_time < 2.0:
        growth_class = "fast"
    elif predicted_doubling_time <= 8.0:
        growth_class = "moderate"
    else:
        growth_class = "slow"

    return {
        "mean_cai_rp": float(mean_cai_rp),
        "predicted_doubling_time_hours": float(predicted_doubling_time),
        "n_rp_genes": int(len(rp_cai)),
        "growth_class": growth_class,
    }


def quantify_translational_selection(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    ffn_path: Path,
) -> dict[str, pd.DataFrame]:
    """Quantify translational selection via optimal codon identification and Fop analysis.

    Three sub-analyses:
    1. Optimal codon identification per AA (most enriched in high-expression genes)
    2. Fop (frequency of optimal codons) across expression quintiles
    3. Within-gene codon position effects (5', middle, 3')

    Args:
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC table with gene column.
        expr_df: Expression table with gene and CAI (or MELP/Fop) column.
        ffn_path: Path to CDS nucleotide FASTA.

    Returns:
        Dict with keys: "optimal_codons", "fop_gradient", "position_effects".
    """
    if rscu_gene_df.empty or expr_df.empty:
        logger.warning("Empty RSCU or expression data; skipping translational selection")
        return {
            "optimal_codons": pd.DataFrame(),
            "fop_gradient": pd.DataFrame(),
            "position_effects": pd.DataFrame(),
        }

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        return {
            "optimal_codons": pd.DataFrame(),
            "fop_gradient": pd.DataFrame(),
            "position_effects": pd.DataFrame(),
        }

    # Merge RSCU with expression
    merged = rscu_gene_df.merge(expr_df[["gene", "CAI"]], on="gene", how="inner")
    if merged.empty:
        logger.info("SKIPPED: translational selection (no overlap between RSCU and expression genes)")
        return {
            "optimal_codons": pd.DataFrame(),
            "fop_gradient": pd.DataFrame(),
            "position_effects": pd.DataFrame(),
        }

    # --- A. Optimal codon identification ---
    genome_avg_rscu = merged[rscu_cols].mean()
    high_mask = merged["CAI"] >= merged["CAI"].quantile(0.75)
    if high_mask.sum() < 3:
        logger.info("SKIPPED: optimal codon identification (fewer than 3 high-expression genes)")
        optimal_df = pd.DataFrame()
    else:
        high_avg_rscu = merged.loc[high_mask, rscu_cols].mean()
        optimal_rows = []
        for col in rscu_cols:
            parts = col.split("-")
            aa = parts[0]
            codon = parts[-1]
            delta = high_avg_rscu[col] - genome_avg_rscu[col]
            optimal_rows.append({
                "amino_acid": aa,
                "codon": codon,
                "codon_col": col,
                "delta_rscu": round(delta, 4),
                "genome_avg_rscu": round(genome_avg_rscu[col], 4),
                "high_expr_rscu": round(high_avg_rscu[col], 4),
                "is_optimal": 1 if delta > 0 else 0,
            })
        optimal_df = pd.DataFrame(optimal_rows)

        # Mark the single most enriched codon per AA
        for aa in optimal_df["amino_acid"].unique():
            aa_mask = optimal_df["amino_acid"] == aa
            if aa_mask.sum() > 0:
                max_idx = optimal_df.loc[aa_mask, "delta_rscu"].idxmax()
                optimal_df.loc[max_idx, "is_optimal"] = 2  # Mark as top optimal

    # --- B. Fop gradient across expression quintiles ---
    merged_sorted = merged.sort_values("CAI")
    n_genes = len(merged_sorted)
    q_size = n_genes // 5
    quintiles = []
    for q in range(5):
        start_idx = q * q_size
        end_idx = start_idx + q_size if q < 4 else n_genes
        q_genes = merged_sorted.iloc[start_idx:end_idx]["gene"].values
        quintiles.append((q + 1, set(q_genes)))

    fop_rows = []
    # Build set of optimal codons (RNA triplets)
    if not optimal_df.empty:
        optimal_codons_set = set(
            optimal_df.loc[optimal_df["is_optimal"] > 0, "codon"]
        )
    else:
        optimal_codons_set = set()

    for q_num, q_genes in quintiles:
        if not optimal_codons_set:
            break

        # Compute true Fop per gene from actual codon counts
        fop_vals = []
        for rec in SeqIO.parse(str(ffn_path), "fasta"):
            if rec.id not in q_genes:
                continue
            seq = str(rec.seq)
            if len(seq) < 240:
                continue
            gene_counts = count_codons(seq)
            n_opt = sum(gene_counts.get(c, 0) for c in optimal_codons_set
                        if c in SENSE_CODONS)
            n_total = sum(gene_counts.get(c, 0) for c in SENSE_CODONS)
            if n_total > 0:
                fop_vals.append(n_opt / n_total)

        if fop_vals:
            fop_rows.append({
                "quintile": q_num,
                "mean_fop": round(np.mean(fop_vals), 4),
                "std_fop": round(np.std(fop_vals), 4),
                "n_genes": len(fop_vals),
            })

    fop_gradient_df = pd.DataFrame(fop_rows)

    # --- C. Within-gene codon position effects ---
    pos_rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < 300:
            continue

        gene_id = rec.id
        if gene_id not in merged["gene"].values:
            continue

        # Split into regions (30 codons = 90 nt at start/end)
        region_nt = 90
        seq_upper = seq.upper()

        # 5' region
        seq_5p = seq_upper[:region_nt]
        counts_5p = count_codons(seq_5p)
        rscu_5p = compute_rscu_from_counts(counts_5p)
        fop_5p = _compute_fop_from_rscu(rscu_5p, optimal_df) if not optimal_df.empty else np.nan

        # 3' region
        seq_3p = seq_upper[-region_nt:]
        counts_3p = count_codons(seq_3p)
        rscu_3p = compute_rscu_from_counts(counts_3p)
        fop_3p = _compute_fop_from_rscu(rscu_3p, optimal_df) if not optimal_df.empty else np.nan

        # Middle region
        mid_start = region_nt
        mid_end = len(seq_upper) - region_nt
        if mid_end > mid_start:
            seq_mid = seq_upper[mid_start:mid_end]
            counts_mid = count_codons(seq_mid)
            rscu_mid = compute_rscu_from_counts(counts_mid)
            fop_mid = _compute_fop_from_rscu(rscu_mid, optimal_df) if not optimal_df.empty else np.nan
        else:
            fop_mid = np.nan

        pos_rows.append({
            "gene": gene_id,
            "fop_5prime": round(fop_5p, 4) if not np.isnan(fop_5p) else np.nan,
            "fop_middle": round(fop_mid, 4) if not np.isnan(fop_mid) else np.nan,
            "fop_3prime": round(fop_3p, 4) if not np.isnan(fop_3p) else np.nan,
            "length": len(seq),
        })

    position_effects_df = pd.DataFrame(pos_rows)

    logger.info("Translational selection: identified %d optimal codons, %d genes with position data",
                len(optimal_df), len(position_effects_df))

    return {
        "optimal_codons": optimal_df,
        "fop_gradient": fop_gradient_df,
        "position_effects": position_effects_df,
    }


def _compute_fop_from_rscu(rscu_dict: dict[str, float], optimal_df: pd.DataFrame) -> float:
    """Helper: compute Fop from RSCU dict and optimal codon table."""
    if optimal_df.empty:
        return np.nan
    optimal_codons = set(optimal_df.loc[optimal_df["is_optimal"] > 0, "codon"])
    total = 0
    opt_sum = 0
    for col_name, rscu_val in rscu_dict.items():
        if col_name in RSCU_COL_TO_CODON:
            codon = RSCU_COL_TO_CODON[col_name]
            total += rscu_val
            if codon in optimal_codons:
                opt_sum += rscu_val
    return opt_sum / total if total > 0 else np.nan


def detect_phage_mobile_elements(
    hgt_df: pd.DataFrame,
    cog_result_tsv: Path | str | None = None,
    kofam_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect phage and mobile element genes via COG/KOFam annotation of HGT candidates.

    Flags genes with COG category "X" (Mobilome) or "L" + HGT signal.
    If KOFam available, also flags transposases and phage-related KOs.

    Args:
        hgt_df: HGT candidate DataFrame (from detect_hgt_candidates).
        cog_result_tsv: Path to COGclassifier result.tsv.
        kofam_df: Optional KOFam annotation DataFrame with gene and KO columns.

    Returns:
        DataFrame with columns: gene, mahalanobis_dist, gc3_deviation, hgt_flag,
        cog_category, cog_description, is_mobilome, is_phage_related, (ko if kofam_df given).
    """
    if hgt_df.empty:
        logger.info("SKIPPED: phage/mobile element detection (no HGT candidates)")
        return pd.DataFrame()

    result = hgt_df[["gene", "mahalanobis_dist", "gc3_deviation", "hgt_flag"]].copy()
    result["cog_category"] = None
    result["cog_description"] = None
    result["is_mobilome"] = False
    result["is_phage_related"] = False

    # COG annotations
    if cog_result_tsv is not None:
        cog_path = Path(cog_result_tsv)
        if cog_path.exists():
            try:
                cog_df = pd.read_csv(cog_path, sep="\t")

                # Find COG category and query columns
                func_col = None
                for candidate in ["FUNCTIONAL_CATEGORY", "functional_category", "COG_category",
                                   "cog_category", "LETTER", "letter"]:
                    if candidate in cog_df.columns:
                        func_col = candidate
                        break
                if func_col is None:
                    for col in cog_df.columns:
                        vals = cog_df[col].dropna().astype(str)
                        if vals.str.match(r"^[A-Z]$").mean() > 0.5:
                            func_col = col
                            break

                query_col = None
                for candidate in ["QUERY_ID", "query_id", "Query", "query", "gene_id", "protein_id"]:
                    if candidate in cog_df.columns:
                        query_col = candidate
                        break
                if query_col is None:
                    query_col = cog_df.columns[0]

                # COG category descriptions
                cog_descriptions = {
                    "X": "Mobilome: prophages, transposons",
                    "L": "Replication, recombination and repair",
                }

                cog_map = {}
                for _, row in cog_df.iterrows():
                    gene = row[query_col]
                    cat = str(row[func_col]).strip() if func_col else None
                    if cat:
                        cog_map[gene] = (cat, cog_descriptions.get(cat, "Unknown"))

                # Merge into result
                for idx, row in result.iterrows():
                    gene = row["gene"]
                    if gene in cog_map:
                        cat, desc = cog_map[gene]
                        result.at[idx, "cog_category"] = cat
                        result.at[idx, "cog_description"] = desc
                        if cat == "X":
                            result.at[idx, "is_mobilome"] = True
                        elif cat == "L" and row["hgt_flag"]:
                            result.at[idx, "is_phage_related"] = True

            except Exception as e:
                logger.warning("Error reading COG file: %s", e)

    # KOFam phage/transposase annotations
    if kofam_df is not None and not kofam_df.empty:
        if "gene" in kofam_df.columns and "KO" in kofam_df.columns:
            phage_kos = {
                "K07483", "K07497", "K07504", "K07505", "K07506", "K07507",  # Transposases
                "K10999", "K11005", "K11006",  # Phage head proteins
                "K10998",  # Phage integrase
            }
            ko_map = {}
            for _, row in kofam_df.iterrows():
                gene = row["gene"]
                ko = str(row["KO"]).strip()
                if gene not in ko_map:
                    ko_map[gene] = []
                ko_map[gene].append(ko)

            for idx, row in result.iterrows():
                gene = row["gene"]
                if gene in ko_map:
                    kos = ko_map[gene]
                    if any(k in phage_kos for k in kos):
                        result.at[idx, "is_phage_related"] = True
                    result.at[idx, "ko"] = ";".join(kos[:3])  # Top 3 KOs

    logger.info("Phage/mobile element detection: flagged %d/%d genes as mobilome/phage-related",
                result["is_mobilome"].sum() + result["is_phage_related"].sum(), len(result))

    return result


def compute_strand_asymmetry(
    ffn_path: Path,
    gff_path: Path | None = None,
    rscu_gene_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Compute RSCU asymmetry between leading and lagging strands.

    Args:
        ffn_path: Path to CDS nucleotide FASTA.
        gff_path: Path to GFF3 annotation (for strand determination).
        rscu_gene_df: Per-gene RSCU table (optional; used to link to strand info).

    Returns:
        DataFrame with codon, amino_acid, mean_rscu_plus, mean_rscu_minus,
        u_statistic, p_value. Returns None if no strand info available.
    """
    if gff_path is None:
        logger.warning("No GFF path provided; cannot compute strand asymmetry")
        return None

    # Parse GFF to get gene -> strand mapping, resolving IDs against
    # RSCU gene names when available
    known_genes: set[str] = set()
    if rscu_gene_df is not None and not rscu_gene_df.empty:
        known_genes = set(rscu_gene_df["gene"].astype(str))
    # Also include FASTA record IDs so we still get strand info even
    # without an RSCU table
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        known_genes.add(rec.id)

    try:
        gene_coords = _parse_gff_gene_map(gff_path, known_genes)
    except Exception as e:
        logger.warning("Error parsing GFF: %s", e)
        return None

    gene_strand = {gid: strand for gid, (_, _, strand) in gene_coords.items()}

    if not gene_strand:
        logger.warning("No genes with strand info found in GFF")
        return None

    # Compute per-strand RSCU
    rscu_cols = [c for c in RSCU_COLUMN_NAMES]
    plus_rscu = {col: [] for col in rscu_cols}
    minus_rscu = {col: [] for col in rscu_cols}

    if rscu_gene_df is not None:
        for _, row in rscu_gene_df.iterrows():
            gene = row["gene"]
            if gene in gene_strand:
                strand = gene_strand[gene]
                target = plus_rscu if strand == "+" else minus_rscu
                for col in rscu_cols:
                    if col in row.index:
                        target[col].append(row[col])

    # Mann-Whitney U test per codon
    rows = []
    for col in rscu_cols:
        plus_vals = [v for v in plus_rscu[col] if not np.isnan(v)]
        minus_vals = [v for v in minus_rscu[col] if not np.isnan(v)]

        if len(plus_vals) > 1 and len(minus_vals) > 1:
            u_stat, p_val = stats.mannwhitneyu(
                plus_vals, minus_vals, alternative="two-sided"
            )
            codon = RSCU_COL_TO_CODON[col]
            aa = col.split("-")[0]
            rows.append({
                "codon": codon,
                "amino_acid": aa,
                "codon_col": col,
                "mean_rscu_plus": round(np.mean(plus_vals), 4),
                "mean_rscu_minus": round(np.mean(minus_vals), 4),
                "u_statistic": round(u_stat, 2),
                "p_value": round(p_val, 6),
            })

    result_df = pd.DataFrame(rows) if rows else None
    if result_df is not None:
        logger.info("Strand asymmetry: analyzed %d codons", len(result_df))
    return result_df


def compute_operon_codon_coadaptation(
    rscu_gene_df: pd.DataFrame,
    gff_path: Path | None = None,
) -> pd.DataFrame | None:
    """Detect codon coadaptation in operons via RSCU distance between adjacent genes.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        gff_path: Path to GFF3 annotation (for gene order and strand).

    Returns:
        DataFrame with gene1, gene2, strand, distance_bp (intergenic), rscu_distance,
        same_operon_prediction. Returns None if no GFF available.
    """
    if gff_path is None:
        logger.warning("No GFF path provided; cannot compute operon coadaptation")
        return None

    if rscu_gene_df.empty:
        logger.info("SKIPPED: operon coadaptation (empty RSCU data)")
        return None

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        logger.info("SKIPPED: operon coadaptation (no RSCU columns)")
        return None

    # Build RSCU lookup
    rscu_map = {}
    for _, row in rscu_gene_df.iterrows():
        rscu_map[row["gene"]] = row[rscu_cols].values

    # Parse GFF for gene order, resolving IDs against RSCU gene names
    known_genes = set(rscu_map.keys())
    try:
        gene_coords = _parse_gff_gene_map(gff_path, known_genes)
    except Exception as e:
        logger.warning("Error parsing GFF: %s", e)
        return None

    if not gene_coords:
        logger.warning(
            "No GFF features matched RSCU gene names (%d RSCU genes, GFF: %s)",
            len(known_genes), gff_path,
        )
        return None

    # Split by strand and sort by position
    genes_plus = []
    genes_minus = []
    for gid, (start, end, strand) in gene_coords.items():
        if strand == "+":
            genes_plus.append((start, gid))
        else:
            genes_minus.append((start, gid))
    genes_plus.sort()
    genes_minus.sort()

    rows = []
    for gene_list in [genes_plus, genes_minus]:
        for i in range(len(gene_list) - 1):
            g1_id = gene_list[i][1]
            g2_id = gene_list[i + 1][1]

            rscu1 = rscu_map[g1_id]
            rscu2 = rscu_map[g2_id]

            # RSCU Euclidean distance
            rscu_dist = euclidean(rscu1, rscu2)

            # Intergenic distance (bp)
            _, g1_end, strand = gene_coords[g1_id]
            g2_start, _, _ = gene_coords[g2_id]
            intergenic_bp = max(0, g2_start - g1_end)

            rows.append({
                "gene1": g1_id,
                "gene2": g2_id,
                "strand": strand,
                "intergenic_bp": intergenic_bp,
                "rscu_distance": round(rscu_dist, 4),
            })

    if not rows:
        logger.warning("No adjacent gene pairs found")
        return None

    result_df = pd.DataFrame(rows)

    # Permutation baseline: shuffle gene order many times and collect
    # pairwise distances to build a null distribution for comparison.
    rng = np.random.RandomState(42)
    random_genes = list(rscu_map.keys())
    random_dists = []
    n_permutations = 100
    for _ in range(n_permutations):
        rng.shuffle(random_genes)
        for i in range(min(len(random_genes) - 1, len(result_df))):
            g1, g2 = random_genes[i], random_genes[i + 1]
            random_dists.append(euclidean(rscu_map[g1], rscu_map[g2]))

    if random_dists:
        median_random = np.median(random_dists)
        result_df["same_operon_prediction"] = (
            result_df["rscu_distance"] < median_random
        )
        result_df["random_baseline_median"] = median_random
    else:
        result_df["same_operon_prediction"] = False
        result_df["random_baseline_median"] = np.nan

    logger.info("Operon coadaptation: analyzed %d gene pairs", len(result_df))

    return result_df


def _build_gene_annotation_map(
    kofam_df: pd.DataFrame | None = None,
    cog_result_tsv: Path | str | None = None,
) -> dict[str, dict[str, str]]:
    """Build a gene -> annotation dict from KOfam and COG data.

    Returns:
        Dict mapping gene ID to {"KO": ..., "KO_definition": ...,
        "COG_ID": ..., "COG_category": ...}. Missing fields are "".
    """
    ann: dict[str, dict[str, str]] = {}

    # KOfam annotations
    if kofam_df is not None and not kofam_df.empty:
        gene_col = next(
            (c for c in ("gene_name", "gene", "gene_id") if c in kofam_df.columns),
            None,
        )
        if gene_col and "KO" in kofam_df.columns:
            ko_def_col = "KO_definition" if "KO_definition" in kofam_df.columns else None
            for _, row in kofam_df.iterrows():
                gene = str(row[gene_col])
                if gene not in ann:
                    ann[gene] = {"KO": "", "KO_definition": "", "COG_ID": "", "COG_category": ""}
                ann[gene]["KO"] = str(row["KO"]) if pd.notna(row["KO"]) else ""
                if ko_def_col and pd.notna(row.get(ko_def_col)):
                    ann[gene]["KO_definition"] = str(row[ko_def_col])

    # COG annotations
    if cog_result_tsv is not None:
        cog_path = Path(cog_result_tsv)
        if cog_path.exists():
            try:
                cog_df = pd.read_csv(cog_path, sep="\t")

                # Find query column
                query_col = None
                for candidate in ["QUERY_ID", "query_id", "Query", "query",
                                   "gene_id", "protein_id"]:
                    if candidate in cog_df.columns:
                        query_col = candidate
                        break
                if query_col is None:
                    query_col = cog_df.columns[0]

                # Find COG ID column
                cog_id_col = None
                for candidate in ["COG_ID", "COG", "cog_id", "COG_id", "best_hit_cog"]:
                    if candidate in cog_df.columns:
                        cog_id_col = candidate
                        break
                if cog_id_col is None:
                    for col in cog_df.columns:
                        vals = cog_df[col].dropna().astype(str).head(20)
                        if vals.str.match(r"^COG\d+$").any():
                            cog_id_col = col
                            break

                # Find functional category column
                func_col = None
                for candidate in ["FUNCTIONAL_CATEGORY", "functional_category",
                                   "COG_category", "cog_category", "LETTER", "letter"]:
                    if candidate in cog_df.columns:
                        func_col = candidate
                        break
                if func_col is None:
                    for col in cog_df.columns:
                        vals = cog_df[col].dropna().astype(str)
                        if vals.str.match(r"^[A-Z]$").mean() > 0.5:
                            func_col = col
                            break

                for _, row in cog_df.iterrows():
                    gene = str(row[query_col])
                    if gene not in ann:
                        ann[gene] = {"KO": "", "KO_definition": "", "COG_ID": "", "COG_category": ""}
                    if cog_id_col and pd.notna(row.get(cog_id_col)):
                        ann[gene]["COG_ID"] = str(row[cog_id_col])
                    if func_col and pd.notna(row.get(func_col)):
                        ann[gene]["COG_category"] = str(row[func_col])
            except Exception as e:
                logger.warning("Error building COG annotation map: %s", e)

    return ann


def _annotate_df(
    df: pd.DataFrame,
    ann_map: dict[str, dict[str, str]],
    gene_col: str = "gene",
) -> pd.DataFrame:
    """Merge KO/COG annotations into a DataFrame by gene column.

    Adds columns: KO, KO_definition, COG_ID, COG_category.
    For operon tables with gene1/gene2, call once per gene column.
    """
    if df.empty or not ann_map or gene_col not in df.columns:
        return df

    ann_cols = ["KO", "KO_definition", "COG_ID", "COG_category"]

    # If annotating gene1/gene2 (operon table), prefix columns
    suffix = ""
    if gene_col != "gene":
        suffix = f"_{gene_col}"  # e.g., _gene1, _gene2

    for col_name in ann_cols:
        out_col = f"{col_name}{suffix}"
        if out_col not in df.columns:
            df[out_col] = df[gene_col].map(
                lambda g, cn=col_name: ann_map.get(str(g), {}).get(cn, "")
            )

    return df


def run_bio_ecology_analyses(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
    rp_ids_file: Path | str | None = None,
    cog_result_tsv: Path | str | None = None,
    kofam_df: pd.DataFrame | None = None,
    gff_path: Path | None = None,
) -> dict[str, pd.DataFrame | Path | dict]:
    """Run all biological and ecological analyses.

    Orchestrator function that calls all bio-ecology analyses and saves outputs.
    Catches per-analysis exceptions to prevent cascade failures.

    Args:
        ffn_path: Path to CDS nucleotide FASTA.
        output_dir: Base output directory.
        sample_id: Sample identifier.
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC table with GC3 column.
        expr_df: Optional expression table (for growth rate, translational selection).
        rp_ids_file: Optional file with ribosomal protein IDs.
        cog_result_tsv: Optional COG result TSV.
        kofam_df: Optional KOFam annotation DataFrame.
        gff_path: Optional GFF3 annotation file.

    Returns:
        Dict of analysis names -> DataFrames/file paths. Includes nested dicts for
        multi-output analyses (e.g., translational_selection).
    """
    eco_dir = output_dir / "bio_ecology"
    eco_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame | Path | dict] = {}

    # Build gene annotation lookup (KO + COG) for annotating output tables
    ann_map = _build_gene_annotation_map(kofam_df, cog_result_tsv)
    if ann_map:
        logger.info("Built annotation map with %d genes (KO + COG)", len(ann_map))
    else:
        logger.info("SKIPPED: gene annotation map (no KOfam or COG data available)")

    # 1. HGT detection
    logger.info("Detecting HGT candidates for %s", sample_id)
    try:
        hgt_df = detect_hgt_candidates(rscu_gene_df, enc_df, expr_df)
        if not hgt_df.empty:
            hgt_df = _annotate_df(hgt_df, ann_map, "gene")
            out_path = eco_dir / f"{sample_id}_hgt_candidates.tsv"
            hgt_df.to_csv(out_path, sep="\t", index=False)
            outputs["hgt_candidates"] = hgt_df
            outputs["hgt_candidates_path"] = out_path
            logger.info("HGT detection complete: %d candidates", len(hgt_df))
    except Exception as e:
        logger.warning("HGT detection failed: %s", e)

    # 2. Growth rate prediction
    logger.info("Predicting growth rate for %s", sample_id)
    try:
        if expr_df is not None:
            growth_result = predict_growth_rate(expr_df, rp_ids_file)
            if growth_result is not None:
                outputs["growth_rate_prediction"] = growth_result
                logger.info("Growth rate prediction: %.2f h (class: %s)",
                           growth_result["predicted_doubling_time_hours"],
                           growth_result["growth_class"])
            else:
                logger.info("SKIPPED: growth rate prediction (no CAI data or ribosomal protein genes)")
        else:
            logger.info("SKIPPED: growth rate prediction (no expression data)")
    except Exception as e:
        logger.warning("Growth rate prediction failed: %s", e)

    # 3. Translational selection
    logger.info("Quantifying translational selection for %s", sample_id)
    try:
        if expr_df is not None:
            trans_sel = quantify_translational_selection(
                rscu_gene_df, enc_df, expr_df, ffn_path
            )
            trans_sel_outputs = {}
            for key, df in trans_sel.items():
                if not df.empty:
                    # Annotate gene-level tables with KO/COG
                    if "gene" in df.columns:
                        df = _annotate_df(df, ann_map, "gene")
                    out_path = eco_dir / f"{sample_id}_translational_selection_{key}.tsv"
                    df.to_csv(out_path, sep="\t", index=False)
                    trans_sel_outputs[key] = df
                    trans_sel_outputs[f"{key}_path"] = out_path
            if trans_sel_outputs:
                outputs["translational_selection"] = trans_sel_outputs
                logger.info("Translational selection complete")
            else:
                logger.info("SKIPPED: translational selection analysis (all sub-analyses returned empty)")
        else:
            logger.info("SKIPPED: translational selection analysis (no expression data)")
    except Exception as e:
        logger.warning("Translational selection analysis failed: %s", e)

    # 4. Phage/mobile element detection
    logger.info("Detecting phage/mobile elements for %s", sample_id)
    try:
        if "hgt_candidates" in outputs:
            hgt_df = outputs["hgt_candidates"]
        else:
            # Need HGT results first
            hgt_df = detect_hgt_candidates(rscu_gene_df, enc_df, expr_df)

        if not hgt_df.empty:
            phage_df = detect_phage_mobile_elements(hgt_df, cog_result_tsv, kofam_df)
            if not phage_df.empty:
                phage_df = _annotate_df(phage_df, ann_map, "gene")
                out_path = eco_dir / f"{sample_id}_phage_mobile_elements.tsv"
                phage_df.to_csv(out_path, sep="\t", index=False)
                outputs["phage_mobile_elements"] = phage_df
                outputs["phage_mobile_elements_path"] = out_path
                logger.info("Phage/mobile detection complete: %d mobilome, %d phage-related",
                           phage_df["is_mobilome"].sum(), phage_df["is_phage_related"].sum())
            else:
                logger.info("SKIPPED: phage/mobile element detection (no mobilome or phage genes found)")
        else:
            logger.info("SKIPPED: phage/mobile element detection (no HGT candidates)")
    except Exception as e:
        logger.warning("Phage/mobile element detection failed: %s", e)

    # 5. Strand asymmetry
    logger.info("Computing strand asymmetry for %s", sample_id)
    try:
        strand_df = compute_strand_asymmetry(ffn_path, gff_path, rscu_gene_df)
        if strand_df is not None and not strand_df.empty:
            out_path = eco_dir / f"{sample_id}_strand_asymmetry.tsv"
            strand_df.to_csv(out_path, sep="\t", index=False)
            outputs["strand_asymmetry"] = strand_df
            outputs["strand_asymmetry_path"] = out_path
            logger.info("Strand asymmetry complete: %d codons analyzed", len(strand_df))
        elif strand_df is None:
            logger.info("SKIPPED: strand asymmetry analysis (no GFF or no strand data)")
        else:
            logger.info("SKIPPED: strand asymmetry analysis (no significant strand differences found)")
    except Exception as e:
        logger.warning("Strand asymmetry analysis failed: %s", e)

    # 6. Operon coadaptation
    logger.info("Computing operon codon coadaptation for %s", sample_id)
    try:
        operon_df = compute_operon_codon_coadaptation(rscu_gene_df, gff_path)
        if operon_df is not None and not operon_df.empty:
            operon_df = _annotate_df(operon_df, ann_map, "gene1")
            operon_df = _annotate_df(operon_df, ann_map, "gene2")
            out_path = eco_dir / f"{sample_id}_operon_coadaptation.tsv"
            operon_df.to_csv(out_path, sep="\t", index=False)
            outputs["operon_coadaptation"] = operon_df
            outputs["operon_coadaptation_path"] = out_path
            logger.info("Operon coadaptation complete: %d gene pairs", len(operon_df))
        elif operon_df is None:
            logger.info("SKIPPED: operon coadaptation analysis (no GFF or insufficient data)")
        else:
            logger.info("SKIPPED: operon coadaptation analysis (no adjacent gene pairs found)")
    except Exception as e:
        logger.warning("Operon coadaptation analysis failed: %s", e)

    n_analyses = sum(1 for k in outputs if not k.endswith("_path") and k != "translational_selection")
    if "translational_selection" in outputs:
        n_analyses += 1

    logger.info("Bio-ecology analyses complete for %s: %d datasets produced", sample_id, n_analyses)

    return outputs
