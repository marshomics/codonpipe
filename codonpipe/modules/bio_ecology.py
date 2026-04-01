"""Biological and ecological analyses for codon usage in CodonPipe.

Includes HGT detection via Mahalanobis distance, growth rate prediction from
ribosomal protein adaptation, gRodon2 growth rate prediction, translational
selection quantification, phage/mobile element detection, strand asymmetry
analysis, and operon-level codon coadaptation.
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
from scipy.spatial.distance import euclidean
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from codonpipe.modules.rscu import count_codons, compute_rscu_from_counts
from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    CODON_TABLE_11,
    MIN_GENE_LENGTH,
    RSCU_COLUMN_NAMES,
    RSCU_COL_TO_CODON,
    SENSE_CODONS,
    dna_to_rna,
)
from codonpipe.utils.io import find_gene_id_column
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")

# Expression metric preference order — MELP accounts for both codon
# adaptation and amino acid usage, making it more robust than CAI
# (especially in high-GC genomes where CAI conflates mutational bias
# with translational selection).
_EXPR_METRIC_PREFERENCE = ("MELP", "CAI", "Fop")


def _resolve_expression_metric(expr_df: pd.DataFrame) -> str | None:
    """Return the best available expression metric column in *expr_df*.

    Preference: MELP > CAI > Fop.  Returns ``None`` if none are present.
    """
    for col in _EXPR_METRIC_PREFERENCE:
        if col in expr_df.columns and expr_df[col].notna().sum() > 0:
            return col
    return None

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


# ── CU-annotated GFF output ─────────────────────────────────────────────

# Map dual-anchor categories to biologically meaningful CU class labels.
_DUAL_CAT_TO_CU_CLASS = {
    "both":      "streamlined_CU",
    "rp_only":   "streamlined_CU",
    "dens_only": "genome_standard_CU",
    "neither":   "HGT_candidate",
}


def annotate_gff_with_cu_class(
    gff_path: Path,
    dual_anchor_df: pd.DataFrame,
    output_path: Path,
    hgt_df: pd.DataFrame | None = None,
) -> Path:
    """Write a new GFF3 file with a ``codon_usage_class`` attribute on each CDS.

    Reads the original GFF line-by-line.  For each CDS feature, resolves
    the gene ID against the dual-anchor DataFrame and appends a
    ``codon_usage_class=`` attribute with one of three values:

      - **streamlined_CU** — genes in the RP cluster (``both`` or
        ``rp_only`` dual-anchor categories).  These share the
        translationally optimised codon usage of ribosomal proteins.
      - **genome_standard_CU** — genes in the density cluster only
        (``dens_only``).  Typical compositional baseline for this genome.
      - **HGT_candidate** — genes outside both clusters (``neither``).
        Codon usage diverges from both the translational-selection
        anchor and the genome-wide baseline.

    When *hgt_df* is supplied (from :func:`detect_hgt_candidates`), a
    secondary ``hgt_flag_combined`` attribute is added for CDS features
    that were also flagged by the RSCU-space Mahalanobis HGT test.

    Non-CDS lines (headers, ``gene`` features, ``##FASTA`` blocks, etc.)
    are passed through unmodified.

    Args:
        gff_path: Path to the original Prokka/PGAP GFF3 file.
        dual_anchor_df: DataFrame with ``gene`` and ``dual_category``
            columns from dual-anchor clustering.
        output_path: Destination path for the annotated GFF3.
        hgt_df: Optional HGT candidates DataFrame with ``gene`` and
            ``hgt_flag_combined`` columns.

    Returns:
        The *output_path* that was written.
    """
    # Build lookup: gene_id → CU class
    cu_map: dict[str, str] = {}
    for _, row in dual_anchor_df.iterrows():
        gid = str(row["gene"])
        cat = str(row.get("dual_category", ""))
        cu_map[gid] = _DUAL_CAT_TO_CU_CLASS.get(cat, "unclassified")

    # Optional HGT flag lookup
    hgt_flag_map: dict[str, str] = {}
    if hgt_df is not None and not hgt_df.empty and "hgt_flag_combined" in hgt_df.columns:
        for _, row in hgt_df.iterrows():
            gid = str(row["gene"])
            flag = str(row["hgt_flag_combined"])
            if flag.lower() in ("true", "1", "yes"):
                hgt_flag_map[gid] = "True"

    known_genes = set(cu_map.keys())
    n_annotated = 0
    n_cds_total = 0

    with open(gff_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            # Pass headers and FASTA blocks through unchanged
            if line.startswith("#") or line.startswith(">"):
                fout.write(line)
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                fout.write(line)
                continue

            n_cds_total += 1
            attrs = parts[8]

            # Resolve gene ID
            candidates = _extract_gene_ids_from_attrs(attrs)
            resolved = None
            for cid in candidates:
                if cid in known_genes:
                    resolved = cid
                    break

            if resolved is not None:
                cu_class = cu_map[resolved]
                attrs = attrs.rstrip(";") + f";codon_usage_class={cu_class}"
                if resolved in hgt_flag_map:
                    attrs += ";hgt_mahal_flag=True"
                n_annotated += 1
            else:
                attrs = attrs.rstrip(";") + ";codon_usage_class=unclassified"

            parts[8] = attrs
            fout.write("\t".join(parts) + "\n")

    logger.info(
        "CU-annotated GFF written to %s: %d/%d CDS features classified",
        output_path.name, n_annotated, n_cds_total,
    )
    return output_path


def detect_hgt_candidates(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
    sensitivity: str = "moderate",
    expected_hgt_frac: float = 0.05,
    reference_rscu: dict[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Detect horizontal gene transfer (HGT) candidates via Mahalanobis distance.

    Computes Mahalanobis distance of each gene's RSCU vector from a reference
    centroid using robust covariance estimation (LedoitWolf shrinkage). Also
    computes per-gene GC3 deviation from genome mean and a combined HGT flag.

    When ``reference_rscu`` is provided (e.g. Mahalanobis cluster RSCU), distances are
    measured from that reference instead of the genome mean, making HGT calls
    relative to the translationally optimised gene pool.

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        enc_df: ENC table with GC3 column.
        expr_df: Optional expression table with gene and expression_class columns.
        sensitivity: HGT detection stringency. One of "conservative" (FDR < 0.001),
            "moderate" (FDR < 0.01), or "sensitive" (FDR < 0.05).
        expected_hgt_frac: Expected fraction of horizontally transferred genes
            (default 0.05). Used to compute an adaptive Mahalanobis distance
            threshold targeting this fraction.
        reference_rscu: Optional reference RSCU centroid (dict or Series keyed by
            RSCU column names). When provided, Mahalanobis distance is computed
            from this reference instead of the genome mean. Typically the Mahalanobis
            cluster RSCU from concatenated pooling.

    Returns:
        DataFrame with columns: gene, mahalanobis_dist, gc3_deviation, p_value,
        p_adjusted, hgt_flag_fdr, hgt_flag_adaptive, gc3_outlier, hgt_flag_combined,
        and optionally expression_class.
    """
    if rscu_gene_df.empty or enc_df.empty:
        logger.warning("Empty RSCU or ENC DataFrame; skipping HGT detection")
        return pd.DataFrame(columns=[
            "gene", "mahalanobis_dist", "gc3_deviation", "p_value", "p_adjusted",
            "hgt_flag_fdr", "hgt_flag_adaptive", "gc3_outlier", "hgt_flag_combined",
        ])

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        logger.warning("No RSCU columns found; skipping HGT detection")
        return pd.DataFrame(columns=[
            "gene", "mahalanobis_dist", "gc3_deviation", "p_value", "p_adjusted",
            "hgt_flag_fdr", "hgt_flag_adaptive", "gc3_outlier", "hgt_flag_combined",
        ])

    # Extract RSCU matrix
    X = rscu_gene_df[rscu_cols].values.copy()
    genes = rscu_gene_df["gene"].values
    n_genes, n_features = X.shape

    # Remove genes with missing RSCU values rather than imputing zeros
    n_before = len(X)
    valid_mask = ~(np.isnan(X) | np.isinf(X)).any(axis=1)
    n_after = valid_mask.sum()
    if n_before > n_after:
        logger.info("HGT detection: dropped %d/%d genes with NaN/inf RSCU values", n_before - n_after, n_before)
    if n_after < 10:
        logger.warning("Too few genes with complete RSCU data (%d) for HGT detection", n_after)
        return pd.DataFrame(columns=[
            "gene", "mahalanobis_dist", "gc3_deviation", "p_value", "p_adjusted",
            "hgt_flag_fdr", "hgt_flag_adaptive", "gc3_outlier", "hgt_flag_combined",
        ])
    X = X[valid_mask]
    genes = genes[valid_mask]
    n_genes = X.shape[0]

    # If fewer genes than features, use PCA reduction first
    pca_applied = False
    if n_genes < n_features:
        logger.info("Fewer genes (%d) than RSCU features (%d); applying PCA reduction",
                    n_genes, n_features)
        pca = PCA(n_components=min(n_genes - 1, n_features))
        X = pca.fit_transform(X)
        n_features = X.shape[1]
        pca_applied = True

    # Compute robust covariance via LedoitWolf
    try:
        lw = LedoitWolf()
        cov, _ = lw.fit(X).covariance_, lw.shrinkage_
        cov_inv = np.linalg.pinv(cov)
    except Exception as e:
        logger.warning("Covariance estimation failed (%s); using standard covariance", e)
        cov = np.cov(X.T)
        cov_inv = np.linalg.pinv(cov)

    # Reference centroid: Mahalanobis cluster RSCU if provided, else genome mean
    if reference_rscu is not None:
        ref_series = pd.Series(reference_rscu) if isinstance(reference_rscu, dict) else reference_rscu
        # Align to the same RSCU column order used in X
        ref_vals = np.array([ref_series.get(c, np.nan) for c in rscu_cols])
        if np.any(np.isnan(ref_vals)):
            # Fill missing codons with genome mean for those positions
            genome_mean = X.mean(axis=0) if not pca_applied else rscu_df[rscu_cols].values.mean(axis=0)
            nan_mask = np.isnan(ref_vals)
            ref_vals[nan_mask] = genome_mean[nan_mask]
        # If PCA was applied, project reference into PCA space
        if pca_applied:
            n_expected = pca.n_features_in_
            if ref_vals.shape[0] != n_expected:
                logger.warning(
                    "HGT detection: reference RSCU has %d features but PCA expects %d; "
                    "falling back to genome mean centroid",
                    ref_vals.shape[0], n_expected,
                )
                reference_rscu = None  # fall through to genome-mean below
            else:
                ref_vals = pca.transform(ref_vals.reshape(1, -1)).flatten()
        if reference_rscu is not None:
            mean_rscu = ref_vals
            logger.info("HGT detection: using provided reference RSCU (e.g. Mahalanobis cluster) as centroid")

    if reference_rscu is None:
        mean_rscu = X.mean(axis=0)
        logger.info("HGT detection: using genome mean RSCU as centroid")

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
    # Note: p-values are approximate because LedoitWolf shrinkage introduces bias
    # in the covariance estimate, invalidating the exact chi-squared assumption.
    logger.info("HGT p-values are approximate (LedoitWolf shrinkage covariance used)")

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

    # Benjamini-Hochberg FDR correction across all genes
    p_adjusted = benjamini_hochberg(p_values)

    # ── FDR-based flag (sensitivity parameter) ──────────────────────────
    fdr_thresholds = {"conservative": 0.001, "moderate": 0.01, "sensitive": 0.05}
    fdr_alpha = fdr_thresholds.get(sensitivity, 0.01)
    hgt_flags_fdr = p_adjusted < fdr_alpha
    logger.info("HGT FDR threshold (sensitivity=%s): alpha=%.4f, flagged %d/%d genes",
                sensitivity, fdr_alpha, hgt_flags_fdr.sum(), len(genes))

    # ── Adaptive Mahalanobis threshold ────────────────────────────────
    # Find the distance threshold that flags approximately expected_hgt_frac of genes.
    # Use quantile of the Mahalanobis distribution: top expected_hgt_frac are outliers.
    adaptive_cutoff = np.quantile(mahal_dists, 1.0 - expected_hgt_frac)
    hgt_flags_adaptive = mahal_dists > adaptive_cutoff
    logger.info("HGT adaptive threshold (expected_frac=%.2f): cutoff=%.3f, flagged %d/%d genes",
                expected_hgt_frac, adaptive_cutoff, hgt_flags_adaptive.sum(), len(genes))

    # ── GC3 outlier flag (#7) ─────────────────────────────────────────
    # Flag genes whose GC3 deviates > 2 SD from the genome mean
    gc3_valid = gc3_deviations[~np.isnan(gc3_deviations)]
    if len(gc3_valid) > 5:
        gc3_sd = np.std(gc3_valid)
        gc3_outlier = np.abs(gc3_deviations) > 2 * gc3_sd
    else:
        gc3_outlier = np.full(len(gc3_deviations), False)
    gc3_outlier = np.where(np.isnan(gc3_deviations), False, gc3_outlier)
    logger.info("GC3 outlier flag: %d/%d genes beyond 2 SD",
                gc3_outlier.sum(), len(genes))

    # ── Combined flag: FDR OR GC3 outlier ─────────────────────────────
    hgt_combined = hgt_flags_fdr | gc3_outlier

    # Build result DataFrame
    result = pd.DataFrame({
        "gene": genes,
        "mahalanobis_dist": mahal_dists,
        "gc3_deviation": gc3_deviations,
        "p_value": p_values,
        "p_adjusted": p_adjusted,
        "hgt_flag_fdr": hgt_flags_fdr,
        "hgt_flag_adaptive": hgt_flags_adaptive,
        "gc3_outlier": gc3_outlier.astype(bool),
        "hgt_flag_combined": hgt_combined,
        "hgt_flag": hgt_combined,  # backward-compatible alias
        "sensitivity": sensitivity,
        "fdr_alpha": fdr_alpha,
        "adaptive_cutoff": adaptive_cutoff,
    })

    # Merge expression class if available
    if expr_df is not None and "gene" in expr_df.columns:
        if "expression_class" in expr_df.columns:
            result = result.merge(
                expr_df[["gene", "expression_class"]], on="gene", how="left"
            )

    logger.info("HGT detection: %d FDR, %d adaptive, %d GC3-outlier, %d combined out of %d genes",
                hgt_flags_fdr.sum(), hgt_flags_adaptive.sum(),
                gc3_outlier.sum(), hgt_combined.sum(), len(genes))

    return result


def predict_growth_rate(
    expr_df: pd.DataFrame,
    rp_ids_file: Path | str | None = None,
) -> dict | None:
    """Predict minimum doubling time from CAI of ribosomal proteins.

    Uses the Vieira-Silva & Rocha (2010) empirical model:
    ``doubling_time = exp(a + b * mean_CAI)``.

    The coefficients (a=7.15, b=-7.38) were calibrated on mean CAI of
    ribosomal proteins across ~200 prokaryotic genomes.  This function
    uses **only** RP genes and **only** CAI to match the original
    calibration conditions.

    Args:
        expr_df: Expression table with gene and CAI column.
        rp_ids_file: File with ribosomal protein gene IDs (one per line).
                     Falls back to heuristic name matching if not provided.

    Returns:
        Dict with mean_metric, predicted_doubling_time_hours, n_reference_genes,
        growth_class, expression_metric, reference_set.
        Returns None if no expression data or reference genes found.
    """
    if expr_df.empty:
        logger.warning("Empty expression data; cannot predict growth rate")
        return None

    # Force CAI: the Vieira-Silva & Rocha coefficients were calibrated on CAI
    metric = "CAI"
    if metric not in expr_df.columns or expr_df[metric].notna().sum() == 0:
        logger.warning("CAI column not available in expression data; "
                       "cannot run Vieira-Silva & Rocha growth rate prediction")
        return None

    # Reference set: ribosomal proteins only
    ref_genes: set[str] = set()
    if rp_ids_file is not None:
        rp_path = Path(rp_ids_file)
        if rp_path.exists():
            with open(rp_path) as f:
                ref_genes = set(line.strip() for line in f if line.strip())
    if not ref_genes:
        # Heuristic fallback: match genes with "ribosomal" or "rp" in name
        gene_names = expr_df["gene"].str.lower()
        ref_genes = set(expr_df.loc[
            gene_names.str.contains("ribosomal|rp", na=False), "gene"
        ])
    ref_label = "ribosomal_proteins"
    logger.info("Growth rate prediction using CAI on ribosomal proteins (%d genes)",
                len(ref_genes))

    if not ref_genes:
        logger.warning("No reference genes identified for growth rate prediction")
        return None

    ref_mask = expr_df["gene"].isin(ref_genes)
    ref_vals_series = expr_df.loc[ref_mask, metric].dropna()

    if len(ref_vals_series) < 3:
        logger.warning("Too few reference genes with %s values (%d); need at least 3",
                       metric, len(ref_vals_series))
        return None

    mean_metric = ref_vals_series.mean()

    # Empirical coefficients from Vieira-Silva & Rocha (2010)
    a = 7.15
    b = -7.38
    predicted_doubling_time = np.exp(a + b * mean_metric)

    # Bootstrap 95% CI on the predicted doubling time
    rng = np.random.default_rng(42)
    n_boot = 1000
    ref_vals = ref_vals_series.values
    boot_dts = np.empty(n_boot)
    for i in range(n_boot):
        boot_sample = rng.choice(ref_vals, size=len(ref_vals), replace=True)
        boot_dts[i] = np.exp(a + b * boot_sample.mean())
    ci_lower = float(np.percentile(boot_dts, 2.5))
    ci_upper = float(np.percentile(boot_dts, 97.5))

    # Growth class
    if predicted_doubling_time < 2.0:
        growth_class = "fast"
    elif predicted_doubling_time <= 8.0:
        growth_class = "moderate"
    else:
        growth_class = "slow"

    return {
        "expression_metric": metric,
        "reference_set": ref_label,
        "mean_metric": float(mean_metric),
        # Backward compatibility aliases
        "mean_metric_rp": float(mean_metric),
        "predicted_doubling_time_hours": float(predicted_doubling_time),
        "ci_lower_hours": ci_lower,
        "ci_upper_hours": ci_upper,
        "n_reference_genes": int(len(ref_vals_series)),
        "n_rp_genes": int(len(ref_vals_series)),  # backward compat
        "growth_class": growth_class,
        "caveat": (
            f"Vieira-Silva & Rocha (2010) model using CAI on "
            f"ribosomal proteins ({len(ref_vals_series)} genes); "
            "coefficients (a=7.15, b=-7.38) calibrated on proteobacteria."
        ),
    }


def quantify_translational_selection(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    ffn_path: Path,
    optimal_percentile: float = 0.75,
) -> dict[str, pd.DataFrame]:
    """Quantify translational selection via optimal codon identification and Fop analysis.

    Uses the best available expression metric (MELP preferred over CAI)
    to stratify genes by expression level.

    Three sub-analyses:
    1. Optimal codon identification per AA (most enriched in high-expression genes)
    2. Fop (frequency of optimal codons) across expression quintiles
    3. Within-gene codon position effects (5', middle, 3')

    Args:
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC table with gene column.
        expr_df: Expression table with gene and at least one of MELP, CAI,
            or Fop columns.
        ffn_path: Path to CDS nucleotide FASTA.

    Returns:
        Dict with keys: "optimal_codons", "fop_gradient", "position_effects".
    """
    empty_result = {
        "optimal_codons": pd.DataFrame(),
        "fop_gradient": pd.DataFrame(),
        "position_effects": pd.DataFrame(),
    }

    if rscu_gene_df.empty or expr_df.empty:
        logger.warning("Empty RSCU or expression data; skipping translational selection")
        return empty_result

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        return empty_result

    metric = _resolve_expression_metric(expr_df)
    if metric is None:
        logger.warning("No expression metric (MELP/CAI/Fop) available; "
                       "skipping translational selection")
        return empty_result
    logger.info("Translational selection analysis using %s", metric)

    # Merge RSCU with expression
    merged = rscu_gene_df.merge(expr_df[["gene", metric]], on="gene", how="inner")
    if merged.empty:
        logger.info("SKIPPED: translational selection (no overlap between RSCU and expression genes)")
        return empty_result

    # --- A. Optimal codon identification ---
    genome_avg_rscu = merged[rscu_cols].mean()
    high_mask = merged[metric] >= merged[metric].quantile(optimal_percentile)
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
                "is_optimal": 1 if delta > 0.05 else 0,
                "expression_metric": metric,
            })
        optimal_df = pd.DataFrame(optimal_rows)

        # Mark the single most enriched codon per AA
        for aa in optimal_df["amino_acid"].unique():
            aa_mask = optimal_df["amino_acid"] == aa
            if aa_mask.sum() > 0:
                max_idx = optimal_df.loc[aa_mask, "delta_rscu"].idxmax()
                optimal_df.loc[max_idx, "is_optimal"] = 2  # Mark as top optimal

    # --- B. Fop gradient across expression quintiles ---
    merged_sorted = merged.sort_values(metric)
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

    # Pre-parse FASTA once: store codon counts, Fop, and raw sequences for all
    # merged genes.  This avoids re-reading the file per quintile and again
    # for position effects (was 5+2 passes, now 1).
    all_gene_ids = set(merged["gene"].values)
    _gene_codon_counts: dict[str, dict[str, int]] = {}
    _gene_fop: dict[str, float] = {}
    _gene_seqs: dict[str, str] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        if rec.id not in all_gene_ids:
            continue
        seq = str(rec.seq)
        _gene_seqs[rec.id] = seq
        if len(seq) < 240:
            continue
        gene_counts = count_codons(seq)
        _gene_codon_counts[rec.id] = gene_counts
        if optimal_codons_set:
            n_opt = sum(gene_counts.get(c, 0) for c in optimal_codons_set
                        if c in SENSE_CODONS)
            n_total = sum(gene_counts.get(c, 0) for c in SENSE_CODONS)
            if n_total > 0:
                _gene_fop[rec.id] = n_opt / n_total

    for q_num, q_genes in quintiles:
        if not optimal_codons_set:
            break

        fop_vals = [_gene_fop[g] for g in q_genes if g in _gene_fop]

        if fop_vals:
            fop_rows.append({
                "quintile": q_num,
                "mean_fop": round(np.mean(fop_vals), 4),
                "std_fop": round(np.std(fop_vals), 4),
                "n_genes": len(fop_vals),
            })

    fop_gradient_df = pd.DataFrame(fop_rows)

    # Continuous Spearman correlation (expression metric vs Fop)
    if optimal_codons_set and not merged.empty:
        metric_lookup = merged.set_index("gene")[metric]
        all_fop_vals = []
        all_metric_vals = []
        for gene_id, fop_val in _gene_fop.items():
            if gene_id in metric_lookup.index:
                all_fop_vals.append(fop_val)
                all_metric_vals.append(metric_lookup[gene_id])

        if len(all_fop_vals) >= 10:
            from scipy.stats import spearmanr
            spearman_rho, spearman_p = spearmanr(all_metric_vals, all_fop_vals)
            fop_gradient_df.attrs["spearman_rho"] = round(spearman_rho, 4)
            fop_gradient_df.attrs["spearman_p"] = spearman_p
            fop_gradient_df.attrs["expression_metric"] = metric
            logger.info("%s-Fop Spearman correlation: rho=%.4f, p=%.2e",
                        metric, spearman_rho, spearman_p)

    # --- C. Within-gene codon position effects ---
    # Reuse cached sequences from initial parse instead of re-reading the FASTA.
    pos_rows = []
    merged_gene_set = set(merged["gene"].values)
    for gene_id, seq in _gene_seqs.items():
        if len(seq) < 300:
            continue
        if gene_id not in merged_gene_set:
            continue

        # Use 20% of gene length for terminal regions, minimum 90 nt, rounded to codon boundary
        seq_upper = seq.upper()
        region_nt = max(90, (len(seq_upper) // 5 // 3) * 3)

        # 5' region
        seq_5p = seq_upper[:region_nt]
        counts_5p = count_codons(seq_5p)
        fop_5p = _compute_fop_from_counts(counts_5p, optimal_df) if not optimal_df.empty else np.nan

        # 3' region
        seq_3p = seq_upper[-region_nt:]
        counts_3p = count_codons(seq_3p)
        fop_3p = _compute_fop_from_counts(counts_3p, optimal_df) if not optimal_df.empty else np.nan

        # Middle region
        mid_start = region_nt
        mid_end = len(seq_upper) - region_nt
        if mid_end > mid_start:
            seq_mid = seq_upper[mid_start:mid_end]
            counts_mid = count_codons(seq_mid)
            fop_mid = _compute_fop_from_counts(counts_mid, optimal_df) if not optimal_df.empty else np.nan
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


def _compute_fop_from_counts(codon_counts: Counter, optimal_df: pd.DataFrame) -> float:
    """Helper: compute Fop (fraction of optimal codons) from raw codon counts."""
    if optimal_df.empty:
        return np.nan
    optimal_codons = set(optimal_df.loc[optimal_df["is_optimal"] > 0, "codon"])
    opt_count = sum(codon_counts.get(c, 0) for c in optimal_codons)
    total_count = sum(codon_counts.get(c, 0) for c in SENSE_CODONS)
    return opt_count / total_count if total_count > 0 else np.nan


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

                query_col = find_gene_id_column(cog_df, fallback_to_first=True)

                # COG category descriptions
                cog_descriptions = {
                    "X": "Mobilome: prophages, transposons",
                    "L": "Replication, recombination and repair",
                }

                # Build a slim COG lookup DataFrame and merge vectorised
                cog_slim = cog_df[[query_col, func_col]].copy()
                cog_slim.columns = ["gene", "cog_cat"]
                cog_slim["cog_cat"] = cog_slim["cog_cat"].astype(str).str.strip()
                cog_slim = cog_slim[cog_slim["cog_cat"].str.len() > 0]
                cog_slim["cog_description"] = cog_slim["cog_cat"].map(
                    lambda c: cog_descriptions.get(c, "Unknown")
                )
                # Keep first COG assignment per gene (avoid duplicates)
                cog_slim = cog_slim.drop_duplicates(subset="gene", keep="first")

                result = result.merge(
                    cog_slim.rename(columns={"cog_cat": "cog_category"}),
                    on="gene", how="left", suffixes=("", "_cog"),
                )
                result["is_mobilome"] = result["is_mobilome"] | (result["cog_category"] == "X")
                result["is_phage_related"] = result["is_phage_related"] | (
                    (result["cog_category"] == "L") & result["hgt_flag"]
                )

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
    if result_df is not None and len(result_df) > 1:
        # Benjamini-Hochberg FDR correction across all per-codon tests
        result_df["p_adjusted"] = np.round(benjamini_hochberg(result_df["p_value"].values), 6)
        result_df["significant"] = result_df["p_adjusted"] < 0.05

        # Check for severe strand imbalance
        n_plus = len(plus_rscu.get(rscu_cols[0], [])) if rscu_cols else 0
        n_minus = len(minus_rscu.get(rscu_cols[0], [])) if rscu_cols else 0
        if n_plus > 0 and n_minus > 0 and max(n_plus, n_minus) / min(n_plus, n_minus) > 3:
            logger.warning("Severe strand imbalance: %d genes on (+) strand vs %d on (-) strand. "
                           "Mann-Whitney results may be unreliable.", n_plus, n_minus)

        logger.info("Strand asymmetry: %d codons analyzed (%d significant, FDR < 0.05), n_plus=%d, n_minus=%d",
                    len(result_df), result_df["significant"].sum(), n_plus, n_minus)
    elif result_df is not None:
        result_df["p_adjusted"] = result_df["p_value"]
        result_df["significant"] = result_df["p_adjusted"] < 0.05
        n_plus = len(plus_rscu.get(rscu_cols[0], [])) if rscu_cols else 0
        n_minus = len(minus_rscu.get(rscu_cols[0], [])) if rscu_cols else 0
        logger.info("Strand asymmetry: %d codons analyzed, n_plus=%d, n_minus=%d",
                    len(result_df), n_plus, n_minus)
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

    # Build RSCU lookup (fill NaN with 0.0 — unobserved family → zero usage
    # for Euclidean distance computation)
    rscu_map = {}
    for _, row in rscu_gene_df.iterrows():
        rscu_map[row["gene"]] = np.nan_to_num(row[rscu_cols].values, nan=0.0)

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
    n_permutations = 1000
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
                query_col = find_gene_id_column(cog_df, fallback_to_first=True)

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
    mahal_cluster_rscu: dict[str, float] | pd.Series | None = None,
    mahal_cluster_gene_ids: list[str] | set[str] | None = None,
    mahal_cluster_ids_path: Path | str | None = None,
    dual_anchor_df: pd.DataFrame | None = None,
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
        mahal_cluster_rscu: Optional Mahalanobis cluster RSCU (dict or Series). When
            provided, HGT detection uses this as the reference centroid
            instead of the genome mean.
        mahal_cluster_gene_ids: Optional gene IDs from the Mahalanobis optimal cluster.
            Used for HGT detection and translational selection analyses.
            Growth rate prediction always uses ribosomal proteins only.
        mahal_cluster_ids_path: Optional path to the Mahalanobis-defined
            optimised gene IDs file (one ID per line).  When provided
            alongside *expr_df* containing MELP-based expression_class,
            gRodon2 uses the optimised set as the background and high-MELP
            genes as the highly-expressed set.
        dual_anchor_df: Optional DataFrame from dual-anchor clustering with
            'gene' and 'dual_category' columns.  When provided, a
            ``dual_anchor_category`` column is merged into the HGT
            candidates table.

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
        hgt_df = detect_hgt_candidates(
            rscu_gene_df, enc_df, expr_df,
            reference_rscu=mahal_cluster_rscu,
        )
        if not hgt_df.empty:
            hgt_df = _annotate_df(hgt_df, ann_map, "gene")
            # Merge dual-anchor category when available
            if dual_anchor_df is not None and not dual_anchor_df.empty:
                _da_map = dict(zip(
                    dual_anchor_df["gene"].astype(str),
                    dual_anchor_df["dual_category"].astype(str),
                ))
                hgt_df["dual_anchor_category"] = (
                    hgt_df["gene"].astype(str).map(_da_map).fillna("unknown")
                )
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
            growth_result = predict_growth_rate(
                expr_df, rp_ids_file,
            )
            if growth_result is not None:
                outputs["growth_rate_prediction"] = growth_result
                # Save as single-row TSV for comparative analysis reader
                growth_df = pd.DataFrame([growth_result])
                out_path = eco_dir / f"{sample_id}_growth_rate_prediction.tsv"
                growth_df.to_csv(out_path, sep="\t", index=False)
                outputs["growth_rate_prediction_path"] = out_path
                logger.info("Growth rate prediction: %.2f h (class: %s)",
                           growth_result["predicted_doubling_time_hours"],
                           growth_result["growth_class"])
            else:
                logger.info("SKIPPED: growth rate prediction (no expression data or ribosomal protein genes)")
        else:
            logger.info("SKIPPED: growth rate prediction (no expression data)")
    except Exception as e:
        logger.warning("Growth rate prediction failed: %s", e)

    # 2b. gRodon2 growth rate prediction (requires R + gRodon2 package)
    #
    # HE set: all ribosomal protein genes.  gRodon2's CUBHE/ConsistencyHE/CPB
    # regression was trained with ribosomal proteins (~55 genes) as the HE
    # anchor (Weissman et al. 2021).  We use the full RP set to match the
    # original training conditions.
    #
    # Background: always the FULL genome CDS set.  gRodon2's CUBHE
    # measures HE bias relative to the genomic average; restricting the
    # background to any cluster collapses the contrast.
    logger.info("Running gRodon2 growth rate prediction for %s", sample_id)
    try:
        from codonpipe.modules.grodon import run_grodon

        grodon_result = run_grodon(
            ffn_path, output_dir, sample_id,
            rp_ids_file=rp_ids_file,
        )

        if grodon_result is not None:
            grodon_path = grodon_result.pop("path", None)
            outputs["grodon2_prediction"] = grodon_result
            if grodon_path is not None:
                outputs["grodon2_prediction_path"] = grodon_path
            logger.info(
                "gRodon2 complete: %.2f h [%.2f, %.2f] (ref: %s, %d HE / %d bg)",
                grodon_result["predicted_doubling_time_hours"],
                grodon_result["lower_ci_hours"],
                grodon_result["upper_ci_hours"],
                grodon_result.get("reference_mode", "unknown"),
                grodon_result.get("n_highly_expressed", 0),
                grodon_result.get("n_background", 0),
            )
        else:
            logger.info("SKIPPED: gRodon2 prediction (R or gRodon2 not available)")
    except Exception as e:
        logger.warning("gRodon2 prediction failed: %s", e)

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
            hgt_df = detect_hgt_candidates(
                rscu_gene_df, enc_df, expr_df,
                reference_rscu=mahal_cluster_rscu,
            )

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
