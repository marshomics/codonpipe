"""Statistical analysis module for codon usage comparisons (batch mode)."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from codonpipe.utils.codon_tables import AMINO_ACID_FAMILIES, RSCU_COLUMN_NAMES
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")


def pairwise_wilcoxon(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str],
    alpha: float = 0.01,
    correction: str = "fdr_bh",
) -> pd.DataFrame:
    """Perform pairwise Wilcoxon rank-sum tests between groups for each codon.

    Args:
        df: DataFrame with group column and RSCU value columns.
        group_col: Column defining groups (e.g., "geo_category", "phylum").
        value_cols: RSCU column names to test.
        alpha: Significance threshold after correction.
        correction: Multiple testing correction method.
            'fdr_bh' (default): Benjamini-Hochberg FDR correction.
            'bonferroni': Bonferroni correction (more conservative).

    Returns:
        DataFrame with columns:
            codon, amino_acid, group1, group2, statistic, p_value,
            corrected_p_value, significant
    """
    groups = sorted(df[group_col].dropna().unique())
    pairs = list(itertools.combinations(groups, 2))
    results = []

    for col in value_cols:
        for g1, g2 in pairs:
            vals1 = df.loc[df[group_col] == g1, col].dropna()
            vals2 = df.loc[df[group_col] == g2, col].dropna()

            if len(vals1) < 5 or len(vals2) < 5:
                continue

            try:
                stat, p_val = sp_stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            except ValueError:
                continue

            if len(vals1) < 10 or len(vals2) < 10:
                logger.debug("Small sample sizes for %s: n1=%d, n2=%d; interpret with caution",
                            col, len(vals1), len(vals2))

            # Extract amino acid from column name (format: {AA}{digits}-{codon}).
            # Validate the expected format before extracting.
            parts = col.split("-")
            aa = parts[0].rstrip("0123456789") if len(parts) >= 2 and parts[0] else col
            results.append({
                "codon": col,
                "amino_acid": aa,
                "group1": g1,
                "group2": g2,
                "n_group1": len(vals1),
                "n_group2": len(vals2),
                "statistic": stat,
                "p_value": p_val,
            })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Multiple testing correction
    n_tests = len(result_df)
    if correction == "bonferroni":
        result_df["corrected_p_value"] = np.minimum(result_df["p_value"] * n_tests, 1.0)
    elif correction == "fdr_bh":
        result_df["corrected_p_value"] = benjamini_hochberg(result_df["p_value"].values)
    else:
        raise ValueError(f"Unknown correction method: {correction!r}. Use 'fdr_bh' or 'bonferroni'.")
    result_df["significant"] = result_df["corrected_p_value"] < alpha

    return result_df


def per_amino_acid_tests(
    df: pd.DataFrame,
    group_col: str,
    alpha: float = 0.01,
) -> dict[str, pd.DataFrame]:
    """Run pairwise Wilcoxon tests for each amino acid family separately.

    Args:
        df: DataFrame with RSCU values and group column.
        group_col: Grouping column.
        alpha: Significance threshold.

    Returns:
        Dict mapping amino acid name to results DataFrame.
    """
    results = {}
    for aa, cols in AMINO_ACID_FAMILIES.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue
        aa_result = pairwise_wilcoxon(df, group_col, present_cols, alpha)
        if not aa_result.empty:
            results[aa] = aa_result
    return results


def compute_zscore_normalization(
    df: pd.DataFrame,
    rscu_cols: list[str] | None = None,
    method: str = "clr",
) -> pd.DataFrame:
    """Normalize RSCU values for downstream analysis.

    Args:
        df: DataFrame with RSCU columns.
        rscu_cols: Columns to normalize. Defaults to RSCU_COLUMN_NAMES.
        method: Normalization method.
            'clr' (default): Centered log-ratio transform, appropriate for
                compositional data like RSCU values.
            'zscore': Standard z-score normalization. Note: RSCU data are bounded
                and compositional, so z-scoring may introduce artifacts. Provided
                for backward compatibility.

    Returns:
        Copy of df with normalized RSCU values.
    """
    if rscu_cols is None:
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]

    result = df.copy()

    if method == "clr":
        # Centered log-ratio transform: log(x / geometric_mean(x)) per row.
        # Replace zeros/NaN with a small pseudocount before the log transform
        # to avoid -inf values while minimally perturbing the ratios.
        pseudocount = 1e-6
        mat = result[rscu_cols].values.astype(float)
        mat = np.where(np.isnan(mat) | (mat == 0), pseudocount, mat)
        log_mat = np.log(mat)
        geo_means = np.exp(log_mat.mean(axis=1, keepdims=True))
        result[rscu_cols] = np.log(mat / geo_means)
    elif method == "zscore":
        for col in rscu_cols:
            vals = result[col]
            mean_val = vals.mean()
            std_val = vals.std()
            if std_val > 0:
                result[col] = (vals - mean_val) / std_val
            else:
                result[col] = np.nan  # Flag zero-variance columns as NaN, not 0
                logger.warning("Zero variance in column '%s'; set to NaN", col)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'clr' or 'zscore'.")

    return result


def run_batch_statistics(
    combined_rscu_df: pd.DataFrame,
    output_dir: Path,
    group_col: str = "sample_id",
    metadata_cols: list[str] | None = None,
) -> dict[str, Path]:
    """Run statistical analyses on combined RSCU data from batch mode.

    Args:
        combined_rscu_df: Combined RSCU data from all samples.
        output_dir: Directory for statistical output files.
        group_col: Column to group by for comparisons.
        metadata_cols: Additional metadata columns to use as grouping variables.

    Returns:
        Dict of output file paths.
    """
    stats_dir = output_dir / "batch_rscu_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Z-score normalized table
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in combined_rscu_df.columns]
    zscored = compute_zscore_normalization(combined_rscu_df, rscu_cols)
    zscore_path = stats_dir / "rscu_zscored.tsv"
    zscored.to_csv(zscore_path, sep="\t", index=False)
    outputs["zscore"] = zscore_path

    # Run pairwise tests for each metadata grouping variable
    test_cols = metadata_cols or []
    for col in test_cols:
        if col not in combined_rscu_df.columns:
            continue
        n_groups = combined_rscu_df[col].nunique()
        if n_groups < 2 or n_groups > 50:
            logger.info(
                "Skipping pairwise tests for '%s' (%d groups — need 2-50)",
                col, n_groups,
            )
            continue

        logger.info("Running pairwise Wilcoxon tests grouped by '%s'", col)
        aa_results = per_amino_acid_tests(combined_rscu_df, col)

        for aa, result_df in aa_results.items():
            out_path = stats_dir / f"{col}_{aa}_wilcoxon.tsv"
            result_df.to_csv(out_path, sep="\t", index=False)
            outputs[f"{col}_{aa}_wilcoxon"] = out_path

        # Combined results
        if aa_results:
            all_results = pd.concat(aa_results.values(), ignore_index=True)
            combined_path = stats_dir / f"{col}_all_wilcoxon.tsv"
            all_results.to_csv(combined_path, sep="\t", index=False)
            outputs[f"{col}_all_wilcoxon"] = combined_path

    return outputs
