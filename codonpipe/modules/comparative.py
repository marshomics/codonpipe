"""Condition-aware comparative analyses for batch mode.

Collects per-sample summary metrics from individual pipeline outputs
and performs within-condition and between-condition statistical comparisons
across every analysis domain: RSCU, expression, growth rate, HGT, translational
selection, strand asymmetry, operon coadaptation, GC content, and ENC.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform

from codonpipe.utils.codon_tables import (
    RSCU_COLUMN_NAMES,
    COL_MELP, COL_CAI, COL_FOP, EXPRESSION_METRICS,
    COL_EXPRESSION_CLASS,
)
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# 1. Per-sample metric collection
# ---------------------------------------------------------------------------

def collect_sample_metrics(
    sample_outputs: dict[str, dict[str, Path]],
    batch_df: pd.DataFrame,
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Build a sample × metric table from individual pipeline outputs.

    Reads per-sample TSV files and extracts genome-level summary values for
    every analysis domain.  The resulting wide-format DataFrame has one row
    per sample and columns for every metric.

    Args:
        sample_outputs: ``{sample_id: {output_key: Path}}``.
        batch_df: Original batch table (used for condition and metadata).
        condition_col: Column in *batch_df* designating the experimental
            condition.  Copied into the output table.

    Returns:
        DataFrame with columns: sample_id, condition (if provided), plus
        all collected metrics.
    """
    rows: list[dict] = []

    for sid, paths in sample_outputs.items():
        row: dict = {"sample_id": sid}

        # Condition label
        if condition_col and condition_col in batch_df.columns:
            match = batch_df.loc[batch_df["sample_id"] == sid, condition_col]
            if not match.empty:
                row[condition_col] = match.iloc[0]

        # --- RSCU median (genome-level) ---
        _read_rscu_median(paths, row)

        # --- Expression scores (genome-level medians) ---
        _read_expression_summary(paths, row)

        # --- ENC / GC3 ---
        _read_enc_summary(paths, row)

        # --- Growth rate prediction ---
        _read_growth_rate(paths, row)

        # --- gRodon2 growth rate prediction ---
        _read_grodon2(paths, row)

        # --- HGT prevalence ---
        _read_hgt_summary(paths, row)

        # --- Strand asymmetry summary ---
        _read_strand_asymmetry(paths, row)

        # --- Operon coadaptation summary ---
        _read_operon_summary(paths, row)

        # --- GC content (neutrality plot data) ---
        _read_gc_content(paths, row)

        # --- S-value (adaptation to ribosomal reference) ---
        _read_s_value(paths, row)

        # --- Translational selection ---
        _read_translational_selection(paths, row)

        # --- ENCprime ---
        _read_enc_prime(paths, row)

        # --- MILC ---
        _read_milc(paths, row)

        # --- CBI ---
        _read_cbi(paths, row)

        # --- Ribosomal protein RSCU profile ---
        _read_ribosomal_rscu(paths, row)

        # --- High-expression gene RSCU profile ---
        _read_high_expression_rscu(paths, row)

        # --- Mahalanobis RP-cluster RSCU profile ---
        _read_mahal_cluster_rscu(paths, row)

        # --- Mahalanobis clustering summary ---
        _read_mahal_summary(paths, row)

        # --- Pathway enrichment summary ---
        _read_enrichment_summary(paths, row)

        # --- Phage / mobile element prevalence ---
        _read_phage_mobile_summary(paths, row)

        # --- HGT Mahalanobis distance (genome heterogeneity) ---
        _read_hgt_mahalanobis(paths, row)

        rows.append(row)

    return pd.DataFrame(rows)


def _read_rscu_median(paths: dict, row: dict) -> None:
    p = paths.get("rscu_median")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]
            for c in rscu_cols:
                row[c] = df[c].iloc[0] if len(df) > 0 else np.nan
        except Exception as e:
            logger.debug("Failed to read RSCU median: %s", e)


def _read_expression_summary(paths: dict, row: dict) -> None:
    p = paths.get("expression_combined")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            for metric in EXPRESSION_METRICS:
                if metric in df.columns:
                    row[f"median_{metric}"] = df[metric].median()
                    row[f"mean_{metric}"] = df[metric].mean()
        except Exception as e:
            logger.debug("Failed to read expression summary: %s", e)


def _read_enc_summary(paths: dict, row: dict) -> None:
    p = paths.get("enc")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            if "ENC" in df.columns:
                row["median_ENC"] = df["ENC"].median()
                row["mean_ENC"] = df["ENC"].mean()
            gc3_col = next((c for c in ("GC3", "GC3s", "gc3") if c in df.columns), None)
            if gc3_col:
                row["mean_GC3"] = df[gc3_col].mean()
        except Exception as e:
            logger.debug("Failed to read ENC summary: %s", e)


def _read_growth_rate(paths: dict, row: dict) -> None:
    """Read growth rate prediction (stored as a single-row TSV or in bio outputs)."""
    # Try bio ecology output path pattern
    for key in ("bio_growth_rate_prediction_path", "bio_growth_rate_prediction", "growth_rate_prediction"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "predicted_doubling_time_hours" in df.columns:
                    row["doubling_time_hours"] = df["predicted_doubling_time_hours"].iloc[0]
                if "mean_cai_rp" in df.columns:
                    row["mean_cai_rp"] = df["mean_cai_rp"].iloc[0]
                if "growth_class" in df.columns:
                    row["growth_class"] = df["growth_class"].iloc[0]
                return
            except Exception as e:
                logger.debug("Failed to read growth rate (key=%s): %s", key, e)


def _read_grodon2(paths: dict, row: dict) -> None:
    """Read gRodon2 growth rate prediction into the metrics row."""
    for key in ("bio_grodon2_prediction_path", "bio_grodon2_prediction"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "predicted_doubling_time_hours" in df.columns:
                    row["grodon2_doubling_time_hours"] = df["predicted_doubling_time_hours"].iloc[0]
                if "lower_ci_hours" in df.columns:
                    row["grodon2_lower_ci_hours"] = df["lower_ci_hours"].iloc[0]
                if "upper_ci_hours" in df.columns:
                    row["grodon2_upper_ci_hours"] = df["upper_ci_hours"].iloc[0]
                if "CUBHE" in df.columns:
                    row["grodon2_CUBHE"] = df["CUBHE"].iloc[0]
                if "ConsistencyHE" in df.columns:
                    row["grodon2_ConsistencyHE"] = df["ConsistencyHE"].iloc[0]
                if "CPB" in df.columns:
                    row["grodon2_CPB"] = df["CPB"].iloc[0]
                if "growth_class" in df.columns:
                    row["grodon2_growth_class"] = df["growth_class"].iloc[0]
                return
            except Exception as e:
                logger.debug("Failed to read gRodon2 prediction (key=%s): %s", key, e)


def _read_hgt_summary(paths: dict, row: dict) -> None:
    for key in ("bio_hgt_candidates_path", "bio_hgt_candidates", "hgt_candidates"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                row["n_hgt_candidates"] = len(df)
                if "hgt_flag" in df.columns:
                    row["n_hgt_positive"] = int(df["hgt_flag"].sum())
                    row["hgt_fraction"] = df["hgt_flag"].mean()
                elif "is_hgt" in df.columns:
                    row["n_hgt_positive"] = int(df["is_hgt"].sum())
                    row["hgt_fraction"] = df["is_hgt"].mean()
                elif "p_adjusted" in df.columns:
                    n_sig = (df["p_adjusted"] < 0.001).sum()
                    row["n_hgt_positive"] = int(n_sig)
                    row["hgt_fraction"] = n_sig / len(df) if len(df) > 0 else 0
                elif "p_value" in df.columns:
                    n_sig = (df["p_value"] < 0.05).sum()
                    row["n_hgt_positive"] = int(n_sig)
                    row["hgt_fraction"] = n_sig / len(df) if len(df) > 0 else 0
                return
            except Exception as e:
                logger.debug("Failed to read HGT summary (key=%s): %s", key, e)


def _read_strand_asymmetry(paths: dict, row: dict) -> None:
    for key in ("bio_strand_asymmetry_path", "bio_strand_asymmetry", "strand_asymmetry"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                row["n_strand_asym_codons"] = len(df)
                # Use FDR-adjusted p-values if available, fall back to raw
                p_col = "p_adjusted" if "p_adjusted" in df.columns else "p_value"
                if p_col in df.columns:
                    row["n_strand_asym_sig"] = int((df[p_col] < 0.05).sum())
                    row["strand_asym_fraction"] = row["n_strand_asym_sig"] / max(len(df), 1)
                return
            except Exception as e:
                logger.debug("Failed to read strand asymmetry (key=%s): %s", key, e)


def _read_operon_summary(paths: dict, row: dict) -> None:
    for key in ("bio_operon_coadaptation_path", "bio_operon_coadaptation", "operon_coadaptation"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "rscu_distance" in df.columns:
                    row["operon_n_pairs"] = len(df)
                    row["operon_mean_rscu_dist"] = df["rscu_distance"].mean()
                    row["operon_median_rscu_dist"] = df["rscu_distance"].median()
                if "same_operon_prediction" in df.columns:
                    row["operon_predicted_fraction"] = df["same_operon_prediction"].mean()
                return
            except Exception as e:
                logger.debug("Failed to read operon summary (key=%s): %s", key, e)


def _read_gc_content(paths: dict, row: dict) -> None:
    p = paths.get("gc12_gc3")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            gc12_col = next((c for c in ("GC12", "gc12") if c in df.columns), None)
            gc3_col = next((c for c in ("GC3", "gc3") if c in df.columns), None)
            if gc12_col:
                row["mean_GC12"] = df[gc12_col].mean()
            if gc3_col:
                row["mean_GC3_neutrality"] = df[gc3_col].mean()
            # Neutrality regression slope
            if gc12_col and gc3_col:
                x = df[gc3_col].dropna()
                y = df[gc12_col].dropna()
                common = x.index.intersection(y.index)
                if len(common) > 10:
                    slope, _, r, p_val, _ = sp_stats.linregress(x[common], y[common])
                    row["neutrality_slope"] = slope
                    row["neutrality_r2"] = r ** 2
        except Exception as e:
            logger.debug("Failed to read GC content: %s", e)


def _read_s_value(paths: dict, row: dict) -> None:
    p = paths.get("s_value")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            s_col = next((c for c in ("S", "s_value", "S_value") if c in df.columns), None)
            if s_col:
                row["mean_S_value"] = df[s_col].mean()
                row["median_S_value"] = df[s_col].median()
        except Exception as e:
            logger.debug("Failed to read S-value: %s", e)


def _read_translational_selection(paths: dict, row: dict) -> None:
    """Read translational selection metrics from fop_gradient and position_effects."""
    # Read fop_gradient (quintile analysis)
    p_grad = paths.get("bio_trans_sel_fop_gradient_path") or paths.get("bio_trans_sel_fop_gradient")
    if p_grad and Path(p_grad).exists():
        try:
            df = pd.read_csv(p_grad, sep="\t")
            if "quintile" in df.columns and "mean_fop" in df.columns:
                # Linear regression of mean_fop vs quintile number
                valid = df[["quintile", "mean_fop"]].dropna()
                if len(valid) > 2:
                    slope, _, _, _, _ = sp_stats.linregress(
                        valid["quintile"].values, valid["mean_fop"].values
                    )
                    row["fop_gradient_slope"] = slope
        except Exception as e:
            logger.debug("Failed to read translational selection fop_gradient: %s", e)

    # Read position_effects (5prime, middle, 3prime)
    p_pos = paths.get("bio_trans_sel_position_effects_path") or paths.get("bio_trans_sel_position_effects")
    if p_pos and Path(p_pos).exists():
        try:
            df = pd.read_csv(p_pos, sep="\t")
            for pos in ("fop_5prime", "fop_middle", "fop_3prime"):
                if pos in df.columns:
                    row[f"mean_{pos}"] = df[pos].mean()
        except Exception as e:
            logger.debug("Failed to read translational selection position_effects: %s", e)


def _read_enc_prime(paths: dict, row: dict) -> None:
    """Read ENCprime and compute median ENC-ENCprime difference."""
    p = paths.get("encprime")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            encprime_col = next((c for c in ("ENCprime", "encprime") if c in df.columns), None)
            if encprime_col:
                row["median_ENCprime"] = df[encprime_col].median()
                # Compute ENC-ENCprime difference if ENC is available
                if "median_ENC" in row and not np.isnan(row["median_ENC"]):
                    row["median_ENC_diff"] = row["median_ENC"] - row["median_ENCprime"]
        except Exception as e:
            logger.debug("Failed to read ENCprime: %s", e)


def _read_milc(paths: dict, row: dict) -> None:
    """Read MILC metric."""
    p = paths.get("milc")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            milc_col = next((c for c in ("MILC", "milc") if c in df.columns), None)
            if milc_col:
                row["median_MILC"] = df[milc_col].median()
        except Exception as e:
            logger.debug("Failed to read MILC: %s", e)


def _read_cbi(paths: dict, row: dict) -> None:
    """Read CBI (Codon Bias Index) metric."""
    # Try direct CBI output first
    for key in ("cbi", "codon_bias_index"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                cbi_col = next((c for c in ("CBI", "cbi") if c in df.columns), None)
                if cbi_col:
                    row["mean_CBI"] = df[cbi_col].mean()
                    return
            except Exception as e:
                logger.debug("Failed to read CBI (key=%s): %s", key, e)


def _read_ribosomal_rscu(paths: dict, row: dict) -> None:
    """Read ribosomal protein RSCU profile (concatenated)."""
    p = paths.get("rscu_ribosomal")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]
            for c in rscu_cols:
                row[f"rp_{c}"] = df[c].iloc[0] if len(df) > 0 else np.nan
        except Exception as e:
            logger.debug("Failed to read ribosomal RSCU: %s", e)


def _read_high_expression_rscu(paths: dict, row: dict) -> None:
    """Read high-expression gene RSCU from codon table format (long format)."""
    p = paths.get("high_expression_rscu")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            if "codon" in df.columns and "rscu" in df.columns:
                # Convert long format to wide using RSCU_COLUMN_NAMES mapping
                for _, r in df.iterrows():
                    codon = r["codon"]
                    aa = r.get("amino_acid", "")
                    col_name = f"{aa}-{codon}" if aa else codon
                    if col_name in RSCU_COLUMN_NAMES:
                        row[f"he_{col_name}"] = r["rscu"]
        except Exception as e:
            logger.debug("Failed to read high-expression RSCU: %s", e)


def _read_mahal_cluster_rscu(paths: dict, row: dict) -> None:
    """Read Mahalanobis RP-cluster RSCU profile."""
    for key in ("mahal_mahal_cluster_rscu_path", "mahal_cluster_rscu_path"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t", index_col=0)
                for col_name in RSCU_COLUMN_NAMES:
                    if col_name in df.index:
                        row[f"mahal_{col_name}"] = df.loc[col_name, "RSCU"]
                return
            except Exception as e:
                logger.debug("Failed to read Mahalanobis cluster RSCU (key=%s): %s", key, e)

    # Fallback: try codon table format output
    p = paths.get("mahal_cluster_rscu")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]
            for c in rscu_cols:
                row[f"mahal_{c}"] = df[c].iloc[0] if len(df) > 0 else np.nan
        except Exception as e:
            logger.debug("Failed to read Mahalanobis cluster RSCU fallback: %s", e)


def _read_mahal_summary(paths: dict, row: dict) -> None:
    """Read Mahalanobis clustering summary (best_k, cluster size, cosine similarity)."""
    for key in ("mahal_mahal_summary_path", "mahal_summary_path"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if len(df) > 0:
                    r = df.iloc[0]
                    for col in ("best_k", "rp_cluster_size", "rp_genes_in_cluster",
                                "non_rp_in_cluster", "rp_cosine_similarity"):
                        if col in df.columns:
                            row[f"mahal_{col}"] = r[col]
                return
            except Exception as e:
                logger.debug("Failed to read Mahalanobis summary (key=%s): %s", key, e)


def _read_enrichment_summary(paths: dict, row: dict) -> None:
    """Read pathway enrichment results and extract counts of significant pathways."""
    # Standard RP-based enrichment
    for metric in EXPRESSION_METRICS:
        key = f"enrichment_{metric}_high"
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                sig_col = "significant" if "significant" in df.columns else None
                fdr_col = "fdr" if "fdr" in df.columns else None
                if sig_col:
                    row[f"n_enriched_{metric}_high"] = int(df[sig_col].sum())
                    row[f"n_pathways_{metric}_high"] = len(df)
                elif fdr_col:
                    row[f"n_enriched_{metric}_high"] = int((df[fdr_col] < 0.05).sum())
                    row[f"n_pathways_{metric}_high"] = len(df)
            except Exception as e:
                logger.debug("Failed to read enrichment summary (metric=%s): %s", metric, e)



def _read_phage_mobile_summary(paths: dict, row: dict) -> None:
    """Read phage/mobile element detection results."""
    for key in ("bio_phage_mobile_elements_path", "bio_phage_mobile_elements", "phage_mobile_elements"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                row["n_phage_mobile"] = len(df)
                # Count mobilome-flagged genes
                mob_col = next((c for c in ("is_mobilome", "mobilome") if c in df.columns), None)
                if mob_col:
                    row["n_mobilome"] = int(df[mob_col].sum())
                phage_col = next((c for c in ("is_phage", "phage_related") if c in df.columns), None)
                if phage_col:
                    row["n_phage"] = int(df[phage_col].sum())
                return
            except Exception as e:
                logger.debug("Failed to read phage/mobile summary (key=%s): %s", key, e)


def _read_hgt_mahalanobis(paths: dict, row: dict) -> None:
    """Read mean Mahalanobis distance from HGT results as genome heterogeneity metric."""
    for key in ("bio_hgt_candidates_path", "bio_hgt_candidates", "hgt_candidates"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "mahalanobis_dist" in df.columns:
                    vals = df["mahalanobis_dist"].dropna()
                    row["mean_mahalanobis_dist"] = vals.mean()
                    row["median_mahalanobis_dist"] = vals.median()
                if "gc3_deviation" in df.columns:
                    row["mean_abs_gc3_deviation"] = df["gc3_deviation"].abs().mean()
                return
            except Exception as e:
                logger.debug("Failed to read HGT Mahalanobis distance (key=%s): %s", key, e)


# ---------------------------------------------------------------------------
# 2. Within-condition analyses
# ---------------------------------------------------------------------------

def within_condition_stats(
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Compute summary statistics for each metric within each condition.

    Returns a long-format table: condition, metric, n, mean, sd, cv, median,
    q25, q75, min, max.
    """
    if condition_col not in metrics_df.columns:
        return pd.DataFrame()

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude RSCU columns from per-metric summaries (too many; they get their
    # own dedicated analysis)
    summary_metrics = [c for c in numeric_cols if c not in RSCU_COLUMN_NAMES]

    rows: list[dict] = []
    for cond, grp in metrics_df.groupby(condition_col):
        for metric in summary_metrics:
            vals = grp[metric].dropna()
            if len(vals) == 0:
                continue
            mu = vals.mean()
            sd = vals.std()
            rows.append({
                "condition": cond,
                "metric": metric,
                "n": len(vals),
                "mean": round(mu, 6),
                "sd": round(sd, 6),
                "cv": round(sd / mu, 4) if mu != 0 else np.nan,
                "median": round(vals.median(), 6),
                "q25": round(vals.quantile(0.25), 6),
                "q75": round(vals.quantile(0.75), 6),
                "min": round(vals.min(), 6),
                "max": round(vals.max(), 6),
            })

    return pd.DataFrame(rows)


def within_condition_rscu_dispersion(
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Coefficient of variation per RSCU codon within each condition.

    High CV = inconsistent codon usage within the condition; low CV =
    conserved codon preference.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    if not rscu_cols or condition_col not in metrics_df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for cond, grp in metrics_df.groupby(condition_col):
        for col in rscu_cols:
            vals = grp[col].dropna()
            if len(vals) < 2:
                continue
            mu = vals.mean()
            sd = vals.std()
            rows.append({
                "condition": cond,
                "codon": col,
                "amino_acid": col.split("-")[0],
                "mean_rscu": round(mu, 4),
                "sd_rscu": round(sd, 4),
                "cv": round(sd / mu, 4) if mu != 0 else np.nan,
                "n": len(vals),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Between-condition analyses
# ---------------------------------------------------------------------------

def between_condition_tests(
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Pairwise condition comparisons for every numeric metric.

    Uses Mann-Whitney U for 2-group comparisons and Kruskal-Wallis for 3+.
    Reports Cliff's delta effect size for 2-group tests.

    Returns long-format table: metric, test, group1, group2, statistic,
    p_value, corrected_p, effect_size, effect_label, significant.
    """
    if condition_col not in metrics_df.columns:
        return pd.DataFrame()

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    if len(conditions) < 2:
        return pd.DataFrame()

    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    summary_metrics = [c for c in numeric_cols if c not in RSCU_COLUMN_NAMES]

    rows: list[dict] = []

    # Kruskal-Wallis omnibus test (if 3+ conditions)
    if len(conditions) >= 3:
        for metric in summary_metrics:
            groups = [
                metrics_df.loc[metrics_df[condition_col] == c, metric].dropna()
                for c in conditions
            ]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) < 3:
                continue
            try:
                stat, p_val = sp_stats.kruskal(*groups)
                rows.append({
                    "metric": metric,
                    "test": "kruskal_wallis",
                    "group1": "omnibus",
                    "group2": "omnibus",
                    "statistic": round(stat, 4),
                    "p_value": p_val,
                    "effect_size": np.nan,
                    "effect_label": "",
                })
            except ValueError:
                continue

    # Pairwise Mann-Whitney U
    for metric in summary_metrics:
        for g1, g2 in itertools.combinations(conditions, 2):
            vals1 = metrics_df.loc[metrics_df[condition_col] == g1, metric].dropna()
            vals2 = metrics_df.loc[metrics_df[condition_col] == g2, metric].dropna()
            if len(vals1) < 3 or len(vals2) < 3:
                continue
            try:
                stat, p_val = sp_stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            except ValueError:
                continue

            # Cliff's delta effect size
            delta = _cliffs_delta(vals1.values, vals2.values)
            label = _effect_size_label(abs(delta))

            rows.append({
                "metric": metric,
                "test": "mann_whitney_u",
                "group1": g1,
                "group2": g2,
                "n_group1": len(vals1),
                "n_group2": len(vals2),
                "statistic": round(stat, 4),
                "p_value": p_val,
                "effect_size": round(delta, 4),
                "effect_label": label,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # BH FDR correction (within each test type)
    for test_type in result["test"].unique():
        mask = result["test"] == test_type
        pvals = result.loc[mask, "p_value"].values
        if len(pvals) == 0:
            continue
        result.loc[mask, "corrected_p"] = benjamini_hochberg(pvals)

    result["significant"] = result["corrected_p"] < 0.05
    return result


def between_condition_effect_summary(
    between_tests: pd.DataFrame,
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Summarize between-condition tests by effect size.

    Filters to Mann-Whitney U results only and produces a summary table sorted by
    absolute effect size, with columns: metric, group1, group2, mean_g1, mean_g2,
    percent_diff, effect_size, effect_label, corrected_p, significant.

    Args:
        between_tests: Output from between_condition_tests().
        metrics_df: Original metrics DataFrame.
        condition_col: Name of condition column in metrics_df.

    Returns:
        Filtered and annotated DataFrame sorted by absolute effect size.
    """
    if between_tests.empty:
        return pd.DataFrame()

    # Keep only Mann-Whitney U results
    mw_tests = between_tests[between_tests["test"] == "mann_whitney_u"].copy()
    if mw_tests.empty:
        return pd.DataFrame()

    # Add mean values from metrics_df
    summary_rows = []
    for _, row in mw_tests.iterrows():
        metric = row["metric"]
        g1, g2 = row["group1"], row["group2"]

        # Get means from metrics_df
        vals_g1 = metrics_df.loc[metrics_df[condition_col] == g1, metric].dropna()
        vals_g2 = metrics_df.loc[metrics_df[condition_col] == g2, metric].dropna()

        mean_g1 = vals_g1.mean() if len(vals_g1) > 0 else np.nan
        mean_g2 = vals_g2.mean() if len(vals_g2) > 0 else np.nan

        # Compute percent difference
        if not np.isnan(mean_g1) and mean_g1 != 0:
            percent_diff = 100 * (mean_g2 - mean_g1) / mean_g1
        else:
            percent_diff = np.nan

        summary_rows.append({
            "metric": metric,
            "group1": g1,
            "group2": g2,
            "mean_g1": round(mean_g1, 6),
            "mean_g2": round(mean_g2, 6),
            "percent_diff": round(percent_diff, 2) if not np.isnan(percent_diff) else np.nan,
            "effect_size": row["effect_size"],
            "effect_label": row["effect_label"],
            "corrected_p": round(row["corrected_p"], 4),
            "significant": row["significant"],
        })

    result = pd.DataFrame(summary_rows)
    # Sort by absolute effect size (descending)
    result["abs_effect"] = result["effect_size"].abs()
    result = result.sort_values("abs_effect", ascending=False).drop(columns=["abs_effect"])
    return result.reset_index(drop=True)


def between_condition_rscu_tests(
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
) -> pd.DataFrame:
    """Per-codon Mann-Whitney U tests between conditions.

    Returns: codon, amino_acid, group1, group2, mean_g1, mean_g2,
    log2_fold_change, statistic, p_value, corrected_p, effect_size,
    significant.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    if not rscu_cols or condition_col not in metrics_df.columns:
        return pd.DataFrame()

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    if len(conditions) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for codon in rscu_cols:
        for g1, g2 in itertools.combinations(conditions, 2):
            vals1 = metrics_df.loc[metrics_df[condition_col] == g1, codon].dropna()
            vals2 = metrics_df.loc[metrics_df[condition_col] == g2, codon].dropna()
            if len(vals1) < 3 or len(vals2) < 3:
                continue
            try:
                stat, p_val = sp_stats.mannwhitneyu(vals1, vals2, alternative="two-sided")
            except ValueError:
                continue

            mean1, mean2 = vals1.mean(), vals2.mean()
            pseudocount = 1e-4
            lfc = np.log2((mean2 + pseudocount) / (mean1 + pseudocount))
            delta = _cliffs_delta(vals1.values, vals2.values)

            rows.append({
                "codon": codon,
                "amino_acid": codon.split("-")[0],
                "group1": g1,
                "group2": g2,
                "mean_g1": round(mean1, 4),
                "mean_g2": round(mean2, 4),
                "log2_fold_change": round(lfc, 4) if not np.isnan(lfc) else np.nan,
                "statistic": round(stat, 4),
                "p_value": p_val,
                "effect_size": round(delta, 4),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # BH FDR per comparison pair
    for (g1, g2), grp in result.groupby(["group1", "group2"]):
        idx = grp.index
        result.loc[idx, "corrected_p"] = benjamini_hochberg(grp["p_value"].values)

    result["significant"] = result["corrected_p"] < 0.05
    return result


def permanova_rscu(
    metrics_df: pd.DataFrame,
    condition_col: str = "condition",
    n_perm: int = 999,
) -> dict:
    """PERMANOVA on Euclidean distances of genome-level RSCU profiles.

    Tests whether RSCU composition differs significantly between conditions.

    Returns dict with F_statistic, p_value, R2, n_perm.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    if not rscu_cols or condition_col not in metrics_df.columns:
        return {}

    df = metrics_df.dropna(subset=[condition_col])
    conditions = df[condition_col].values
    X = df[rscu_cols].values

    if len(np.unique(conditions)) < 2 or len(df) < 6:
        return {}

    # Remove rows with NaN in RSCU
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    conditions = conditions[mask]

    if len(np.unique(conditions)) < 2:
        return {}

    dist_matrix = squareform(pdist(X, metric="euclidean"))
    n = len(X)

    # Observed F
    f_obs = _permanova_f(dist_matrix, conditions)
    if f_obs is None:
        return {}

    # Permutation
    rng = np.random.default_rng(42)
    n_greater = 0
    for _ in range(n_perm):
        perm = rng.permutation(conditions)
        f_perm = _permanova_f(dist_matrix, perm)
        if f_perm is not None and f_perm >= f_obs:
            n_greater += 1

    p_value = (n_greater + 1) / (n_perm + 1)

    # R2 = SS_between / SS_total
    ss_total = (dist_matrix ** 2).sum() / (2 * n)
    groups = np.unique(conditions)
    ss_within = 0
    for g in groups:
        idx = np.where(conditions == g)[0]
        ng = len(idx)
        if ng > 1:
            ss_within += (dist_matrix[np.ix_(idx, idx)] ** 2).sum() / (2 * ng)
    # Approximate R2
    ss_between = ss_total - ss_within if ss_within < ss_total else 0
    r2 = ss_between / ss_total if ss_total > 0 else 0

    return {
        "F_statistic": round(f_obs, 4),
        "p_value": round(p_value, 4),
        "R2": round(r2, 4),
        "n_perm": n_perm,
        "n_samples": n,
        "n_groups": len(groups),
    }


def _permanova_f(dist_mat: np.ndarray, groups: np.ndarray) -> float | None:
    """Compute pseudo-F statistic for PERMANOVA."""
    n = len(groups)
    unique_g = np.unique(groups)
    a = len(unique_g)  # number of groups

    if a < 2 or n <= a:
        return None

    # SS_total
    ss_total = (dist_mat ** 2).sum() / (2 * n)

    # SS_within
    ss_within = 0
    for g in unique_g:
        idx = np.where(groups == g)[0]
        ng = len(idx)
        if ng > 1:
            ss_within += (dist_mat[np.ix_(idx, idx)] ** 2).sum() / (2 * ng)

    ss_between = ss_total - ss_within
    if ss_within == 0:
        return None

    f_stat = (ss_between / (a - 1)) / (ss_within / (n - a))
    return f_stat


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    more = 0
    less = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    return (more - less) / (nx * ny)


def _effect_size_label(d: float) -> str:
    """Interpret absolute Cliff's delta."""
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# 3b. Expression-class RSCU and enrichment between-condition comparisons
# ---------------------------------------------------------------------------


def between_condition_expression_class_rscu(
    metrics_df: pd.DataFrame,
    condition_col: str,
    prefix: str = "rp_",
    label: str = "ribosomal",
) -> pd.DataFrame:
    """Compare expression-class-specific RSCU between conditions.

    Tests each codon's RSCU (ribosomal or high-expression) between conditions
    using Mann-Whitney U with BH FDR correction per comparison pair.
    For 3+ conditions, performs all pairwise comparisons using itertools.combinations.

    Args:
        metrics_df: Sample metrics with prefixed RSCU columns.
        condition_col: Condition column name.
        prefix: Column prefix (``"rp_"`` for ribosomal, ``"he_"`` for high-expression).
        label: Human label for output table.

    Returns:
        DataFrame with per-codon test results sorted by adjusted p-value.
        For 2+ conditions: includes group1, group2, mean_group1, mean_group2 columns.
    """
    conditions = metrics_df[condition_col].dropna().unique()
    if len(conditions) < 2:
        return pd.DataFrame()

    rscu_cols = [c for c in metrics_df.columns
                 if c.startswith(prefix) and c.replace(prefix, "") in RSCU_COLUMN_NAMES]
    if not rscu_cols:
        return pd.DataFrame()

    cond_list = sorted(conditions)
    rows = []
    for col in rscu_cols:
        codon_name = col.replace(prefix, "")
        for g1, g2 in itertools.combinations(cond_list, 2):
            v1 = metrics_df.loc[metrics_df[condition_col] == g1, col].dropna()
            v2 = metrics_df.loc[metrics_df[condition_col] == g2, col].dropna()
            if len(v1) < 3 or len(v2) < 3:
                continue
            try:
                stat, p_val = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
            except ValueError:
                continue

            delta = _cliffs_delta(v1.values, v2.values)
            m1, m2 = v1.mean(), v2.mean()
            pseudocount = 1e-4
            log2fc = np.log2((m2 + pseudocount) / (m1 + pseudocount))

            parts = codon_name.split("-")
            aa = parts[0] if len(parts) == 2 else ""
            codon = parts[-1]

            rows.append({
                "gene_set": label,
                "amino_acid": aa,
                "codon": codon,
                "codon_col": codon_name,
                "group1": g1,
                "group2": g2,
                "mean_group1": round(m1, 4),
                "mean_group2": round(m2, 4),
                "log2_fc": round(log2fc, 4),
                "U_statistic": round(stat, 2),
                "p_value": p_val,
                "cliffs_delta": round(delta, 4),
                "effect_size": _effect_size_label(abs(delta)),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # BH FDR per comparison pair
    for (g1, g2), grp in result.groupby(["group1", "group2"]):
        idx = grp.index
        result.loc[idx, "p_adjusted"] = benjamini_hochberg(grp["p_value"].values)

    result["significant"] = result["p_adjusted"] < 0.05
    return result.sort_values("p_adjusted")


def between_condition_enrichment_comparison(
    sample_outputs: dict[str, dict[str, Path]],
    metrics_df: pd.DataFrame,
    condition_col: str,
) -> pd.DataFrame:
    """Compare pathway enrichment results between conditions.

    For each KEGG pathway, counts how many samples in each condition had it
    significantly enriched in high-expression genes.
    For 2 conditions: Fisher's exact test.
    For 3+ conditions: chi-squared test on full contingency table (enriched/not-enriched × conditions)
    plus all pairwise Fisher's exact tests.

    Returns a DataFrame with pathway, test type, per-condition enrichment counts,
    p-value, FDR-adjusted p-value, and for pairwise tests: group1, group2, odds_ratio.
    """
    conditions = metrics_df[condition_col].dropna().unique()
    if len(conditions) < 2:
        return pd.DataFrame()

    sid_cond = dict(zip(metrics_df["sample_id"], metrics_df[condition_col]))
    cond_list = sorted(conditions)

    # Collect enrichment results per sample for CAI_high (primary metric)
    cond_pathways: dict[str, dict[str, int]] = {c: {} for c in cond_list}
    cond_n_samples: dict[str, int] = {c: 0 for c in cond_list}
    all_pathways: dict[str, str] = {}  # pathway_id -> pathway_name

    for sid, paths in sample_outputs.items():
        if sid not in sid_cond:
            continue
        cond = sid_cond[sid]

        # Try CAI_high first, then MELP_high, then Fop_high
        for metric in ("CAI", "MELP", "Fop"):
            key = f"enrichment_{metric}_high"
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    sig_col = "significant" if "significant" in df.columns else None
                    fdr_col = "fdr" if "fdr" in df.columns else None
                    pw_col = "pathway" if "pathway" in df.columns else None
                    name_col = "pathway_name" if "pathway_name" in df.columns else None

                    if pw_col:
                        if sig_col:
                            sig_df = df[df[sig_col]]
                        elif fdr_col:
                            sig_df = df[df[fdr_col] < 0.05]
                        else:
                            sig_df = pd.DataFrame()

                        cond_n_samples[cond] = cond_n_samples.get(cond, 0) + 1
                        for _, row in sig_df.iterrows():
                            pw = row[pw_col]
                            cond_pathways[cond][pw] = cond_pathways[cond].get(pw, 0) + 1
                            if name_col and pw not in all_pathways:
                                all_pathways[pw] = row[name_col]
                except Exception as e:
                    logger.debug("Failed to read enrichment summary for between-condition comparison (metric=%s): %s", metric, e)
                break  # Use first available metric only

    if not all_pathways:
        return pd.DataFrame()

    rows = []

    # Omnibus chi-squared test (if 3+ conditions)
    if len(cond_list) >= 3:
        for pw in sorted(all_pathways.keys()):
            # Build contingency table: rows = [enriched, not_enriched], cols = conditions
            contingency = []
            skip_pw = False
            for cond in cond_list:
                n_enriched = cond_pathways[cond].get(pw, 0)
                n_total = cond_n_samples[cond]
                if n_total == 0:
                    skip_pw = True
                    break
                contingency.append([n_enriched, n_total - n_enriched])
            if skip_pw or len(contingency) != len(cond_list):
                continue
            try:
                chi2, p_val, dof, expected = sp_stats.chi2_contingency(np.array(contingency).T)
                rows.append({
                    "pathway": pw,
                    "pathway_name": all_pathways.get(pw, ""),
                    "test": "chi_squared",
                    "group1": "omnibus",
                    "group2": "omnibus",
                    "p_value": p_val,
                })
            except (ValueError, RuntimeError):
                pass

    # Pairwise Fisher's exact test for all pairs
    for g1, g2 in itertools.combinations(cond_list, 2):
        for pw in sorted(all_pathways.keys()):
            n1_enriched = cond_pathways[g1].get(pw, 0)
            n2_enriched = cond_pathways[g2].get(pw, 0)
            n1_total = cond_n_samples[g1]
            n2_total = cond_n_samples[g2]

            if n1_total == 0 or n2_total == 0:
                continue

            # 2x2 contingency table
            table = np.array([
                [n1_enriched, n1_total - n1_enriched],
                [n2_enriched, n2_total - n2_enriched],
            ])
            try:
                odds_ratio, p_val = sp_stats.fisher_exact(table, alternative="two-sided")
            except ValueError:
                continue

            rows.append({
                "pathway": pw,
                "pathway_name": all_pathways.get(pw, ""),
                "test": "fisher_exact",
                "group1": g1,
                "group2": g2,
                f"n_enriched_{g1}": n1_enriched,
                f"n_samples_{g1}": n1_total,
                f"frac_enriched_{g1}": round(n1_enriched / n1_total, 3),
                f"n_enriched_{g2}": n2_enriched,
                f"n_samples_{g2}": n2_total,
                f"frac_enriched_{g2}": round(n2_enriched / n2_total, 3),
                "odds_ratio": round(odds_ratio, 3) if np.isfinite(odds_ratio) else np.inf,
                "p_value": p_val,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # BH FDR globally across all tests
    result["p_adjusted"] = np.round(benjamini_hochberg(result["p_value"].values), 6)
    result["significant"] = result["p_adjusted"] < 0.05

    return result.sort_values("p_adjusted")


# ---------------------------------------------------------------------------
# 3c. Bio/ecology between-condition comparisons
# ---------------------------------------------------------------------------


def between_condition_strand_asymmetry_patterns(
    sample_outputs: dict[str, dict[str, Path]],
    metrics_df: pd.DataFrame,
    condition_col: str,
) -> pd.DataFrame:
    """Compare per-codon strand asymmetry patterns between conditions.

    For each codon, aggregates the plus-minus RSCU difference across samples
    within each condition. For 3+ conditions, performs Kruskal-Wallis omnibus test,
    then all pairwise Mann-Whitney U tests.

    Returns a DataFrame with one row per codon and test: mean asymmetry per condition,
    p-value, FDR-adjusted p-value, and Cliff's delta (for Mann-Whitney).
    Includes 'test' column ("kruskal_wallis" or "mann_whitney_u") and group1/group2 columns.
    """
    conditions = metrics_df[condition_col].dropna().unique()
    if len(conditions) < 2:
        return pd.DataFrame()

    # Build sample→condition map
    sid_cond = dict(zip(metrics_df["sample_id"], metrics_df[condition_col]))

    # Collect per-sample strand asymmetry data
    per_sample: dict[str, pd.DataFrame] = {}
    for sid, paths in sample_outputs.items():
        if sid not in sid_cond:
            continue
        for key in ("bio_strand_asymmetry_path", "bio_strand_asymmetry", "strand_asymmetry"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    if "mean_rscu_plus" in df.columns and "mean_rscu_minus" in df.columns:
                        df["asymmetry"] = df["mean_rscu_plus"] - df["mean_rscu_minus"]
                        per_sample[sid] = df
                except Exception as e:
                    logger.debug("Failed to read strand asymmetry for between-condition comparison (key=%s): %s", key, e)
                break

    if len(per_sample) < 4:
        return pd.DataFrame()

    # Identify common codons
    all_codons = set()
    for df in per_sample.values():
        codon_col = "codon_col" if "codon_col" in df.columns else "codon"
        all_codons.update(df[codon_col].values)

    cond_list = sorted(conditions)
    rows = []

    # Kruskal-Wallis omnibus test (if 3+ conditions)
    if len(cond_list) >= 3:
        for codon in sorted(all_codons):
            cond_vals: dict[str, list] = {c: [] for c in cond_list}
            for sid, df in per_sample.items():
                codon_col = "codon_col" if "codon_col" in df.columns else "codon"
                match = df.loc[df[codon_col] == codon, "asymmetry"]
                if not match.empty:
                    cond_vals[sid_cond[sid]].append(match.iloc[0])

            # Need ≥3 samples per condition
            groups = [np.array(cond_vals[c]) for c in cond_list]
            groups = [g for g in groups if len(g) >= 3]
            if len(groups) < 3:
                continue

            try:
                stat, p_val = sp_stats.kruskal(*groups)
                rows.append({
                    "codon": codon,
                    "test": "kruskal_wallis",
                    "group1": "omnibus",
                    "group2": "omnibus",
                    "statistic": round(stat, 4),
                    "p_value": p_val,
                })
            except ValueError:
                continue

    # Pairwise Mann-Whitney U tests
    for codon in sorted(all_codons):
        cond_vals: dict[str, list] = {c: [] for c in cond_list}
        for sid, df in per_sample.items():
            codon_col = "codon_col" if "codon_col" in df.columns else "codon"
            match = df.loc[df[codon_col] == codon, "asymmetry"]
            if not match.empty:
                cond_vals[sid_cond[sid]].append(match.iloc[0])

        for g1, g2 in itertools.combinations(cond_list, 2):
            v1 = np.array(cond_vals[g1])
            v2 = np.array(cond_vals[g2])
            if len(v1) < 3 or len(v2) < 3:
                continue
            try:
                stat, p_val = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
            except ValueError:
                continue

            delta = _cliffs_delta(v1, v2)
            mean1, mean2 = np.mean(v1), np.mean(v2)
            rows.append({
                "codon": codon,
                "test": "mann_whitney_u",
                "group1": g1,
                "group2": g2,
                "mean_group1": round(mean1, 5),
                "mean_group2": round(mean2, 5),
                "diff": round(mean1 - mean2, 5),
                "U_statistic": round(stat, 2),
                "p_value": p_val,
                "cliffs_delta": round(delta, 4),
                "effect_size": _effect_size_label(abs(delta)),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # BH FDR correction per test type
    for test_type in result["test"].unique():
        mask = result["test"] == test_type
        idx = result[mask].index
        pvals = result.loc[mask, "p_value"].values
        n = len(pvals)
        if n == 0:
            continue
        result.loc[idx, "p_adjusted"] = np.round(benjamini_hochberg(pvals), 6)

    result["significant"] = result["p_adjusted"] < 0.05
    return result.sort_values("p_adjusted")


def between_condition_optimal_codons(
    sample_outputs: dict[str, dict[str, Path]],
    metrics_df: pd.DataFrame,
    condition_col: str,
) -> pd.DataFrame:
    """Compare optimal codon identity between conditions.

    Pools optimal codon tables across samples within each condition,
    computing mean delta-RSCU per codon for ALL conditions. For exactly 2 conditions,
    output is identical to before (backward compatible).

    Returns a DataFrame with codon, amino_acid, mean delta-RSCU for each condition,
    optimal_in_{condition} for each condition, n_conditions_optimal (count),
    unanimous (True if all or none are optimal), and delta_difference (max - min).
    """
    conditions = metrics_df[condition_col].dropna().unique()
    if len(conditions) < 2:
        return pd.DataFrame()

    sid_cond = dict(zip(metrics_df["sample_id"], metrics_df[condition_col]))

    # Collect per-sample optimal codon data
    cond_deltas: dict[str, list[pd.DataFrame]] = {c: [] for c in conditions}
    for sid, paths in sample_outputs.items():
        if sid not in sid_cond:
            continue
        for key in ("bio_trans_sel_optimal_codons_path", "bio_trans_sel_optimal_codons"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    if "delta_rscu" in df.columns and "codon" in df.columns:
                        cond_deltas[sid_cond[sid]].append(df[["amino_acid", "codon", "delta_rscu"]])
                except Exception as e:
                    logger.debug("Failed to read translational selection position_effects for between-condition comparison (key=%s): %s", key, e)
                break

    cond_list = sorted(conditions)
    # Check all conditions have data
    if any(not cond_deltas.get(c, []) for c in cond_list):
        return pd.DataFrame()

    # Average delta-RSCU per codon within each condition
    combined = {}
    for cond in cond_list:
        all_df = pd.concat(cond_deltas[cond], ignore_index=True)
        agg = all_df.groupby(["amino_acid", "codon"])["delta_rscu"].mean().reset_index()
        agg.rename(columns={"delta_rscu": f"mean_delta_rscu_{cond}"}, inplace=True)
        combined[cond] = agg

    # Merge all conditions
    merged = combined[cond_list[0]].copy()
    for cond in cond_list[1:]:
        merged = merged.merge(combined[cond], on=["amino_acid", "codon"], how="outer")
    merged = merged.fillna(0)

    # Determine optimal status per condition (delta > 0 = enriched in high-expr genes)
    for cond in cond_list:
        merged[f"optimal_in_{cond}"] = merged[f"mean_delta_rscu_{cond}"] > 0

    # Count how many conditions consider this codon optimal
    optimal_cols = [f"optimal_in_{c}" for c in cond_list]
    merged["n_conditions_optimal"] = merged[optimal_cols].sum(axis=1).astype(int)
    merged["unanimous"] = (merged["n_conditions_optimal"] == len(cond_list)) | (merged["n_conditions_optimal"] == 0)

    # Delta difference: max - min across conditions
    delta_cols = [f"mean_delta_rscu_{c}" for c in cond_list]
    merged["delta_difference"] = (
        merged[delta_cols].max(axis=1) - merged[delta_cols].min(axis=1)
    ).round(4)

    return merged.sort_values("delta_difference", key=abs, ascending=False)


def between_condition_hgt_burden(
    sample_outputs: dict[str, dict[str, Path]],
    metrics_df: pd.DataFrame,
    condition_col: str,
) -> dict:
    """Compare HGT burden distributions between conditions.

    Computes per-sample summary of Mahalanobis distances and HGT fractions.
    For 3+ conditions: Kruskal-Wallis omnibus test plus all pairwise Mann-Whitney U.
    For exactly 2 conditions: backward-compatible behavior (single Mann-Whitney per metric).

    Returns a dict with keys like:
    - For 2 conditions (backward compat): "mahalanobis", "hgt_fraction" with single test result
    - For 3+ conditions: "mahalanobis_omnibus", "mahalanobis_g1_vs_g2", "hgt_fraction_omnibus", etc.
    Each value is a dict with per-condition summary stats and test results.
    """
    conditions = metrics_df[condition_col].dropna().unique()
    if len(conditions) < 2:
        return {}

    sid_cond = dict(zip(metrics_df["sample_id"], metrics_df[condition_col]))
    cond_list = sorted(conditions)

    # Collect per-sample Mahalanobis distance distributions
    cond_medians: dict[str, list[float]] = {c: [] for c in conditions}
    cond_hgt_fracs: dict[str, list[float]] = {c: [] for c in conditions}
    for sid, paths in sample_outputs.items():
        if sid not in sid_cond:
            continue
        for key in ("bio_hgt_candidates_path", "bio_hgt_candidates", "hgt_candidates"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    if "mahalanobis_dist" in df.columns:
                        cond_medians[sid_cond[sid]].append(df["mahalanobis_dist"].median())
                    if "hgt_flag" in df.columns:
                        cond_hgt_fracs[sid_cond[sid]].append(df["hgt_flag"].mean())
                except Exception as e:
                    logger.debug("Failed to read HGT Mahalanobis distance for between-condition comparison (key=%s): %s", key, e)
                break

    result: dict = {}

    # Kruskal-Wallis omnibus tests (if 3+ conditions)
    if len(cond_list) >= 3:
        # Mahalanobis omnibus
        groups = [np.array(cond_medians.get(c, [])) for c in cond_list]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 3:
            try:
                stat, p_val = sp_stats.kruskal(*groups)
                result["mahalanobis_omnibus"] = {
                    "test": "kruskal_wallis",
                    "conditions": cond_list,
                    "H_statistic": round(stat, 4),
                    "p_value": round(p_val, 6),
                }
            except ValueError:
                pass

        # HGT fraction omnibus
        groups = [np.array(cond_hgt_fracs.get(c, [])) for c in cond_list]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) >= 3:
            try:
                stat, p_val = sp_stats.kruskal(*groups)
                result["hgt_fraction_omnibus"] = {
                    "test": "kruskal_wallis",
                    "conditions": cond_list,
                    "H_statistic": round(stat, 4),
                    "p_value": round(p_val, 6),
                }
            except ValueError:
                pass

    # Pairwise Mann-Whitney U tests for all pairs
    for g1, g2 in itertools.combinations(cond_list, 2):
        # Mahalanobis
        v1 = np.array(cond_medians.get(g1, []))
        v2 = np.array(cond_medians.get(g2, []))
        if len(v1) >= 3 and len(v2) >= 3:
            try:
                stat, p_val = sp_stats.mannwhitneyu(v1, v2, alternative="two-sided")
                delta = _cliffs_delta(v1, v2)
                key = f"mahalanobis_{g1}_vs_{g2}"
                result[key] = {
                    "test": "mann_whitney_u",
                    "conditions": [g1, g2],
                    f"median_{g1}": round(float(np.median(v1)), 4),
                    f"median_{g2}": round(float(np.median(v2)), 4),
                    "U_statistic": round(float(stat), 2),
                    "p_value": round(p_val, 6),
                    "cliffs_delta": round(delta, 4),
                    "effect_size": _effect_size_label(abs(delta)),
                    f"values_{g1}": v1.tolist(),
                    f"values_{g2}": v2.tolist(),
                }
                # For backward compatibility with 2 conditions: also set "mahalanobis" key
                if len(cond_list) == 2:
                    result["mahalanobis"] = result[key]
            except ValueError:
                pass

        # HGT fraction
        f1 = np.array(cond_hgt_fracs.get(g1, []))
        f2 = np.array(cond_hgt_fracs.get(g2, []))
        if len(f1) >= 3 and len(f2) >= 3:
            try:
                stat, p_val = sp_stats.mannwhitneyu(f1, f2, alternative="two-sided")
                delta = _cliffs_delta(f1, f2)
                key = f"hgt_fraction_{g1}_vs_{g2}"
                result[key] = {
                    "test": "mann_whitney_u",
                    "conditions": [g1, g2],
                    f"median_{g1}": round(float(np.median(f1)), 4),
                    f"median_{g2}": round(float(np.median(f2)), 4),
                    "U_statistic": round(float(stat), 2),
                    "p_value": round(p_val, 6),
                    "cliffs_delta": round(delta, 4),
                    "effect_size": _effect_size_label(abs(delta)),
                    f"values_{g1}": f1.tolist(),
                    f"values_{g2}": f2.tolist(),
                }
                # For backward compatibility with 2 conditions: also set "hgt_fraction" key
                if len(cond_list) == 2:
                    result["hgt_fraction"] = result[key]
            except ValueError:
                pass

    return result


def between_condition_gc3_gc12(
    sample_outputs: dict[str, dict[str, Path]],
    metrics_df: pd.DataFrame,
    condition_col: str,
) -> dict[str, pd.DataFrame]:
    """Collect per-gene GC3 vs GC12 data split by condition for neutrality plots.

    Returns dict: {condition_name: DataFrame with GC3, GC12 columns}.
    """
    conditions = metrics_df[condition_col].dropna().unique()
    sid_cond = dict(zip(metrics_df["sample_id"], metrics_df[condition_col]))

    cond_data: dict[str, list[pd.DataFrame]] = {c: [] for c in conditions}
    for sid, paths in sample_outputs.items():
        if sid not in sid_cond:
            continue
        p = paths.get("gc12_gc3")
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                gc12_col = next((c for c in ("GC12", "gc12") if c in df.columns), None)
                gc3_col = next((c for c in ("GC3", "gc3") if c in df.columns), None)
                if gc12_col and gc3_col:
                    sub = df[[gc3_col, gc12_col]].rename(
                        columns={gc3_col: "GC3", gc12_col: "GC12"}
                    ).dropna()
                    if not sub.empty:
                        sub["sample_id"] = sid
                        cond_data[sid_cond[sid]].append(sub)
            except Exception as e:
                logger.debug("Failed to read GC content for between-condition comparison: %s", e)

    result = {}
    for cond, dfs in cond_data.items():
        if dfs:
            result[cond] = pd.concat(dfs, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 4. Orchestrator
# ---------------------------------------------------------------------------

def run_comparative_analyses(
    sample_outputs: dict[str, dict[str, Path]],
    batch_df: pd.DataFrame,
    output_dir: Path,
    condition_col: str | None = None,
    metadata_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Run all condition-aware comparative analyses.

    Args:
        sample_outputs: Per-sample output paths dict.
        batch_df: Original batch table.
        output_dir: Base output directory.
        condition_col: Column name for the experimental condition.
        metadata_cols: Additional metadata columns.

    Returns:
        (metrics_df, output_paths) — the collected metrics table and a dict
        of written output file paths.
    """
    comp_dir = output_dir / "batch_condition"
    comp_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Collect per-sample metrics
    logger.info("Collecting per-sample metrics for comparative analyses")
    metrics_df = collect_sample_metrics(sample_outputs, batch_df, condition_col)

    # Save metrics table
    metrics_path = comp_dir / "sample_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    outputs["sample_metrics"] = metrics_path
    logger.info("Collected %d metrics across %d samples", len(metrics_df.columns) - 1, len(metrics_df))

    if condition_col and condition_col in metrics_df.columns:
        conditions = metrics_df[condition_col].dropna().unique()
        logger.info("Conditions detected: %s", list(conditions))

        # ── Within-condition ──────────────────────────────────────────────
        logger.info("Running within-condition summary statistics")
        within_stats = within_condition_stats(metrics_df, condition_col)
        if not within_stats.empty:
            p = comp_dir / "within_condition_stats.tsv"
            within_stats.to_csv(p, sep="\t", index=False)
            outputs["within_condition_stats"] = p

        rscu_disp = within_condition_rscu_dispersion(metrics_df, condition_col)
        if not rscu_disp.empty:
            p = comp_dir / "within_condition_rscu_dispersion.tsv"
            rscu_disp.to_csv(p, sep="\t", index=False)
            outputs["within_condition_rscu_dispersion"] = p

        # ── Between-condition ─────────────────────────────────────────────
        if len(conditions) >= 2:
            logger.info("Running between-condition statistical tests")

            # Metric-level tests
            between_tests = between_condition_tests(metrics_df, condition_col)
            if not between_tests.empty:
                p = comp_dir / "between_condition_tests.tsv"
                between_tests.to_csv(p, sep="\t", index=False)
                outputs["between_condition_tests"] = p
                n_sig = between_tests["significant"].sum()
                logger.info(
                    "Between-condition tests: %d tests, %d significant (FDR < 0.05)",
                    len(between_tests), n_sig,
                )

            # RSCU per-codon tests
            rscu_tests = between_condition_rscu_tests(metrics_df, condition_col)
            if not rscu_tests.empty:
                p = comp_dir / "between_condition_rscu_tests.tsv"
                rscu_tests.to_csv(p, sep="\t", index=False)
                outputs["between_condition_rscu_tests"] = p

            # PERMANOVA on RSCU profiles
            logger.info("Running PERMANOVA on RSCU profiles")
            perm_result = permanova_rscu(metrics_df, condition_col)
            if perm_result:
                p = comp_dir / "permanova_rscu.tsv"
                pd.DataFrame([perm_result]).to_csv(p, sep="\t", index=False)
                outputs["permanova_rscu"] = p
                logger.info(
                    "PERMANOVA: F=%.3f, p=%.4f, R²=%.3f",
                    perm_result["F_statistic"],
                    perm_result["p_value"],
                    perm_result["R2"],
                )

            # Between-condition effect summary
            logger.info("Generating between-condition effect summary")
            effect_summary = between_condition_effect_summary(between_tests, metrics_df, condition_col)
            if not effect_summary.empty:
                p = comp_dir / "between_condition_effect_summary.tsv"
                effect_summary.to_csv(p, sep="\t", index=False)
                outputs["between_condition_effect_summary"] = p
                logger.info(
                    "Effect summary: %d Mann-Whitney U results sorted by effect size",
                    len(effect_summary),
                )

            # ── Expression-class RSCU comparisons ─────────────────────────
            logger.info("Running expression-class RSCU comparisons")

            # Ribosomal protein RSCU
            rp_rscu_tests = between_condition_expression_class_rscu(
                metrics_df, condition_col, prefix="rp_", label="ribosomal",
            )
            if not rp_rscu_tests.empty:
                p = comp_dir / "between_condition_ribosomal_rscu.tsv"
                rp_rscu_tests.to_csv(p, sep="\t", index=False)
                outputs["between_condition_ribosomal_rscu"] = p
                n_sig = rp_rscu_tests["significant"].sum()
                logger.info("Ribosomal RSCU: %d codons tested, %d significant",
                            len(rp_rscu_tests), n_sig)

            # High-expression gene RSCU
            he_rscu_tests = between_condition_expression_class_rscu(
                metrics_df, condition_col, prefix="he_", label="high_expression",
            )
            if not he_rscu_tests.empty:
                p = comp_dir / "between_condition_high_expression_rscu.tsv"
                he_rscu_tests.to_csv(p, sep="\t", index=False)
                outputs["between_condition_high_expression_rscu"] = p
                n_sig = he_rscu_tests["significant"].sum()
                logger.info("High-expression RSCU: %d codons tested, %d significant",
                            len(he_rscu_tests), n_sig)

            # Pathway enrichment comparison
            enrichment_comp = between_condition_enrichment_comparison(
                sample_outputs, metrics_df, condition_col,
            )
            if not enrichment_comp.empty:
                p = comp_dir / "between_condition_enrichment_comparison.tsv"
                enrichment_comp.to_csv(p, sep="\t", index=False)
                outputs["between_condition_enrichment_comparison"] = p
                n_sig = enrichment_comp["significant"].sum()
                logger.info("Enrichment comparison: %d pathways, %d with differential enrichment",
                            len(enrichment_comp), n_sig)

            # ── Bio/ecology between-condition comparisons ─────────────────
            logger.info("Running bio/ecology between-condition comparisons")

            # Per-codon strand asymmetry pattern comparison
            strand_asym_patterns = between_condition_strand_asymmetry_patterns(
                sample_outputs, metrics_df, condition_col,
            )
            if not strand_asym_patterns.empty:
                p = comp_dir / "between_condition_strand_asymmetry_patterns.tsv"
                strand_asym_patterns.to_csv(p, sep="\t", index=False)
                outputs["between_condition_strand_asymmetry_patterns"] = p
                n_sig = strand_asym_patterns["significant"].sum()
                logger.info("Strand asymmetry patterns: %d codons tested, %d significant",
                            len(strand_asym_patterns), n_sig)

            # Optimal codon identity comparison
            opt_codons = between_condition_optimal_codons(
                sample_outputs, metrics_df, condition_col,
            )
            if not opt_codons.empty:
                p = comp_dir / "between_condition_optimal_codons.tsv"
                opt_codons.to_csv(p, sep="\t", index=False)
                outputs["between_condition_optimal_codons"] = p
                n_disagree = (~opt_codons["unanimous"]).sum()
                logger.info("Optimal codons: %d codons compared, %d differ between conditions",
                            len(opt_codons), n_disagree)

            # HGT burden comparison
            hgt_burden = between_condition_hgt_burden(
                sample_outputs, metrics_df, condition_col,
            )
            if hgt_burden:
                rows_list = []
                for test_name, vals in hgt_burden.items():
                    row_out = {k: v for k, v in vals.items()
                               if not isinstance(v, list) and k != "conditions"}
                    row_out["test"] = test_name
                    rows_list.append(row_out)
                if rows_list:
                    p = comp_dir / "between_condition_hgt_burden.tsv"
                    pd.DataFrame(rows_list).to_csv(p, sep="\t", index=False)
                    outputs["between_condition_hgt_burden"] = p
                logger.info("HGT burden comparison: %d tests", len(hgt_burden))

            # GC3 vs GC12 per-condition data (for neutrality scatter plot)
            gc3_gc12_data = between_condition_gc3_gc12(
                sample_outputs, metrics_df, condition_col,
            )
            if gc3_gc12_data:
                # Save combined data for plotting
                combined_rows = []
                for cond, df in gc3_gc12_data.items():
                    df_copy = df.copy()
                    df_copy["condition"] = cond
                    combined_rows.append(df_copy)
                if combined_rows:
                    combined_df = pd.concat(combined_rows, ignore_index=True)
                    p = comp_dir / "between_condition_gc3_gc12.tsv"
                    combined_df.to_csv(p, sep="\t", index=False)
                    outputs["between_condition_gc3_gc12"] = p
                    logger.info("GC3-GC12 neutrality data: %d genes across %d conditions",
                                len(combined_df), len(gc3_gc12_data))
    else:
        logger.info("No condition column specified; skipping condition-aware analyses")

    return metrics_df, outputs
