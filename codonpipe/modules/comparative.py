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

from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES

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
                row["condition"] = match.iloc[0]

        # --- RSCU median (genome-level) ---
        _read_rscu_median(paths, row)

        # --- Expression scores (genome-level medians) ---
        _read_expression_summary(paths, row)

        # --- ENC / GC3 ---
        _read_enc_summary(paths, row)

        # --- Growth rate prediction ---
        _read_growth_rate(paths, row)

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
        except Exception:
            pass


def _read_expression_summary(paths: dict, row: dict) -> None:
    p = paths.get("expression_combined")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            for metric in ("MELP", "CAI", "Fop"):
                if metric in df.columns:
                    row[f"median_{metric}"] = df[metric].median()
                    row[f"mean_{metric}"] = df[metric].mean()
        except Exception:
            pass


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
        except Exception:
            pass


def _read_growth_rate(paths: dict, row: dict) -> None:
    """Read growth rate prediction (stored as a single-row TSV or in bio outputs)."""
    # Try bio ecology output path pattern
    for key in ("bio_growth_rate_prediction", "growth_rate_prediction"):
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
            except Exception:
                pass


def _read_hgt_summary(paths: dict, row: dict) -> None:
    for key in ("bio_hgt_candidates", "hgt_candidates"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                row["n_hgt_candidates"] = len(df)
                if "is_hgt" in df.columns:
                    row["n_hgt_positive"] = int(df["is_hgt"].sum())
                    row["hgt_fraction"] = df["is_hgt"].mean()
                elif "p_value" in df.columns:
                    n_sig = (df["p_value"] < 0.05).sum()
                    row["n_hgt_positive"] = int(n_sig)
                    row["hgt_fraction"] = n_sig / len(df) if len(df) > 0 else 0
                return
            except Exception:
                pass


def _read_strand_asymmetry(paths: dict, row: dict) -> None:
    for key in ("bio_strand_asymmetry", "strand_asymmetry"):
        p = paths.get(key)
        if p and Path(p).exists():
            try:
                df = pd.read_csv(p, sep="\t")
                if "p_value" in df.columns:
                    row["n_strand_asym_codons"] = len(df)
                    row["n_strand_asym_sig"] = int((df["p_value"] < 0.05).sum())
                    row["strand_asym_fraction"] = row["n_strand_asym_sig"] / max(len(df), 1)
                return
            except Exception:
                pass


def _read_operon_summary(paths: dict, row: dict) -> None:
    for key in ("bio_operon_coadaptation", "operon_coadaptation"):
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
            except Exception:
                pass


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
        except Exception:
            pass


def _read_s_value(paths: dict, row: dict) -> None:
    p = paths.get("s_value")
    if p and Path(p).exists():
        try:
            df = pd.read_csv(p, sep="\t")
            s_col = next((c for c in ("S", "s_value", "S_value") if c in df.columns), None)
            if s_col:
                row["mean_S_value"] = df[s_col].mean()
                row["median_S_value"] = df[s_col].median()
        except Exception:
            pass


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
        n_tests = len(pvals)
        if n_tests == 0:
            continue
        sorted_idx = np.argsort(pvals)
        ranks = np.empty_like(sorted_idx)
        ranks[sorted_idx] = np.arange(1, n_tests + 1)
        corrected = np.minimum(pvals * n_tests / ranks, 1.0)
        # Enforce monotonicity
        corrected_sorted = corrected[sorted_idx]
        for i in range(n_tests - 2, -1, -1):
            corrected_sorted[i + 1] = min(corrected_sorted[i + 1], corrected_sorted[i] if i < n_tests - 1 else 1.0)
        corrected[sorted_idx] = corrected_sorted
        result.loc[mask, "corrected_p"] = corrected

    result["significant"] = result["corrected_p"] < 0.05
    return result


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
            lfc = np.log2(mean2 / mean1) if mean1 > 0 and mean2 > 0 else np.nan
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
        pvals = grp["p_value"].values
        n = len(pvals)
        sorted_i = np.argsort(pvals)
        ranks = np.empty_like(sorted_i)
        ranks[sorted_i] = np.arange(1, n + 1)
        corrected = np.minimum(pvals * n / ranks, 1.0)
        result.loc[idx, "corrected_p"] = corrected

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
            ss_within += dist_matrix[np.ix_(idx, idx)].sum() ** 0.5 / (2 * ng)
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


def _permanova_f(dist_sq: np.ndarray, groups: np.ndarray) -> float | None:
    """Compute pseudo-F statistic for PERMANOVA."""
    n = len(groups)
    unique_g = np.unique(groups)
    a = len(unique_g)  # number of groups

    if a < 2 or n <= a:
        return None

    # SS_total
    ss_total = (dist_sq ** 2).sum() / (2 * n)

    # SS_within
    ss_within = 0
    for g in unique_g:
        idx = np.where(groups == g)[0]
        ng = len(idx)
        if ng > 1:
            ss_within += (dist_sq[np.ix_(idx, idx)] ** 2).sum() / (2 * ng)

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
    comp_dir = output_dir / "comparative"
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
    else:
        logger.info("No condition column specified; skipping condition-aware analyses")

    return metrics_df, outputs
