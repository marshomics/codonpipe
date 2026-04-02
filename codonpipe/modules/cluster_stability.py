"""Bootstrap stability analysis for Mahalanobis clustering.

Simplified approach: since the main clustering module now uses bootstrap-
stabilised centroids and chi-squared thresholds, this module only needs to:

1. Run B bootstrap replicates (resample RP genes with replacement)
2. Record per-gene membership frequency across replicates
3. Identify the "core" gene set (genes present in >= core_threshold fraction)
4. Compute frequency-weighted RSCU for the core set
5. Report stability metrics

No multiplier grid sweep — the chi-squared threshold is fixed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.mahal_clustering import (
    _compute_mahalanobis_distances,
    _fit_robust_rp_reference,
    _select_n_axes,
    _compute_cluster_rscu,
    _chi2_threshold,
    _MAX_COA_AXES,
    _MIN_COA_AXES,
    _MIN_GENES_FOR_CLUSTERING,
    _RP_OUTLIER_ALPHA,
    _MIN_RP_FOR_ROBUST,
    _CLUSTER_CHI2_P,
)
from codonpipe.plotting.utils import DPI, FORMATS, STYLE_PARAMS
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_N_BOOTSTRAPS = 200
_DEFAULT_CORE_THRESHOLD = 0.5
# Kept for backward compatibility with pipeline calls
_DEFAULT_MULTIPLIER_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

# Weights for composite stability score
_W_JACCARD = 0.40
_W_MEAN_FREQ = 0.30
_W_COSINE = 0.30


# ---------------------------------------------------------------------------
# COA preparation (shared with main module)
# ---------------------------------------------------------------------------

def _prepare_coa_space(
    rscu_gene_df: pd.DataFrame,
    rp_gene_ids: set[str],
    expr_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, int] | None:
    """Run COA and extract coordinates + RP indices."""
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df=expr_df)
    if not coa_results or "coa_coords" not in coa_results:
        return None

    coa_coords = coa_results["coa_coords"]
    coa_inertia = coa_results.get("coa_inertia", pd.DataFrame())

    n_axes = _select_n_axes(coa_inertia, _MAX_COA_AXES)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n_axes = len(axis_cols)
    if n_axes < _MIN_COA_AXES:
        return None

    gene_ids = coa_coords["gene"].astype(str).tolist()
    X = coa_coords[axis_cols].values
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < _MIN_GENES_FOR_CLUSTERING:
        return None
    X = X[valid_mask]
    gene_ids = [g for g, v in zip(gene_ids, valid_mask) if v]

    rp_indices = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids])
    if len(rp_indices) < 3:
        return None

    return X, gene_ids, rp_indices, n_axes


# ---------------------------------------------------------------------------
# Single bootstrap replicate
# ---------------------------------------------------------------------------

def _bootstrap_rp_reference(
    X: np.ndarray,
    gene_ids: list[str],
    rp_indices: np.ndarray,
    n_axes: int,
    multiplier: float,
    seed: int = 0,
) -> set[str]:
    """One bootstrap replicate using chi-squared threshold.

    The multiplier parameter is accepted for backward compatibility but
    ignored — the threshold is always chi-squared based.
    """
    rng = np.random.RandomState(seed)
    n_rp = len(rp_indices)
    boot_rp_idx = rng.choice(rp_indices, size=n_rp, replace=True)
    X_rp = X[boot_rp_idx]

    centroid, cov, cov_inv, _ = _fit_robust_rp_reference(
        X_rp, n_axes, alpha=_RP_OUTLIER_ALPHA, min_rp=_MIN_RP_FOR_ROBUST,
    )

    distances = _compute_mahalanobis_distances(X, centroid, cov_inv)
    threshold = _chi2_threshold(n_axes, _CLUSTER_CHI2_P)
    opt_mask = distances <= threshold
    return {gid for gid, inside in zip(gene_ids, opt_mask) if inside}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Main stability analysis
# ---------------------------------------------------------------------------

def run_stability_analysis(
    rscu_gene_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    ffn_path: Path | None = None,
    rp_ids_file: Path | None = None,
    rp_rscu_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    n_bootstraps: int = _DEFAULT_N_BOOTSTRAPS,
    multiplier_grid: list[float] | None = None,
    core_threshold: float = _DEFAULT_CORE_THRESHOLD,
    rp_gene_ids_override: set[str] | None = None,
    output_subdir: str | None = None,
) -> dict:
    """Bootstrap stability analysis with chi-squared threshold.

    Runs B bootstrap replicates, records per-gene membership frequency,
    identifies core genes, and computes frequency-weighted core RSCU.

    The multiplier_grid parameter is accepted for backward compatibility
    but only the first value (or default) is used — the chi-squared
    threshold replaces the multiplier sweep.
    """
    stab_dir = output_dir / (output_subdir or "cluster_stability")
    stab_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ── Load RP gene IDs ────────────────────────────────────────────
    if rp_gene_ids_override is not None:
        rp_gene_ids = set(rp_gene_ids_override)
    else:
        rp_gene_ids: set[str] = set()
        if rp_ids_file and rp_ids_file.exists():
            rp_gene_ids = {
                line.strip() for line in rp_ids_file.read_text().splitlines()
                if line.strip()
            }
    if not rp_gene_ids:
        logger.warning("No RP IDs for stability analysis of %s; skipping.", sample_id)
        return results

    n_genes = len(rscu_gene_df)
    if n_genes < _MIN_GENES_FOR_CLUSTERING:
        logger.warning("Too few genes (%d) for stability of %s; skipping.", n_genes, sample_id)
        return results

    all_gene_ids = rscu_gene_df["gene"].astype(str).tolist()

    # ── COA ─────────────────────────────────────────────────────────
    logger.info(
        "Stability analysis for %s: %d RP-bootstrap replicates (core threshold=%.2f)",
        sample_id, n_bootstraps, core_threshold,
    )

    coa_prep = _prepare_coa_space(rscu_gene_df, rp_gene_ids, expr_df=expr_df)
    if coa_prep is None:
        logger.warning("COA failed for stability analysis of %s", sample_id)
        return results

    X, coa_gene_ids, rp_indices, n_axes = coa_prep

    # ── Bootstrap ───────────────────────────────────────────────────
    boot_clusters: list[set[str]] = []
    for b in range(n_bootstraps):
        cluster_ids = _bootstrap_rp_reference(
            X, coa_gene_ids, rp_indices, n_axes,
            multiplier=0.0,  # unused, chi-squared threshold
            seed=b,
        )
        boot_clusters.append(cluster_ids)

    # ── Per-gene membership frequency ──────────────────────────────
    freq: dict[str, float] = {}
    for gid in all_gene_ids:
        count = sum(1 for cluster in boot_clusters if gid in cluster)
        freq[gid] = count / n_bootstraps

    all_freqs = np.array([freq.get(gid, 0.0) for gid in all_gene_ids])

    # ── Core gene set ──────────────────────────────────────────────
    core_mask = all_freqs >= core_threshold
    n_core = int(core_mask.sum())
    core_ids = {gid for gid, f in freq.items() if f >= core_threshold}

    # ── Stability metrics ──────────────────────────────────────────
    # Pairwise Jaccard
    n_pairs = min(500, n_bootstraps * (n_bootstraps - 1) // 2)
    rng = np.random.RandomState(42)
    jaccards = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_bootstraps, size=2, replace=False)
        jaccards.append(_jaccard(boot_clusters[i], boot_clusters[j]))
    mean_jaccard = float(np.mean(jaccards))

    mean_core_freq = float(np.mean(all_freqs[core_mask])) if n_core > 0 else 0.0
    sizes = [len(c) for c in boot_clusters]
    mean_size = float(np.mean(sizes))
    std_size = float(np.std(sizes))

    rp_in_core = sum(1 for g in rp_gene_ids if freq.get(g, 0) >= core_threshold)
    rp_coverage = rp_in_core / len(rp_gene_ids) if rp_gene_ids else 0.0

    # Cosine similarity: core RSCU vs RP-only RSCU
    cosine_sim = np.nan
    if rp_rscu_df is not None and not rp_rscu_df.empty and ffn_path and ffn_path.exists():
        if len(core_ids) >= 5:
            core_rscu_check = _compute_cluster_rscu(ffn_path, core_ids)
            if not core_rscu_check.empty:
                rscu_cols = [c for c in RSCU_COLUMN_NAMES
                             if c in rp_rscu_df.columns and c in core_rscu_check.index]
                if rscu_cols:
                    rp_mean = rp_rscu_df[rscu_cols].mean()
                    cosine_sim = 1.0 - cosine_dist(
                        core_rscu_check[rscu_cols].fillna(0).values,
                        rp_mean.fillna(0).values,
                    )

    # Composite score
    cos_term = cosine_sim if not np.isnan(cosine_sim) else 0.5
    composite = (_W_JACCARD * mean_jaccard + _W_MEAN_FREQ * mean_core_freq
                 + _W_COSINE * cos_term)

    # Build metrics DataFrame (single row, no multiplier sweep)
    # For backward compatibility, report the chi-squared threshold as
    # "recommended multiplier" equivalent
    chi2_thresh = _chi2_threshold(n_axes, _CLUSTER_CHI2_P)
    rp_dists = _compute_mahalanobis_distances(X, X[rp_indices].mean(axis=0),
                                               np.eye(n_axes))
    median_rp = float(np.median(rp_dists[rp_indices]))
    effective_mult = chi2_thresh / median_rp if median_rp > 0 else 2.0

    metrics_row = {
        "multiplier": round(effective_mult, 2),
        "mean_jaccard": round(mean_jaccard, 4),
        "mean_core_freq": round(mean_core_freq, 4),
        "n_core_genes": n_core,
        "mean_cluster_size": round(mean_size, 1),
        "std_cluster_size": round(std_size, 1),
        "cluster_frac": round(mean_size / n_genes, 4) if n_genes > 0 else 0.0,
        "rp_coverage": round(rp_coverage, 4),
        "cosine_sim_rp": round(cosine_sim, 4) if not np.isnan(cosine_sim) else None,
        "size_score": 1.0,
        "composite_score": round(composite, 4),
    }
    metrics_df = pd.DataFrame([metrics_row])

    logger.info(
        "Stability: Jaccard=%.3f, core=%d genes (%.0f%% RP coverage), composite=%.3f",
        mean_jaccard, n_core, rp_coverage * 100, composite,
    )

    results["recommended_multiplier"] = round(effective_mult, 2)
    results["composite_score"] = round(composite, 4)
    results["core_threshold"] = core_threshold
    results["metrics_df"] = metrics_df

    # ── Per-gene frequency table ───────────────────────────────────
    freq_df = pd.DataFrame({"gene": all_gene_ids})
    rec_col = f"freq_m{effective_mult:.2f}"
    freq_df[rec_col] = freq_df["gene"].map(freq)
    freq_df["is_rp"] = freq_df["gene"].isin(rp_gene_ids)
    freq_df["core_at_recommended"] = freq_df[rec_col] >= core_threshold

    _lo = min(0.1, core_threshold * 0.2)
    _hi = min(core_threshold + (1.0 - core_threshold) * 0.5, 0.99)
    freq_df["stability_class"] = pd.cut(
        freq_df[rec_col],
        bins=[-0.01, _lo, core_threshold, _hi, 1.01],
        labels=["absent", "unstable", "moderate", "stable"],
    )
    results["membership_freq_df"] = freq_df

    # ── Save outputs ───────────────────────────────────────────────
    metrics_path = stab_dir / f"{sample_id}_stability_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    results["metrics_path"] = metrics_path

    freq_path = stab_dir / f"{sample_id}_gene_membership_freq.tsv"
    freq_df.to_csv(freq_path, sep="\t", index=False)
    results["freq_path"] = freq_path

    core_path = stab_dir / f"{sample_id}_core_gene_ids.txt"
    core_path.write_text("\n".join(sorted(core_ids)) + "\n")
    results["core_ids_path"] = core_path
    results["core_gene_ids"] = core_ids

    results["membership_frequencies"] = freq

    # ── Frequency-weighted RSCU ────────────────────────────────────
    if ffn_path and ffn_path.exists() and core_ids:
        try:
            freq_weights = {gid: freq.get(gid, 0.0) for gid in core_ids
                           if freq.get(gid, 0.0) > 0}
            if freq_weights:
                core_rscu = _compute_cluster_rscu(ffn_path, core_ids, gene_weights=freq_weights)
                if not core_rscu.empty:
                    results["core_rscu"] = core_rscu
                    rscu_path = stab_dir / f"{sample_id}_core_rscu.tsv"
                    core_rscu.to_frame("RSCU").to_csv(rscu_path, sep="\t")
                    results["core_rscu_path"] = rscu_path
                    logger.info("Core RSCU: %d genes, mean weight %.3f",
                                len(freq_weights),
                                float(np.mean(list(freq_weights.values()))))
        except Exception as e:
            logger.warning("Frequency-weighted core RSCU failed: %s", e)

    # ── Diagnostic plots ───────────────────────────────────────────
    try:
        _plot_gene_stability_distribution(
            freq_df, effective_mult, core_threshold, stab_dir, sample_id,
        )
        results["stability_distribution_plot"] = (
            stab_dir / f"{sample_id}_stability_distribution.png"
        )
    except Exception as e:
        logger.warning("Stability distribution plot failed: %s", e)

    logger.info(
        "Stability complete for %s: %d core genes (stable in >= %.0f%% of %d bootstraps)",
        sample_id, len(core_ids), core_threshold * 100, n_bootstraps,
    )

    return results


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_gene_stability_distribution(
    freq_df: pd.DataFrame,
    recommended_mult: float,
    core_threshold: float,
    output_dir: Path,
    sample_id: str,
) -> None:
    """Histogram of gene membership frequencies."""
    rec_col = f"freq_m{recommended_mult:.2f}"
    if rec_col not in freq_df.columns:
        return

    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    freqs = freq_df[rec_col].values
    bins = np.linspace(0, 1, 21)
    ax.hist(freqs, bins=bins, color="#2c7bb6", alpha=0.7, edgecolor="white")
    ax.axvline(core_threshold, color="#d7191c", linewidth=1.5, linestyle="--",
               label=f"Core threshold ({core_threshold:.2f})")

    n_core = int((freqs >= core_threshold).sum())
    n_stable = int((freqs >= 0.9).sum())
    ax.set_xlabel(f"Membership frequency (chi2 threshold)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{sample_id}: {n_core} core genes, {n_stable} stable (>= 0.9)")
    ax.legend(fontsize=8)

    out_path = output_dir / f"{sample_id}_stability_distribution"
    for fmt in FORMATS:
        fig.savefig(out_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
