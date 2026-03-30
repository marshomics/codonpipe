"""Bootstrap stability analysis for RP-anchored Mahalanobis clustering.

Quantifies how robust the optimized gene set is to:
    1. Resampling noise in the COA (gene-level bootstrap)
    2. Choice of distance multiplier

For each genome the module:
    - Runs B bootstrap replicates of COA + Mahalanobis clustering
    - Sweeps a grid of distance multipliers
    - Records per-gene membership frequency across replicates
    - Computes cluster-level stability metrics (Jaccard stability,
      mean membership frequency, silhouette on distances)
    - Selects the multiplier that maximises a composite stability score
    - Produces diagnostic plots

The recommended multiplier is exported alongside stability statistics so the
pipeline (or user) can decide whether to override the default.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import chi2

from Bio import SeqIO

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.mahal_clustering import (
    _compute_mahalanobis_distances,
    _distance_to_membership,
    _fit_robust_rp_reference,
    _select_n_axes,
    _compute_cluster_rscu,
    _MAX_COA_AXES,
    _MIN_COA_AXES,
    _MIN_GENES_FOR_CLUSTERING,
    _RP_OUTLIER_ALPHA,
    _MIN_RP_FOR_ROBUST,
)
from codonpipe.modules.rscu import compute_rscu_from_counts, count_codons
from codonpipe.plotting.utils import DPI, FORMATS, STYLE_PARAMS, save_fig
from codonpipe.utils.codon_tables import MIN_GENE_LENGTH, RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_N_BOOTSTRAPS = 100
_DEFAULT_MULTIPLIER_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
_DEFAULT_CORE_THRESHOLD = 0.5   # membership frequency threshold for "core" genes

# Weights for composite stability score
_W_JACCARD = 0.35       # higher = more stable membership across bootstraps
_W_MEAN_FREQ = 0.25     # higher = genes consistently assigned
_W_COSINE = 0.25        # higher = RSCU signal preserved
_W_SIZE_PENALTY = 0.15  # penalise very small or very large clusters


# ---------------------------------------------------------------------------
# Core bootstrap engine
# ---------------------------------------------------------------------------

def _bootstrap_coa_mahal(
    rscu_gene_df: pd.DataFrame,
    rp_gene_ids: set[str],
    multiplier: float,
    expr_df: pd.DataFrame | None = None,
    seed: int = 0,
) -> tuple[set[str], np.ndarray, float]:
    """Run one bootstrap replicate: resample genes, COA, Mahalanobis threshold.

    Resamples genes (rows) with replacement, keeping the same gene IDs so
    that RP anchoring is preserved.  Duplicate rows get unique suffixes to
    avoid pandas index collisions, but the RP lookup uses the original ID.

    Returns:
        cluster_ids: set of *original* gene IDs in the optimized set
        distances: Mahalanobis distances for the original (non-duplicated) genes
        threshold: the Mahalanobis threshold used
    """
    rng = np.random.RandomState(seed)

    n = len(rscu_gene_df)
    idx = rng.choice(n, size=n, replace=True)

    boot_df = rscu_gene_df.iloc[idx].copy()
    # Keep original gene IDs for RP matching
    original_genes = boot_df["gene"].values.copy()
    # Deduplicate index for COA
    boot_df = boot_df.reset_index(drop=True)
    boot_df["gene"] = [f"{g}__boot{i}" for i, g in enumerate(original_genes)]

    # Map RP IDs into bootstrap namespace
    boot_rp_ids = set()
    boot_to_orig = {}
    for i, og in enumerate(original_genes):
        bid = f"{og}__boot{i}"
        boot_to_orig[bid] = og
        if og in rp_gene_ids:
            boot_rp_ids.add(bid)

    # COA
    coa_results = compute_coa_on_rscu(boot_df, expr_df=None)
    if not coa_results or "coa_coords" not in coa_results:
        return set(), np.array([]), 0.0

    coa_coords = coa_results["coa_coords"]
    coa_inertia = coa_results.get("coa_inertia", pd.DataFrame())

    n_axes = _select_n_axes(coa_inertia, _MAX_COA_AXES)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n_axes = len(axis_cols)
    if n_axes < _MIN_COA_AXES:
        return set(), np.array([]), 0.0

    gene_ids = coa_coords["gene"].astype(str).tolist()
    X = coa_coords[axis_cols].values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    gene_ids = [g for g, v in zip(gene_ids, valid_mask) if v]

    # RP indices in bootstrap space
    rp_indices = np.array([i for i, g in enumerate(gene_ids) if g in boot_rp_ids])
    if len(rp_indices) < 3:
        return set(), np.array([]), 0.0

    X_rp = X[rp_indices]
    centroid, cov, cov_inv, rp_outlier_mask = _fit_robust_rp_reference(
        X_rp, n_axes,
        alpha=_RP_OUTLIER_ALPHA,
        min_rp=_MIN_RP_FOR_ROBUST,
    )

    distances = _compute_mahalanobis_distances(X, centroid, cov_inv)
    rp_dists = distances[rp_indices]
    rp_dists_clean = rp_dists[~rp_outlier_mask]
    if len(rp_dists_clean) == 0:
        rp_dists_clean = rp_dists

    median_rp_dist = float(np.median(rp_dists_clean))
    threshold = multiplier * median_rp_dist

    opt_mask = distances <= threshold
    # Map back to original gene IDs
    cluster_ids = set()
    for gid, inside in zip(gene_ids, opt_mask):
        if inside:
            orig = boot_to_orig.get(gid, gid)
            cluster_ids.add(orig)

    return cluster_ids, distances, threshold


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Multi-multiplier stability sweep
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
) -> dict:
    """Bootstrap stability analysis across a grid of distance multipliers.

    For each multiplier in the grid, runs ``n_bootstraps`` bootstrap
    replicates of the full COA → Mahalanobis pipeline.  Computes:

    - Per-gene membership frequency (fraction of replicates where the
      gene is inside the optimized set)
    - Pairwise Jaccard stability (mean Jaccard between all bootstrap
      pairs at each multiplier)
    - Mean membership frequency of genes above ``core_threshold``
    - RSCU cosine similarity between bootstrap-consensus cluster and RP
    - Composite stability score combining the above

    Selects the multiplier that maximises the composite score.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        output_dir: Output directory for this sample.
        sample_id: Sample identifier.
        ffn_path: Nucleotide CDS FASTA (for RSCU pooling at recommended
            multiplier).
        rp_ids_file: Path to RP gene IDs file.
        rp_rscu_df: Per-gene RSCU for ribosomal proteins (for cosine
            similarity validation).
        expr_df: Optional expression table.
        n_bootstraps: Number of bootstrap replicates per multiplier.
        multiplier_grid: List of multiplier values to test.
        core_threshold: Membership frequency threshold for a gene to be
            considered "core" (default 0.5).  Set to 0.9 for a
            high-confidence subset.

    Returns:
        Dict with stability results, recommended multiplier, per-gene
        membership frequencies, and diagnostic file paths.
    """
    stab_dir = output_dir / "cluster_stability"
    stab_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    if multiplier_grid is None:
        multiplier_grid = list(_DEFAULT_MULTIPLIER_GRID)

    # ── Load RP gene IDs ─────────────────────────────────────────────
    rp_gene_ids: set[str] = set()
    if rp_ids_file and rp_ids_file.exists():
        rp_gene_ids = {
            line.strip()
            for line in rp_ids_file.read_text().splitlines()
            if line.strip()
        }
    if not rp_gene_ids:
        logger.warning(
            "No RP IDs for stability analysis of %s; skipping.", sample_id
        )
        return results

    n_genes = len(rscu_gene_df)
    if n_genes < _MIN_GENES_FOR_CLUSTERING:
        logger.warning(
            "Too few genes (%d) for stability analysis of %s; skipping.",
            n_genes, sample_id,
        )
        return results

    all_gene_ids = rscu_gene_df["gene"].astype(str).tolist()

    # ── Run the reference (non-bootstrap) clustering at each multiplier ──
    # This gives us the "point estimate" cluster for Jaccard comparison.
    logger.info(
        "Stability analysis for %s: %d bootstraps x %d multipliers",
        sample_id, n_bootstraps, len(multiplier_grid),
    )

    # Storage: multiplier → list of cluster_id sets (one per bootstrap)
    boot_clusters: dict[float, list[set[str]]] = {m: [] for m in multiplier_grid}

    for mult in multiplier_grid:
        logger.info("  Multiplier %.2f: running %d bootstraps...", mult, n_bootstraps)
        for b in range(n_bootstraps):
            cluster_ids, _, _ = _bootstrap_coa_mahal(
                rscu_gene_df, rp_gene_ids, mult,
                expr_df=expr_df, seed=b * 1000 + int(mult * 100),
            )
            boot_clusters[mult].append(cluster_ids)

    # ── Compute per-gene membership frequency at each multiplier ─────
    membership_freq: dict[float, dict[str, float]] = {}
    for mult in multiplier_grid:
        freq = {}
        for gid in all_gene_ids:
            count = sum(1 for cluster in boot_clusters[mult] if gid in cluster)
            freq[gid] = count / n_bootstraps
        membership_freq[mult] = freq

    # ── Compute stability metrics per multiplier ─────────────────────
    metrics_rows = []
    for mult in multiplier_grid:
        clusters = boot_clusters[mult]

        # Pairwise Jaccard (subsample pairs for speed if B is large)
        n_pairs = min(500, n_bootstraps * (n_bootstraps - 1) // 2)
        rng = np.random.RandomState(42)
        jaccards = []
        for _ in range(n_pairs):
            i, j = rng.choice(n_bootstraps, size=2, replace=False)
            jaccards.append(_jaccard(clusters[i], clusters[j]))
        mean_jaccard = float(np.mean(jaccards))

        # Mean membership frequency (all genes)
        freq = membership_freq[mult]
        all_freqs = np.array(list(freq.values()))

        # "Core" genes: present in ≥core_threshold of bootstraps
        core_mask = all_freqs >= core_threshold
        n_core = int(core_mask.sum())
        mean_core_freq = float(np.mean(all_freqs[core_mask])) if n_core > 0 else 0.0

        # Mean cluster size across bootstraps
        sizes = [len(c) for c in clusters]
        mean_size = float(np.mean(sizes))
        std_size = float(np.std(sizes))

        # RP coverage: fraction of RPs in the core set
        rp_in_core = sum(1 for g in rp_gene_ids if freq.get(g, 0) >= core_threshold)
        rp_coverage = rp_in_core / len(rp_gene_ids) if rp_gene_ids else 0.0

        # RSCU cosine similarity of core genes vs RP-only RSCU
        cosine_sim = np.nan
        if rp_rscu_df is not None and not rp_rscu_df.empty and ffn_path and ffn_path.exists():
            core_ids = {g for g, f in freq.items() if f >= core_threshold}
            if len(core_ids) >= 5:
                core_rscu = _compute_cluster_rscu(ffn_path, core_ids)
                if not core_rscu.empty:
                    rscu_cols = [
                        c for c in RSCU_COLUMN_NAMES
                        if c in rp_rscu_df.columns and c in core_rscu.index
                    ]
                    if rscu_cols:
                        rp_mean = rp_rscu_df[rscu_cols].mean()
                        cosine_sim = 1.0 - cosine_dist(
                            core_rscu[rscu_cols].fillna(0).values,
                            rp_mean.fillna(0).values,
                        )

        # Size penalty: peaks when cluster is ~5-15% of genome,
        # penalises <1% or >30%.
        frac = mean_size / n_genes if n_genes > 0 else 0
        if frac < 0.01:
            size_score = frac / 0.01  # linear ramp 0→1
        elif frac <= 0.20:
            size_score = 1.0
        elif frac <= 0.40:
            size_score = 1.0 - (frac - 0.20) / 0.20  # linear decline
        else:
            size_score = 0.0

        # Composite stability score
        cos_term = cosine_sim if not np.isnan(cosine_sim) else 0.8  # default if unavailable
        composite = (
            _W_JACCARD * mean_jaccard
            + _W_MEAN_FREQ * mean_core_freq
            + _W_COSINE * cos_term
            + _W_SIZE_PENALTY * size_score
        )

        metrics_rows.append({
            "multiplier": mult,
            "mean_jaccard": round(mean_jaccard, 4),
            "mean_core_freq": round(mean_core_freq, 4),
            "n_core_genes": n_core,
            "mean_cluster_size": round(mean_size, 1),
            "std_cluster_size": round(std_size, 1),
            "cluster_frac": round(frac, 4),
            "rp_coverage": round(rp_coverage, 4),
            "cosine_sim_rp": round(cosine_sim, 4) if not np.isnan(cosine_sim) else None,
            "size_score": round(size_score, 4),
            "composite_score": round(composite, 4),
        })

    metrics_df = pd.DataFrame(metrics_rows)

    # ── Select recommended multiplier ────────────────────────────────
    best_idx = metrics_df["composite_score"].idxmax()
    recommended_mult = float(metrics_df.loc[best_idx, "multiplier"])
    best_score = float(metrics_df.loc[best_idx, "composite_score"])

    logger.info(
        "Stability analysis recommends multiplier=%.2f (composite=%.4f) for %s",
        recommended_mult, best_score, sample_id,
    )
    results["recommended_multiplier"] = recommended_mult
    results["composite_score"] = best_score
    results["core_threshold"] = core_threshold
    results["metrics_df"] = metrics_df

    # ── Build per-gene membership frequency table ────────────────────
    freq_df = pd.DataFrame({"gene": all_gene_ids})
    for mult in multiplier_grid:
        freq_df[f"freq_m{mult:.2f}"] = freq_df["gene"].map(membership_freq[mult])
    freq_df["is_rp"] = freq_df["gene"].isin(rp_gene_ids)

    # Core membership at recommended multiplier
    rec_col = f"freq_m{recommended_mult:.2f}"
    freq_df["core_at_recommended"] = freq_df[rec_col] >= core_threshold

    # Stability classes adapt to the chosen core_threshold.  The bins
    # always produce four categories: absent (< low), unstable (low ..
    # core_threshold), moderate (core_threshold .. high), stable (>= high).
    _lo = min(0.1, core_threshold * 0.2)
    _hi = min(core_threshold + (1.0 - core_threshold) * 0.5, 0.99)
    freq_df["stability_class"] = pd.cut(
        freq_df[rec_col],
        bins=[-0.01, _lo, core_threshold, _hi, 1.01],
        labels=["absent", "unstable", "moderate", "stable"],
    )
    results["membership_freq_df"] = freq_df

    # ── Save outputs ─────────────────────────────────────────────────
    metrics_path = stab_dir / f"{sample_id}_stability_metrics.tsv"
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    results["metrics_path"] = metrics_path

    freq_path = stab_dir / f"{sample_id}_gene_membership_freq.tsv"
    freq_df.to_csv(freq_path, sep="\t", index=False)
    results["freq_path"] = freq_path

    # Core gene IDs at recommended multiplier
    core_ids = set(freq_df.loc[freq_df["core_at_recommended"], "gene"])
    core_path = stab_dir / f"{sample_id}_core_gene_ids.txt"
    core_path.write_text("\n".join(sorted(core_ids)) + "\n")
    results["core_ids_path"] = core_path
    results["core_gene_ids"] = core_ids

    # ── Diagnostic plots ─────────────────────────────────────────────
    try:
        _plot_stability_metrics(metrics_df, stab_dir, sample_id)
        results["stability_metrics_plot"] = stab_dir / f"{sample_id}_stability_metrics.png"
    except Exception as e:
        logger.warning("Stability metrics plot failed: %s", e)

    try:
        _plot_membership_heatmap(freq_df, multiplier_grid, rp_gene_ids, stab_dir, sample_id)
        results["membership_heatmap"] = stab_dir / f"{sample_id}_membership_heatmap.png"
    except Exception as e:
        logger.warning("Membership heatmap failed: %s", e)

    try:
        _plot_gene_stability_distribution(
            freq_df, recommended_mult, core_threshold, stab_dir, sample_id,
        )
        results["stability_distribution_plot"] = (
            stab_dir / f"{sample_id}_stability_distribution.png"
        )
    except Exception as e:
        logger.warning("Stability distribution plot failed: %s", e)

    logger.info(
        "Stability analysis complete for %s: recommended multiplier=%.2f, "
        "%d core genes (stable in ≥%.0f%% of %d bootstraps)",
        sample_id, recommended_mult, len(core_ids),
        core_threshold * 100, n_bootstraps,
    )

    return results


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_stability_metrics(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
) -> None:
    """Multi-panel line plot of stability metrics vs multiplier."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{sample_id}: cluster stability across multipliers", fontsize=13)

    x = metrics_df["multiplier"]

    panels = [
        ("composite_score", "Composite score", "#2c7bb6", axes[0, 0]),
        ("mean_jaccard", "Mean Jaccard stability", "#d7191c", axes[0, 1]),
        ("mean_core_freq", "Mean core frequency", "#fdae61", axes[0, 2]),
        ("mean_cluster_size", "Mean cluster size", "#abd9e9", axes[1, 0]),
        ("rp_coverage", "RP coverage (core set)", "#2ca02c", axes[1, 1]),
    ]

    # Cosine similarity might have None values
    cos_col = "cosine_sim_rp"

    for col, label, color, ax in panels:
        y = metrics_df[col]
        ax.plot(x, y, "o-", color=color, linewidth=2, markersize=6)
        ax.set_xlabel("Distance multiplier")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

        # Highlight recommended
        best_row = metrics_df.loc[metrics_df["composite_score"].idxmax()]
        ax.axvline(best_row["multiplier"], color="gray", linestyle=":", alpha=0.5)

    # Cosine panel (handle None)
    ax_cos = axes[1, 2]
    cos_vals = metrics_df[cos_col].apply(lambda v: float(v) if v is not None else np.nan)
    valid = ~cos_vals.isna()
    if valid.any():
        ax_cos.plot(x[valid], cos_vals[valid], "o-", color="#7570b3", linewidth=2, markersize=6)
    ax_cos.set_xlabel("Distance multiplier")
    ax_cos.set_ylabel("Cosine sim (core vs RP)")
    ax_cos.grid(alpha=0.3)
    best_row = metrics_df.loc[metrics_df["composite_score"].idxmax()]
    ax_cos.axvline(best_row["multiplier"], color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / f"{sample_id}_stability_metrics"
    for fmt in FORMATS:
        fig.savefig(out_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_membership_heatmap(
    freq_df: pd.DataFrame,
    multiplier_grid: list[float],
    rp_gene_ids: set[str],
    output_dir: Path,
    sample_id: str,
    max_genes: int = 200,
) -> None:
    """Heatmap of per-gene membership frequency across multipliers.

    Shows the top ``max_genes`` genes ranked by mean frequency.  RP
    genes are marked with a side colour bar.
    """
    import seaborn as sns

    freq_cols = [f"freq_m{m:.2f}" for m in multiplier_grid]
    present_cols = [c for c in freq_cols if c in freq_df.columns]
    if not present_cols:
        return

    heat_df = freq_df[["gene"] + present_cols].copy()
    heat_df["mean_freq"] = heat_df[present_cols].mean(axis=1)
    heat_df = heat_df.sort_values("mean_freq", ascending=False).head(max_genes)
    heat_df = heat_df.set_index("gene")

    plot_data = heat_df[present_cols]
    plot_data.columns = [f"{m:.2f}" for m in multiplier_grid if f"freq_m{m:.2f}" in present_cols]

    # Row colours: RP vs non-RP
    row_colors = pd.Series(
        ["#d7191c" if g in rp_gene_ids else "#cccccc" for g in heat_df.index],
        index=heat_df.index,
        name="RP",
    )

    plt.rcParams.update(STYLE_PARAMS)
    g = sns.clustermap(
        plot_data,
        cmap="YlGnBu",
        vmin=0, vmax=1,
        row_cluster=True,
        col_cluster=False,
        row_colors=row_colors,
        yticklabels=False,
        figsize=(6, max(4, min(12, max_genes // 15))),
        cbar_kws={"label": "Membership frequency"},
    )
    g.ax_heatmap.set_xlabel("Distance multiplier")
    g.fig.suptitle(
        f"{sample_id}: gene membership stability (top {len(heat_df)} genes)",
        y=1.02, fontsize=12,
    )

    out_path = output_dir / f"{sample_id}_membership_heatmap"
    for fmt in FORMATS:
        g.savefig(out_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(g.fig)


def _plot_gene_stability_distribution(
    freq_df: pd.DataFrame,
    recommended_mult: float,
    core_threshold: float,
    output_dir: Path,
    sample_id: str,
) -> None:
    """Histogram of gene membership frequencies at the recommended multiplier."""
    rec_col = f"freq_m{recommended_mult:.2f}"
    if rec_col not in freq_df.columns:
        return

    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    freqs = freq_df[rec_col].values
    bins = np.linspace(0, 1, 21)
    ax.hist(freqs, bins=bins, color="#2c7bb6", alpha=0.7, edgecolor="white")

    # Mark the core threshold
    ax.axvline(
        core_threshold, color="#d7191c", linewidth=1.5, linestyle="--",
        label=f"Core threshold ({core_threshold:.2f})",
    )

    n_core = int((freqs >= core_threshold).sum())
    n_stable = int((freqs >= 0.9).sum())
    ax.set_xlabel(f"Membership frequency (multiplier={recommended_mult:.2f})")
    ax.set_ylabel("Number of genes")
    ax.set_title(
        f"{sample_id}: {n_core} core genes (≥{core_threshold:.2f}), "
        f"{n_stable} stable (≥0.9)"
    )
    ax.legend(fontsize=8)

    out_path = output_dir / f"{sample_id}_stability_distribution"
    for fmt in FORMATS:
        fig.savefig(out_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
