"""RP-anchored Mahalanobis distance module for translational optimization classification.

Replaces GMM-based clustering with a deterministic, RP-anchored approach:

1. Correspondence Analysis (COA) on per-gene RSCU matrix
2. Robust covariance estimation (MinCovDet) on RP gene coordinates
3. Two-pass outlier removal to exclude atypical RP genes
4. Mahalanobis distance from cleaned RP centroid to all genes
5. Threshold at 2x median RP distance to define the optimized gene set
6. Compute RSCU from the optimized set via concatenated codon pooling

Advantages over GMM:
- Deterministic: no k selection, no random initialization
- Stable cluster size: threshold is anchored to RP distribution, not BIC
- Excludes atypical RP genes that would dilute the reference signal
- Continuous membership scores via distance, not hard cluster labels
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

from Bio import SeqIO

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.rscu import compute_rscu_from_counts, count_codons
from codonpipe.plotting.utils import DPI, FORMATS, STYLE_PARAMS, apply_style, save_fig
from codonpipe.utils.codon_tables import MIN_GENE_LENGTH, RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_GENES_FOR_CLUSTERING = 50       # Minimum genes to attempt clustering
_MIN_COA_AXES = 2             # Minimum COA axes required
_MAX_COA_AXES = 8             # Maximum COA axes to use
_CUMULATIVE_INERTIA_TARGET = 0.80  # Target cumulative inertia for axis selection
_MIN_CLUSTER_SIZE = 10        # Warn if optimized set has fewer genes than this

# RP-anchored Mahalanobis approach
_RP_OUTLIER_ALPHA = 0.025     # Chi-squared alpha for RP outlier detection
_DISTANCE_MULTIPLIER = 2.0    # Threshold = multiplier x median RP Mahalanobis distance
_MIN_RP_FOR_ROBUST = 10       # Min RP genes for MinCovDet; else empirical covariance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_n_axes(coa_inertia: pd.DataFrame, max_axes: int = _MAX_COA_AXES) -> int:
    """Select number of COA axes to retain based on cumulative inertia.

    Retains axes until cumulative inertia exceeds the target threshold,
    with a minimum of 2 and a maximum of ``max_axes``.
    """
    if coa_inertia.empty:
        return _MIN_COA_AXES

    cum_pct = coa_inertia["cum_pct"].values
    # cum_pct is in percentage (0-100)
    target = _CUMULATIVE_INERTIA_TARGET * 100

    n = _MIN_COA_AXES
    for i, cp in enumerate(cum_pct):
        n = i + 1
        if cp >= target:
            break

    return max(_MIN_COA_AXES, min(n, max_axes, len(cum_pct)))


def _fit_robust_rp_reference(
    X_rp: np.ndarray,
    n_axes: int,
    alpha: float = _RP_OUTLIER_ALPHA,
    min_rp: int = _MIN_RP_FOR_ROBUST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit robust covariance on RP genes with two-pass outlier removal.

    Pass 1: Fit MinCovDet on all RP genes, identify outliers beyond
            chi-squared critical value.
    Pass 2: Refit centroid and covariance excluding outliers.

    Args:
        X_rp: (n_rp_genes, n_axes) RP gene COA coordinates.
        n_axes: Number of COA axes (for chi-squared df).
        alpha: Significance level for chi-squared outlier detection.
        min_rp: Minimum RP genes to use MinCovDet; falls back to
                empirical covariance if fewer.

    Returns:
        centroid: (n_axes,) cleaned RP centroid.
        cov: (n_axes, n_axes) covariance matrix.
        cov_inv: (n_axes, n_axes) inverse covariance matrix.
        rp_outlier_mask: boolean array over RP genes, True for outliers.
    """
    n_rp = len(X_rp)

    def _safe_inv(mat):
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            logger.warning("Singular covariance; using pseudo-inverse")
            return np.linalg.pinv(mat)

    def _empirical_cov(X):
        c = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1])
        if c.ndim == 0:
            c = np.array([[float(c)]])
        elif c.ndim == 1:
            c = c.reshape(1, 1)
        return c

    if n_rp < min_rp:
        logger.warning(
            "Only %d RP genes; falling back to empirical covariance (< %d)",
            n_rp, min_rp,
        )
        centroid = X_rp.mean(axis=0)
        cov = _empirical_cov(X_rp)
        rp_outlier_mask = np.zeros(n_rp, dtype=bool)
        return centroid, cov, _safe_inv(cov), rp_outlier_mask

    # Pass 1: Fit MinCovDet to identify outliers
    try:
        support_frac = max(0.5, min(0.9, (n_rp - 2) / n_rp))
        mcd = MinCovDet(random_state=42, support_fraction=support_frac).fit(X_rp)
        centroid_1 = mcd.location_
        cov_1 = mcd.covariance_
    except Exception as e:
        logger.warning("MinCovDet failed (%s); using empirical covariance", e)
        centroid_1 = X_rp.mean(axis=0)
        cov_1 = _empirical_cov(X_rp)

    cov_1_inv = _safe_inv(cov_1)

    # Mahalanobis distances for RP genes
    rp_dists = np.array([
        np.sqrt(max(0.0, (x - centroid_1) @ cov_1_inv @ (x - centroid_1)))
        for x in X_rp
    ])

    chi2_crit = np.sqrt(chi2.ppf(1 - alpha, df=n_axes))
    rp_outlier_mask = rp_dists > chi2_crit

    n_outliers = int(rp_outlier_mask.sum())
    logger.info(
        "Pass 1: %d/%d RP outliers (Mahalanobis > %.2f, alpha=%.3f)",
        n_outliers, n_rp, chi2_crit, alpha,
    )

    # Pass 2: Refit excluding outliers
    X_rp_clean = X_rp[~rp_outlier_mask]
    if len(X_rp_clean) < max(2, n_axes + 1):
        logger.warning(
            "Too few non-outlier RP genes (%d); using all RP genes",
            len(X_rp_clean),
        )
        X_rp_clean = X_rp
        rp_outlier_mask = np.zeros(n_rp, dtype=bool)

    centroid = X_rp_clean.mean(axis=0)
    cov = _empirical_cov(X_rp_clean)
    cov_inv = _safe_inv(cov)

    logger.info(
        "Pass 2: centroid computed from %d non-outlier RP genes",
        len(X_rp_clean),
    )

    return centroid, cov, cov_inv, rp_outlier_mask


def _compute_mahalanobis_distances(
    X: np.ndarray,
    centroid: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Compute Mahalanobis distance from each gene to RP centroid.

    Args:
        X: (n_genes, n_features) COA coordinates.
        centroid: (n_features,) RP centroid.
        cov_inv: (n_features, n_features) inverse covariance matrix.

    Returns:
        (n_genes,) Mahalanobis distances.
    """
    diff = X - centroid
    # d = sqrt(diff @ cov_inv @ diff^T) per row
    left = diff @ cov_inv
    d_sq = np.sum(left * diff, axis=1)
    d_sq = np.maximum(d_sq, 0.0)  # numerical safety
    return np.sqrt(d_sq)


def _distance_to_membership(
    distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert Mahalanobis distances to soft membership scores.

    Uses sigmoid: score = 1 / (1 + exp((d - threshold) / scale))
    where scale = threshold / 5 for a smooth but fairly sharp transition.

    Returns:
        (n_genes, 2) array: col 0 = P(non-optimized), col 1 = P(optimized).
    """
    scale = max(threshold / 5.0, 1e-6)
    z = (distances - threshold) / scale
    z = np.clip(z, -500, 500)  # prevent overflow
    score = 1.0 / (1.0 + np.exp(z))
    return np.column_stack([1.0 - score, score])


def _compute_cluster_rscu(
    ffn_path: Path,
    cluster_gene_ids: set[str],
    min_length: int = MIN_GENE_LENGTH,
    gene_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute RSCU by distance-weighted pooling of codon counts.

    Each gene's raw codon counts are multiplied by its proximity weight
    (1 − distance/threshold) before pooling, so genes closer to the RP
    centroid contribute more to the final RSCU estimate.  Genes at the
    centroid contribute their full codon counts; genes at the threshold
    boundary contribute near-zero.

    When *gene_weights* is None (e.g. called from external code that
    doesn't have distances), all genes contribute equally (weight 1.0).

    Args:
        ffn_path: Path to the nucleotide CDS FASTA containing all genes.
        cluster_gene_ids: Set of gene IDs belonging to the target cluster.
        min_length: Minimum sequence length (nt) to include.
        gene_weights: Dict mapping gene ID → weight in (0, 1].
            Internally always provided by run_mahal_clustering.
            External callers may omit for equal-weight fallback.

    Returns:
        Series indexed by RSCU column names (e.g. 'Phe-UUU') with pooled
        RSCU values.
    """
    total_counts: Counter = Counter()
    n_included = 0

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        if rec.id not in cluster_gene_ids:
            continue
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        gene_counts = count_codons(seq)
        if gene_weights is not None:
            w = gene_weights.get(rec.id, 0.0)
            if w <= 0:
                continue
            gene_counts = Counter({k: v * w for k, v in gene_counts.items()})
        total_counts += gene_counts
        n_included += 1

    if not total_counts:
        logger.warning(
            "No codons pooled from cluster (%d IDs requested, %d passed length filter)",
            len(cluster_gene_ids), n_included,
        )
        return pd.Series(dtype=float)

    rscu_dict = compute_rscu_from_counts(total_counts)
    mode = "distance-weighted" if gene_weights is not None else "equal-weight"
    logger.info(
        "Cluster RSCU computed by %s pooling (%d genes, %.0f effective codons)",
        mode, n_included, sum(total_counts.values()),
    )

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_dict]
    return pd.Series({c: rscu_dict[c] for c in rscu_cols})




# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_coa_mahalanobis(
    coa_coords: pd.DataFrame,
    distances: np.ndarray,
    gene_ids: list[str],
    rp_gene_ids: set[str],
    rp_indices: np.ndarray,
    rp_outlier_mask: np.ndarray,
    threshold: float,
    centroid: np.ndarray,
    cov: np.ndarray,
    output_path: Path,
    sample_id: str,
    inertia_pcts: tuple[float, float] = (0.0, 0.0),
) -> None:
    """COA scatter colored by RSCU weight (1 - d/threshold), with threshold ellipse.

    Genes inside the threshold are colored on a blue-to-yellow gradient
    representing their contribution weight to the pooled RSCU estimate
    (1.0 at the centroid, 0.0 at the boundary).  Genes outside the
    threshold are shown in pale gray.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(8, 6))

    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values

    # Compute per-gene weight: 1 - d/threshold, clipped to [0, 1]
    weights = np.clip(1.0 - distances / threshold, 0.0, 1.0) if threshold > 0 else np.ones_like(distances)
    opt_mask = distances <= threshold

    # Non-optimized genes in pale gray
    ax.scatter(
        x[~opt_mask], y[~opt_mask],
        c="#d0d0d0", alpha=0.25, s=8, edgecolors="none", rasterized=True,
        label=f"Non-optimized (n={int((~opt_mask).sum())})",
    )

    # Optimized genes colored by weight
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sc = ax.scatter(
        x[opt_mask], y[opt_mask],
        c=weights[opt_mask], cmap="YlGnBu", norm=norm,
        alpha=0.7, s=14, edgecolors="none", rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("RSCU weight (1 − d/threshold)", fontsize=9)

    # Mark RP genes
    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            x[rp_idx], y[rp_idx],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    # Mark RP outliers in red
    rp_idx_arr = np.array(rp_idx)
    if rp_outlier_mask.any() and len(rp_idx_arr) == len(rp_outlier_mask):
        outlier_idx = rp_idx_arr[rp_outlier_mask]
        if len(outlier_idx) > 0:
            ax.scatter(
                x[outlier_idx], y[outlier_idx],
                marker="x", color="#d7191c", s=50, linewidths=1.5,
                label=f"RP outliers (n={len(outlier_idx)})", zorder=6,
            )

    # Draw threshold ellipse on axes 1 & 2
    try:
        cov_2d = cov[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        if np.all(eigvals > 0):
            angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
            width = 2 * threshold * np.sqrt(eigvals[1])
            height = 2 * threshold * np.sqrt(eigvals[0])
            ellipse = mpatches.Ellipse(
                (centroid[0], centroid[1]), width, height, angle=angle,
                linewidth=1.5, edgecolor="#d7191c", facecolor="none",
                linestyle="--", alpha=0.7, label=f"Threshold (d={threshold:.1f})",
            )
            ax.add_patch(ellipse)
    except Exception:
        pass

    # RP centroid
    ax.scatter(
        centroid[0], centroid[1], marker="*", s=200,
        color="#d7191c", edgecolors="black", linewidths=0.5,
        zorder=7, label="RP centroid",
    )

    pct1, pct2 = inertia_pcts
    ax.set_xlabel(f"COA Axis 1 ({pct1:.1f}% inertia)")
    ax.set_ylabel(f"COA Axis 2 ({pct2:.1f}% inertia)")
    ax.set_title(f"{sample_id}: distance-weighted RSCU contribution")
    ax.legend(fontsize=7, framealpha=0.7, loc="best")

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_distance_histogram(
    distances: np.ndarray,
    rp_distances: np.ndarray,
    threshold: float,
    output_path: Path,
    sample_id: str,
) -> None:
    """Histogram of Mahalanobis distances with weight function overlay.

    Left y-axis: gene counts (all genes in blue, RP genes in red).
    Right y-axis: RSCU weight curve (1 − d/threshold), showing how
    much each gene contributes to the pooled RSCU at its distance.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_max = min(np.percentile(distances, 99), threshold * 4)
    bins = np.linspace(0, x_max, 60)
    ax.hist(distances, bins=bins, color="#2c7bb6", alpha=0.45, label="All genes")
    ax.hist(rp_distances, bins=bins, color="#d7191c", alpha=0.55, label="RP genes")
    ax.axvline(threshold, color="#fdae61", linewidth=2, linestyle="--",
               label=f"Threshold (d={threshold:.2f})")
    ax.axvline(np.median(rp_distances), color="#abd9e9", linewidth=1.5,
               linestyle=":", label=f"Median RP (d={np.median(rp_distances):.2f})")

    n_opt = (distances <= threshold).sum()
    ax.set_xlabel("Mahalanobis distance from RP centroid")
    ax.set_ylabel("Number of genes")

    # Weight function on secondary y-axis
    ax2 = ax.twinx()
    d_line = np.linspace(0, x_max, 300)
    w_line = np.clip(1.0 - d_line / threshold, 0.0, 1.0) if threshold > 0 else np.ones_like(d_line)
    ax2.plot(d_line, w_line, color="#2ca02c", linewidth=2, alpha=0.85, label="RSCU weight")
    ax2.fill_between(d_line, 0, w_line, color="#2ca02c", alpha=0.08)
    ax2.set_ylabel("RSCU weight (1 − d/threshold)", color="#2ca02c", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(-0.05, 1.1)

    ax.set_title(f"{sample_id}: distance distribution ({n_opt} genes in optimized set)")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, framealpha=0.7, loc="upper right")

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_rp_outlier_detection(
    coa_coords: pd.DataFrame,
    rp_gene_ids: set[str],
    gene_ids: list[str],
    rp_outlier_mask: np.ndarray,
    rp_indices: np.ndarray,
    chi2_crit: float,
    output_path: Path,
    sample_id: str,
) -> None:
    """Scatter of RP genes in COA space with outliers highlighted."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(6, 5))

    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values

    # All genes, pale background
    ax.scatter(x, y, c="#cccccc", alpha=0.15, s=6, edgecolors="none", rasterized=True)

    # RP genes
    rp_idx = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids])
    if len(rp_idx) == 0:
        plt.close(fig)
        return

    if len(rp_idx) == len(rp_outlier_mask):
        non_outlier_rp = rp_idx[~rp_outlier_mask]
        outlier_rp = rp_idx[rp_outlier_mask]
    else:
        logger.warning(
            "RP index / outlier mask shape mismatch (%d vs %d); "
            "using all RP genes with no outlier removal",
            len(rp_idx), len(rp_outlier_mask),
        )
        non_outlier_rp = rp_idx
        outlier_rp = np.array([])

    ax.scatter(
        x[non_outlier_rp], y[non_outlier_rp],
        c="#2c7bb6", s=30, alpha=0.8, edgecolors="black", linewidths=0.5,
        label=f"RP non-outlier (n={len(non_outlier_rp)})",
    )
    if len(outlier_rp) > 0:
        ax.scatter(
            x[outlier_rp], y[outlier_rp],
            marker="x", c="#d7191c", s=60, linewidths=1.5,
            label=f"RP outlier (n={len(outlier_rp)})", zorder=5,
        )

    ax.set_xlabel("COA Axis 1")
    ax.set_ylabel("COA Axis 2")
    ax.set_title(
        f"{sample_id}: RP outlier detection "
        f"(chi2 crit={chi2_crit:.2f}, alpha={_RP_OUTLIER_ALPHA})"
    )
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_mahal_clustering(
    rscu_gene_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    ffn_path: Path | None = None,
    rp_ids_file: Path | None = None,
    rp_rscu_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    min_k: int = 2,
    max_k: int = 8,
) -> dict:
    """RP-anchored Mahalanobis distance clustering for translational optimization.

    Replaces GMM-based clustering. Uses the known RP genes as an anchor to
    define a Mahalanobis distance threshold in COA space, then classifies
    all genes within that threshold as translationally optimized.

    Steps:
        1. Compute COA on per-gene RSCU (reuses advanced_analyses.compute_coa_on_rscu)
        2. Select top COA axes by cumulative inertia
        3. Fit robust covariance on RP genes (MinCovDet + two-pass outlier removal)
        4. Compute Mahalanobis distance from cleaned RP centroid to all genes
        5. Set threshold at 2x median RP Mahalanobis distance
        6. Define optimized gene set as all genes within threshold
        7. Compute mean RSCU from optimized set via concatenated pooling
        8. Export gene IDs and diagnostic plots

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        ffn_path: Path to nucleotide CDS FASTA (for concatenated RSCU pooling).
        rp_ids_file: Path to text file with ribosomal protein gene IDs.
        rp_rscu_df: Per-gene RSCU for ribosomal proteins (optional, for
            validation only).
        expr_df: Optional expression table (passed to COA for tier annotation).
        min_k: Unused (kept for signature compatibility).
        max_k: Unused (kept for signature compatibility).

    Returns:
        Dict with:
            - 'mahal_cluster_gene_ids': set[str] — gene IDs in the optimized set
            - 'mahal_cluster_rscu': pd.Series — pooled RSCU of optimized set
            - 'mahal_labels': np.ndarray — binary labels (1=optimized, 0=not)
            - 'mahal_probabilities': np.ndarray — (n_genes, 2) membership scores
            - 'mahal_rp_cluster': int — always 1 (the optimized label)
            - 'mahal_best_k': int — always 2 (optimized vs non-optimized)
            - 'mahal_bic_scores': list[float] — empty (no BIC)
            - 'mahal_n_axes': int — number of COA axes used
            - 'mahal_coa_coords': pd.DataFrame — gene coordinates in COA space
            - 'mahal_rp_cosine_sim': float — cosine similarity between
                  optimized-set RSCU and RP-only RSCU
            - File paths for diagnostics and exports
    """
    mahal_dir = output_dir / "mahal_clustering"
    mahal_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ── Load RP gene IDs ──────────────────────────────────────────────
    rp_gene_ids: set[str] = set()
    if rp_ids_file and rp_ids_file.exists():
        rp_gene_ids = {
            line.strip()
            for line in rp_ids_file.read_text().splitlines()
            if line.strip()
        }
    if not rp_gene_ids:
        logger.warning(
            "No ribosomal protein IDs available for %s; "
            "RP-anchored clustering cannot proceed. Skipping.",
            sample_id,
        )
        return results

    # ── Validate gene count ───────────────────────────────────────────
    n_genes = len(rscu_gene_df)
    if n_genes < _MIN_GENES_FOR_CLUSTERING:
        logger.warning(
            "Too few genes for clustering (%d < %d) in %s. Skipping.",
            n_genes, _MIN_GENES_FOR_CLUSTERING, sample_id,
        )
        return results

    # ── Step 1: COA ───────────────────────────────────────────────────
    logger.info("RP-anchored clustering: computing COA on %d genes for %s", n_genes, sample_id)
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df=expr_df)

    if not coa_results or "coa_coords" not in coa_results:
        logger.warning("COA failed for %s; skipping clustering", sample_id)
        return results

    coa_coords = coa_results["coa_coords"]
    coa_inertia = coa_results.get("coa_inertia", pd.DataFrame())

    # ── Step 2: Select axes ───────────────────────────────────────────
    n_axes = _select_n_axes(coa_inertia, _MAX_COA_AXES)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n_axes = len(axis_cols)

    if n_axes < _MIN_COA_AXES:
        logger.warning("Insufficient COA axes (%d) for clustering in %s", n_axes, sample_id)
        return results

    gene_ids = coa_coords["gene"].astype(str).tolist()
    X = coa_coords[axis_cols].values

    # Remove rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < _MIN_GENES_FOR_CLUSTERING:
        logger.warning("Too few valid genes after NaN removal (%d) in %s", valid_mask.sum(), sample_id)
        return results
    X = X[valid_mask]
    gene_ids = [g for g, v in zip(gene_ids, valid_mask) if v]

    logger.info(
        "Using %d COA axes (%.1f%% cumulative inertia) for %d genes",
        n_axes,
        coa_inertia["cum_pct"].iloc[n_axes - 1] if len(coa_inertia) >= n_axes else 0,
        len(gene_ids),
    )
    results["mahal_n_axes"] = n_axes

    # ── Step 3: Extract RP gene coordinates ───────────────────────────
    rp_indices = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids])
    n_rp_found = len(rp_indices)

    if n_rp_found < 3:
        logger.warning(
            "Only %d RP genes found in COA space for %s; need >= 3. Skipping.",
            n_rp_found, sample_id,
        )
        return results

    X_rp = X[rp_indices]
    logger.info(
        "Found %d/%d RP genes in COA space for %s",
        n_rp_found, len(rp_gene_ids), sample_id,
    )

    # ── Step 4: Fit robust RP reference ───────────────────────────────
    centroid, cov, cov_inv, rp_outlier_mask = _fit_robust_rp_reference(
        X_rp, n_axes,
        alpha=_RP_OUTLIER_ALPHA,
        min_rp=_MIN_RP_FOR_ROBUST,
    )

    # ── Step 5: Compute Mahalanobis distances for all genes ───────────
    distances = _compute_mahalanobis_distances(X, centroid, cov_inv)

    # Distances for non-outlier RP genes (for threshold computation)
    rp_dists_all = distances[rp_indices]
    rp_dists_clean = rp_dists_all[~rp_outlier_mask]

    if len(rp_dists_clean) == 0:
        rp_dists_clean = rp_dists_all

    median_rp_dist = float(np.median(rp_dists_clean))
    threshold = _DISTANCE_MULTIPLIER * median_rp_dist

    logger.info(
        "Mahalanobis threshold: %.2f x %.2f = %.2f",
        _DISTANCE_MULTIPLIER, median_rp_dist, threshold,
    )

    # ── Step 6: Define optimized gene set ─────────────────────────────
    optimized_mask = distances <= threshold
    cluster_gene_ids = {
        gid for gid, opt in zip(gene_ids, optimized_mask) if opt
    }

    n_cluster = len(cluster_gene_ids)
    n_rp_in_cluster = len(cluster_gene_ids & rp_gene_ids)

    logger.info(
        "Optimized set: %d total genes, %d are RPs, %d are non-RP co-selected genes",
        n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster,
    )

    if n_cluster < _MIN_CLUSTER_SIZE:
        logger.warning(
            "Optimized set is small (%d genes); expression scoring may be unreliable",
            n_cluster,
        )

    results["mahal_cluster_gene_ids"] = cluster_gene_ids
    results["mahal_rp_cluster"] = 1  # optimized = label 1

    # Labels: 1 for optimized, 0 for non-optimized
    labels = optimized_mask.astype(int)
    results["mahal_labels"] = labels
    results["mahal_best_k"] = 2
    results["mahal_bic_scores"] = []

    # Soft membership via sigmoid on distance
    probabilities = _distance_to_membership(distances, threshold)
    results["mahal_probabilities"] = probabilities

    # ── Step 7: Compute distance-weighted RSCU for optimized set ───────
    # Weight = 1 - (distance / threshold), clamped to (0, 1].
    # Genes at the centroid contribute fully; genes at the boundary → ~0.
    gene_weights = {}
    for gid, d in zip(gene_ids, distances):
        if gid in cluster_gene_ids:
            w = max(1.0 - d / threshold, 0.0) if threshold > 0 else 1.0
            if w > 0:
                gene_weights[gid] = w

    if ffn_path and ffn_path.exists():
        cluster_rscu = _compute_cluster_rscu(
            ffn_path, cluster_gene_ids, gene_weights=gene_weights,
        )
    else:
        logger.warning("FFN path not available; falling back to weighted per-gene mean RSCU")
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
        cluster_df = rscu_gene_df[rscu_gene_df["gene"].isin(cluster_gene_ids)].copy()
        if not cluster_df.empty:
            w_arr = cluster_df["gene"].map(gene_weights).fillna(0.0).values
            w_sum = w_arr.sum()
            if w_sum > 0:
                cluster_rscu = (cluster_df[rscu_cols].multiply(w_arr, axis=0).sum() / w_sum)
            else:
                cluster_rscu = cluster_df[rscu_cols].mean()
        else:
            cluster_rscu = pd.Series(dtype=float)
    results["mahal_cluster_rscu"] = cluster_rscu

    # ── Step 8: Quality check — cosine similarity ─────────────────────
    rp_cosine_sim = np.nan
    if rp_rscu_df is not None and not rp_rscu_df.empty:
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rp_rscu_df.columns and c in cluster_rscu.index]
        if rscu_cols:
            rp_only_rscu = rp_rscu_df[rscu_cols].mean()
            rp_cosine_sim = 1.0 - cosine_dist(
                cluster_rscu[rscu_cols].fillna(0).values,
                rp_only_rscu.fillna(0).values,
            )
            logger.info(
                "Optimized-set vs RP-only RSCU cosine similarity: %.4f "
                "(>0.95 = closely matches RP signal; "
                "<0.85 = has diverged, check results)",
                rp_cosine_sim,
            )
    results["mahal_rp_cosine_sim"] = rp_cosine_sim

    # ── Step 9: Save outputs ──────────────────────────────────────────

    # Gene-level assignments
    cluster_df = pd.DataFrame({
        "gene": gene_ids,
        "mahalanobis_distance": distances,
        "membership_score": probabilities[:, 1],
        "in_optimized_set": [gid in cluster_gene_ids for gid in gene_ids],
        "is_ribosomal_protein": [gid in rp_gene_ids for gid in gene_ids],
    })
    cluster_path = mahal_dir / f"{sample_id}_mahal_clusters.tsv"
    cluster_df.to_csv(cluster_path, sep="\t", index=False)
    results["mahal_clusters_path"] = cluster_path

    # Optimized-set RSCU reference
    rscu_path = mahal_dir / f"{sample_id}_mahal_cluster_rscu.tsv"
    cluster_rscu.to_frame("RSCU").to_csv(rscu_path, sep="\t")
    results["mahal_cluster_rscu_path"] = rscu_path

    # Gene IDs (one per line, for expression.py)
    ids_path = mahal_dir / f"{sample_id}_mahal_cluster_ids.txt"
    ids_path.write_text("\n".join(sorted(cluster_gene_ids)) + "\n")
    results["mahal_cluster_ids_path"] = ids_path

    # Summary stats
    mean_weight = float(np.mean([w for w in gene_weights.values()]))
    summary = {
        "sample_id": sample_id,
        "n_genes": len(gene_ids),
        "n_coa_axes": n_axes,
        "best_k": 2,
        "rp_cluster": 1,
        "rp_cluster_size": n_cluster,
        "rp_genes_in_cluster": n_rp_in_cluster,
        "non_rp_in_cluster": n_cluster - n_rp_in_cluster,
        "total_rp_genes": len(rp_gene_ids),
        "rp_outliers_removed": int(rp_outlier_mask.sum()),
        "mahalanobis_threshold": round(threshold, 4),
        "median_rp_distance": round(median_rp_dist, 4),
        "rscu_pooling": "distance-weighted",
        "mean_rscu_weight": round(mean_weight, 4),
        "rp_cosine_similarity": round(rp_cosine_sim, 4) if not np.isnan(rp_cosine_sim) else None,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = mahal_dir / f"{sample_id}_mahal_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    results["mahal_summary_path"] = summary_path

    # COA coordinates with assignments and weights
    rscu_weights = np.clip(1.0 - distances / threshold, 0.0, 1.0) if threshold > 0 else np.ones_like(distances)
    coa_with_assignments = coa_coords.copy()
    assign_merge = pd.DataFrame({
        "gene": gene_ids,
        "mahal_cluster": labels,
        "mahalanobis_distance": distances,
        "rscu_weight": rscu_weights,
    })
    coa_with_assignments = coa_with_assignments.merge(assign_merge, on="gene", how="left")
    results["mahal_coa_coords"] = coa_with_assignments

    # ── Step 10: Diagnostic plots ─────────────────────────────────────
    try:
        inertia_pcts_2 = (0.0, 0.0)
        if len(coa_inertia) >= 2:
            inertia_pcts_2 = (
                float(coa_inertia["pct_inertia"].iloc[0]),
                float(coa_inertia["pct_inertia"].iloc[1]),
            )

        coa_filtered = coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True)

        # COA scatter colored by Mahalanobis distance
        _plot_coa_mahalanobis(
            coa_filtered, distances, gene_ids, rp_gene_ids,
            rp_indices, rp_outlier_mask,
            threshold, centroid, cov,
            mahal_dir / f"{sample_id}_mahal_coa_mahalanobis",
            sample_id,
            inertia_pcts=inertia_pcts_2,
        )
        results["mahal_coa_plot"] = mahal_dir / f"{sample_id}_mahal_coa_mahalanobis.png"

        # Distance histogram
        _plot_distance_histogram(
            distances, rp_dists_all, threshold,
            mahal_dir / f"{sample_id}_mahal_distance_histogram",
            sample_id,
        )
        results["mahal_separation_plot"] = mahal_dir / f"{sample_id}_mahal_distance_histogram.png"

        # RP outlier detection plot
        chi2_crit = np.sqrt(chi2.ppf(1 - _RP_OUTLIER_ALPHA, df=n_axes))
        _plot_rp_outlier_detection(
            coa_filtered, rp_gene_ids, gene_ids, rp_outlier_mask,
            rp_indices, chi2_crit,
            mahal_dir / f"{sample_id}_mahal_rp_outliers",
            sample_id,
        )
        results["mahal_rp_outlier_plot"] = mahal_dir / f"{sample_id}_mahal_rp_outliers.png"

    except Exception as e:
        logger.warning("Diagnostic plot generation failed: %s", e, exc_info=True)

    logger.info(
        "RP-anchored clustering complete for %s: "
        "threshold=%.2f, optimized set=%d genes (%d RPs + %d non-RP)",
        sample_id, threshold, n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster,
    )

    return results
