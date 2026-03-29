"""GMM-based codon usage clustering module.

Replaces ACE convergence with a statistically principled approach:

1. Correspondence Analysis (COA) on per-gene RSCU matrix
2. Gaussian Mixture Model (GMM) clustering on top COA axes (BIC-selected k)
3. Identify the cluster containing ribosomal proteins
4. Compute RSCU from that cluster as the genome-specific optimised reference
5. Export cluster gene IDs for downstream expression scoring and enrichment

The RP-containing cluster captures translationally optimised genes beyond
just ribosomal proteins, providing a broader and biologically grounded
reference set for CAI/MELP scoring.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.mixture import GaussianMixture

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.rscu import compute_rscu_from_counts, count_codons
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_GENES_FOR_GMM = 50       # Minimum genes to attempt clustering
_MAX_K = 8                    # Maximum number of GMM components to test
_MIN_K = 2                    # Minimum number of GMM components
_MIN_COA_AXES = 2             # Minimum COA axes required
_MAX_COA_AXES = 8             # Maximum COA axes to use for clustering
_CUMULATIVE_INERTIA_TARGET = 0.80  # Target cumulative inertia for axis selection
_MIN_CLUSTER_SIZE = 10        # Warn if RP cluster has fewer genes than this

# Plotting
DPI = 300
FORMATS = ["png", "svg"]

STYLE_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "svg.fonttype": "none",
}


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


def _fit_gmm_bic(X: np.ndarray, min_k: int = _MIN_K, max_k: int = _MAX_K) -> tuple[GaussianMixture, int, list[float]]:
    """Fit GMMs for k in [min_k, max_k] and return the model with lowest BIC.

    Args:
        X: (n_samples, n_features) array of COA coordinates.
        min_k: Minimum number of components.
        max_k: Maximum number of components.

    Returns:
        best_gmm: Fitted GaussianMixture with lowest BIC.
        best_k: Optimal number of components.
        bic_scores: BIC for each k tested.
    """
    # Don't test more components than we have samples / 5
    effective_max = min(max_k, max(min_k, X.shape[0] // 5))

    bic_scores = []
    best_bic = np.inf
    best_gmm = None
    best_k = min_k

    for k in range(min_k, effective_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=5,
            max_iter=300,
            random_state=42,
        )
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_k = k

    return best_gmm, best_k, bic_scores


def _identify_rp_cluster(
    gmm: GaussianMixture,
    labels: np.ndarray,
    gene_ids: list[str],
    rp_gene_ids: set[str],
) -> int:
    """Identify which GMM cluster contains the most ribosomal proteins.

    Args:
        gmm: Fitted GMM.
        labels: Cluster assignments for each gene.
        gene_ids: Gene identifiers matching the rows of the clustering input.
        rp_gene_ids: Set of ribosomal protein gene IDs.

    Returns:
        Cluster label (int) of the RP-containing cluster.
    """
    n_clusters = gmm.n_components
    rp_counts = np.zeros(n_clusters, dtype=int)

    for gid, label in zip(gene_ids, labels):
        if gid in rp_gene_ids:
            rp_counts[label] += 1

    best_cluster = int(np.argmax(rp_counts))
    total_rp_in_cluster = rp_counts[best_cluster]
    total_rp_found = rp_counts.sum()

    logger.info(
        "RP cluster identification: cluster %d contains %d/%d RPs (%.1f%%)",
        best_cluster, total_rp_in_cluster, total_rp_found,
        100 * total_rp_in_cluster / max(total_rp_found, 1),
    )

    if total_rp_found == 0:
        logger.warning("No ribosomal proteins found in any cluster; defaulting to cluster 0")
        return 0

    # Warn if RPs are split across clusters
    for k in range(n_clusters):
        if k != best_cluster and rp_counts[k] > 0:
            logger.info(
                "  cluster %d also contains %d RPs (%.1f%%)",
                k, rp_counts[k], 100 * rp_counts[k] / total_rp_found,
            )

    return best_cluster


def _compute_cluster_rscu(
    rscu_gene_df: pd.DataFrame,
    cluster_gene_ids: set[str],
) -> pd.Series:
    """Compute mean RSCU across genes in the specified cluster.

    Args:
        rscu_gene_df: Per-gene RSCU DataFrame with 'gene' column plus codon columns.
        cluster_gene_ids: Set of gene IDs belonging to the target cluster.

    Returns:
        Series indexed by RSCU column names with mean RSCU values.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    cluster_df = rscu_gene_df[rscu_gene_df["gene"].isin(cluster_gene_ids)]

    if cluster_df.empty:
        logger.warning("No genes matched cluster IDs in RSCU table")
        return pd.Series(dtype=float)

    return cluster_df[rscu_cols].mean()


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _plot_bic_selection(
    bic_scores: list[float],
    best_k: int,
    output_path: Path,
    sample_id: str,
) -> None:
    """Plot BIC vs number of components."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ks = list(range(_MIN_K, _MIN_K + len(bic_scores)))
    ax.plot(ks, bic_scores, "o-", color="#2c7bb6", linewidth=1.5, markersize=5)
    ax.axvline(best_k, color="#d7191c", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Number of components (k)")
    ax.set_ylabel("BIC")
    ax.set_title(f"{sample_id}: GMM model selection")
    ax.set_xticks(ks)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_coa_clusters(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    rp_cluster: int,
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
    inertia_pcts: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Plot COA axes 1 & 2 colored by GMM cluster, with RP genes highlighted."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    genes = coa_coords["gene"].values
    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values

    # Color palette
    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("Set2", n_clusters)
    colors = [cmap(l) for l in labels]

    # Plot all genes, pale
    ax.scatter(x, y, c=colors, alpha=0.35, s=12, edgecolors="none", rasterized=True)

    # Overlay RP-cluster genes with stronger alpha
    rp_mask = labels == rp_cluster
    ax.scatter(
        x[rp_mask], y[rp_mask],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.7, s=18, edgecolors="none", label=f"Cluster {rp_cluster} (RP cluster)",
    )

    # Mark actual RP genes
    rp_idx = [i for i, g in enumerate(genes) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            x[rp_idx], y[rp_idx],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    pct1, pct2 = inertia_pcts
    ax.set_xlabel(f"COA Axis 1 ({pct1:.1f}% inertia)")
    ax.set_ylabel(f"COA Axis 2 ({pct2:.1f}% inertia)")
    ax.set_title(f"{sample_id}: GMM clustering on COA space (k={n_clusters})")
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_coa_clusters_cosine(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    rp_cluster: int,
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
    n_axes: int = 2,
) -> None:
    """COA scatter after cosine transformation (unit-sphere projection).

    Projects each gene's COA coordinate vector onto the unit hypersphere,
    removing magnitude (gene-length) effects and emphasising directional
    similarity in codon preference.  The 2-D scatter shows Axis 1 vs
    Axis 2 of the normalised vectors.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    genes = coa_coords["gene"].values
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    if len(axis_cols) < 2:
        plt.close(fig)
        return

    V = coa_coords[axis_cols].values.copy()
    # L2-normalise each row (project onto unit hypersphere)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms

    x, y = V[:, 0], V[:, 1]

    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("Set2", n_clusters)
    colors = [cmap(l) for l in labels]

    ax.scatter(x, y, c=colors, alpha=0.35, s=12, edgecolors="none", rasterized=True)

    rp_mask = labels == rp_cluster
    ax.scatter(
        x[rp_mask], y[rp_mask],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.7, s=18, edgecolors="none", label=f"Cluster {rp_cluster} (RP cluster)",
    )

    rp_idx = [i for i, g in enumerate(genes) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            x[rp_idx], y[rp_idx],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    ax.set_xlabel("COA Axis 1 (cosine-normalised)")
    ax.set_ylabel("COA Axis 2 (cosine-normalised)")
    ax.set_title(f"{sample_id}: cosine-transformed COA space (k={n_clusters})")
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_coa_clusters_zscore(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    rp_cluster: int,
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
    n_axes: int = 2,
) -> None:
    """COA scatter after z-score standardisation per axis.

    Standardising each axis to zero mean, unit variance makes the spread
    comparable across genomes with different inertia distributions.  This
    view is most useful in batch reports where cluster separation needs
    to be visually comparable between samples.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    genes = coa_coords["gene"].values
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    if len(axis_cols) < 2:
        plt.close(fig)
        return

    V = coa_coords[axis_cols].values.copy()
    # Per-axis z-score: (x - mean) / std
    means = V.mean(axis=0, keepdims=True)
    stds = V.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    V = (V - means) / stds

    x, y = V[:, 0], V[:, 1]

    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("Set2", n_clusters)
    colors = [cmap(l) for l in labels]

    ax.scatter(x, y, c=colors, alpha=0.35, s=12, edgecolors="none", rasterized=True)

    rp_mask = labels == rp_cluster
    ax.scatter(
        x[rp_mask], y[rp_mask],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.7, s=18, edgecolors="none", label=f"Cluster {rp_cluster} (RP cluster)",
    )

    rp_idx = [i for i, g in enumerate(genes) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            x[rp_idx], y[rp_idx],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    ax.set_xlabel("COA Axis 1 (z-score)")
    ax.set_ylabel("COA Axis 2 (z-score)")
    ax.set_title(f"{sample_id}: z-score-standardised COA space (k={n_clusters})")
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_cluster_separation(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    rp_cluster: int,
    output_path: Path,
    sample_id: str,
) -> None:
    """Plot GMM posterior probability for the RP cluster across all genes."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(6, 3.5))

    rp_probs = probabilities[:, rp_cluster]
    ax.hist(rp_probs, bins=50, color="#2c7bb6", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="#d7191c", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(f"P(cluster {rp_cluster} | gene)")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"{sample_id}: RP cluster membership probability")

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gmm_clustering(
    rscu_gene_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    rp_ids_file: Path | None = None,
    rp_rscu_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    min_k: int = _MIN_K,
    max_k: int = _MAX_K,
) -> dict:
    """Run GMM clustering on COA-projected per-gene RSCU.

    Steps:
        1. Compute COA on per-gene RSCU (reuses advanced_analyses.compute_coa_on_rscu)
        2. Select top COA axes by cumulative inertia
        3. Fit GMMs for k in [min_k, max_k], select best by BIC
        4. Assign genes to clusters
        5. Identify which cluster contains the ribosomal proteins
        6. Compute mean RSCU from RP cluster
        7. Export cluster gene IDs and diagnostic plots

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        rp_ids_file: Path to text file with ribosomal protein gene IDs.
        rp_rscu_df: Per-gene RSCU for ribosomal proteins (optional, for
            validation only).
        expr_df: Optional expression table (passed to COA for tier annotation).
        min_k: Minimum GMM components to test.
        max_k: Maximum GMM components to test.

    Returns:
        Dict with:
            - 'gmm_cluster_gene_ids': set[str] — gene IDs in the RP cluster
            - 'gmm_cluster_rscu': pd.Series — mean RSCU of RP cluster
            - 'gmm_labels': np.ndarray — cluster label per gene
            - 'gmm_probabilities': np.ndarray — posterior probabilities (n_genes, k)
            - 'gmm_rp_cluster': int — which cluster is the RP cluster
            - 'gmm_best_k': int — optimal number of components
            - 'gmm_bic_scores': list[float] — BIC for each k tested
            - 'gmm_n_axes': int — number of COA axes used
            - 'gmm_coa_coords': pd.DataFrame — gene coordinates in COA space
            - 'gmm_rp_cosine_sim': float — cosine similarity between
                  RP-cluster RSCU and RP-only RSCU (quality check)
            - File paths for diagnostics and plots
    """
    gmm_dir = output_dir / "gmm_clustering"
    gmm_dir.mkdir(parents=True, exist_ok=True)
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
            "GMM clustering cannot identify the RP cluster. Skipping.",
            sample_id,
        )
        return results

    # ── Validate gene count ───────────────────────────────────────────
    n_genes = len(rscu_gene_df)
    if n_genes < _MIN_GENES_FOR_GMM:
        logger.warning(
            "Too few genes for GMM clustering (%d < %d) in %s. Skipping.",
            n_genes, _MIN_GENES_FOR_GMM, sample_id,
        )
        return results

    # ── Step 1: COA ───────────────────────────────────────────────────
    logger.info("GMM clustering: computing COA on %d genes for %s", n_genes, sample_id)
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df=expr_df)

    if not coa_results or "coa_coords" not in coa_results:
        logger.warning("COA failed for %s; skipping GMM clustering", sample_id)
        return results

    coa_coords = coa_results["coa_coords"]
    coa_inertia = coa_results.get("coa_inertia", pd.DataFrame())

    # ── Step 2: Select axes ───────────────────────────────────────────
    n_axes = _select_n_axes(coa_inertia, _MAX_COA_AXES)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n_axes = len(axis_cols)

    if n_axes < _MIN_COA_AXES:
        logger.warning("Insufficient COA axes (%d) for GMM clustering in %s", n_axes, sample_id)
        return results

    X = coa_coords[axis_cols].values
    gene_ids = coa_coords["gene"].astype(str).tolist()

    # Remove any rows with NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < _MIN_GENES_FOR_GMM:
        logger.warning("Too few valid genes after NaN removal (%d) in %s", valid_mask.sum(), sample_id)
        return results
    X = X[valid_mask]
    gene_ids = [g for g, v in zip(gene_ids, valid_mask) if v]

    logger.info(
        "GMM clustering: using %d COA axes (%.1f%% cumulative inertia) for %d genes",
        n_axes,
        coa_inertia["cum_pct"].iloc[n_axes - 1] if len(coa_inertia) >= n_axes else 0,
        len(gene_ids),
    )
    results["gmm_n_axes"] = n_axes

    # ── Step 3: Fit GMM with BIC selection ────────────────────────────
    gmm, best_k, bic_scores = _fit_gmm_bic(X, min_k=min_k, max_k=max_k)
    labels = gmm.predict(X)
    probabilities = gmm.predict_proba(X)

    logger.info(
        "GMM model selection: best k=%d (BIC=%.1f) for %s",
        best_k, gmm.bic(X), sample_id,
    )

    results["gmm_best_k"] = best_k
    results["gmm_bic_scores"] = bic_scores
    results["gmm_labels"] = labels
    results["gmm_probabilities"] = probabilities

    # Cluster sizes
    for k in range(best_k):
        n_in_k = (labels == k).sum()
        logger.info("  cluster %d: %d genes (%.1f%%)", k, n_in_k, 100 * n_in_k / len(labels))

    # ── Step 4: Identify RP cluster ───────────────────────────────────
    rp_cluster = _identify_rp_cluster(gmm, labels, gene_ids, rp_gene_ids)
    results["gmm_rp_cluster"] = rp_cluster

    # Collect gene IDs in RP cluster
    cluster_gene_ids = {
        gid for gid, lbl in zip(gene_ids, labels) if lbl == rp_cluster
    }
    results["gmm_cluster_gene_ids"] = cluster_gene_ids

    n_cluster = len(cluster_gene_ids)
    n_rp_in_cluster = len(cluster_gene_ids & rp_gene_ids)
    logger.info(
        "RP cluster %d: %d total genes, %d are RPs, %d are non-RP co-clustered genes",
        rp_cluster, n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster,
    )

    if n_cluster < _MIN_CLUSTER_SIZE:
        logger.warning(
            "RP cluster is small (%d genes); expression scoring may be unreliable",
            n_cluster,
        )

    # ── Step 5: Compute RP-cluster RSCU ───────────────────────────────
    cluster_rscu = _compute_cluster_rscu(rscu_gene_df, cluster_gene_ids)
    results["gmm_cluster_rscu"] = cluster_rscu

    # ── Step 6: Quality check — compare RP-cluster RSCU to RP-only RSCU ─
    rp_cosine_sim = np.nan
    if rp_rscu_df is not None and not rp_rscu_df.empty:
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rp_rscu_df.columns and c in cluster_rscu.index]
        if rscu_cols:
            rp_only_rscu = rp_rscu_df[rscu_cols].mean()
            # cosine_dist returns distance; similarity = 1 - distance
            rp_cosine_sim = 1.0 - cosine_dist(
                cluster_rscu[rscu_cols].fillna(0).values,
                rp_only_rscu.fillna(0).values,
            )
            logger.info(
                "RP-cluster vs RP-only RSCU cosine similarity: %.4f "
                "(>0.95 = cluster closely matches RP signal; "
                "<0.85 = cluster has diverged, check results)",
                rp_cosine_sim,
            )
    results["gmm_rp_cosine_sim"] = rp_cosine_sim

    # ── Step 7: Save outputs ──────────────────────────────────────────

    # Gene-level cluster assignments
    cluster_df = pd.DataFrame({
        "gene": gene_ids,
        "gmm_cluster": labels,
        "gmm_rp_cluster_prob": probabilities[:, rp_cluster],
        "in_rp_cluster": [gid in cluster_gene_ids for gid in gene_ids],
        "is_ribosomal_protein": [gid in rp_gene_ids for gid in gene_ids],
    })
    cluster_path = gmm_dir / f"{sample_id}_gmm_clusters.tsv"
    cluster_df.to_csv(cluster_path, sep="\t", index=False)
    results["gmm_clusters_path"] = cluster_path

    # RP-cluster RSCU reference
    rscu_path = gmm_dir / f"{sample_id}_gmm_cluster_rscu.tsv"
    cluster_rscu.to_frame("RSCU").to_csv(rscu_path, sep="\t")
    results["gmm_cluster_rscu_path"] = rscu_path

    # RP-cluster gene IDs (one per line, for expression.py)
    ids_path = gmm_dir / f"{sample_id}_gmm_cluster_ids.txt"
    ids_path.write_text("\n".join(sorted(cluster_gene_ids)) + "\n")
    results["gmm_cluster_ids_path"] = ids_path

    # Summary stats
    summary = {
        "sample_id": sample_id,
        "n_genes": len(gene_ids),
        "n_coa_axes": n_axes,
        "best_k": best_k,
        "rp_cluster": rp_cluster,
        "rp_cluster_size": n_cluster,
        "rp_genes_in_cluster": n_rp_in_cluster,
        "non_rp_in_cluster": n_cluster - n_rp_in_cluster,
        "total_rp_genes": len(rp_gene_ids),
        "rp_cosine_similarity": round(rp_cosine_sim, 4) if not np.isnan(rp_cosine_sim) else None,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = gmm_dir / f"{sample_id}_gmm_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    results["gmm_summary_path"] = summary_path

    # COA coordinates with cluster assignments
    coa_with_clusters = coa_coords.copy()
    # Only merge for genes that were clustered (valid_mask filtered)
    cluster_merge = pd.DataFrame({"gene": gene_ids, "gmm_cluster": labels})
    coa_with_clusters = coa_with_clusters.merge(cluster_merge, on="gene", how="left")
    results["gmm_coa_coords"] = coa_with_clusters

    # ── Step 8: Diagnostic plots ──────────────────────────────────────
    try:
        inertia_pcts = (0.0, 0.0)
        if len(coa_inertia) >= 2:
            inertia_pcts = (
                float(coa_inertia["pct_inertia"].iloc[0]),
                float(coa_inertia["pct_inertia"].iloc[1]),
            )

        _plot_bic_selection(
            bic_scores, best_k,
            gmm_dir / f"{sample_id}_gmm_bic",
            sample_id,
        )
        results["gmm_bic_plot"] = gmm_dir / f"{sample_id}_gmm_bic.png"

        _plot_coa_clusters(
            coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True),
            labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_clusters",
            sample_id,
            inertia_pcts=inertia_pcts,
        )
        results["gmm_coa_plot"] = gmm_dir / f"{sample_id}_gmm_coa_clusters.png"

        _plot_cluster_separation(
            coa_coords, labels, probabilities, rp_cluster,
            gmm_dir / f"{sample_id}_gmm_cluster_separation",
            sample_id,
        )
        results["gmm_separation_plot"] = gmm_dir / f"{sample_id}_gmm_cluster_separation.png"

        # Cosine-transformed view: removes gene-length magnitude effects,
        # projects onto unit hypersphere so cluster boundaries reflect
        # codon preference direction rather than RSCU amplitude.
        coa_filtered = coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True)
        _plot_coa_clusters_cosine(
            coa_filtered, labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_cosine",
            sample_id,
            n_axes=n_axes,
        )
        results["gmm_coa_cosine_plot"] = gmm_dir / f"{sample_id}_gmm_coa_cosine.png"

        # Z-score-standardised view: per-axis zero-mean unit-variance,
        # enables visual comparison of cluster separation across genomes.
        _plot_coa_clusters_zscore(
            coa_filtered, labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_zscore",
            sample_id,
            n_axes=n_axes,
        )
        results["gmm_coa_zscore_plot"] = gmm_dir / f"{sample_id}_gmm_coa_zscore.png"

    except Exception as e:
        logger.warning("GMM diagnostic plot generation failed: %s", e)

    logger.info(
        "GMM clustering complete for %s: k=%d, RP cluster=%d (%d genes)",
        sample_id, best_k, rp_cluster, n_cluster,
    )

    return results
