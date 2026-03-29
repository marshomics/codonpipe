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

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

from Bio import SeqIO

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.rscu import compute_rscu_from_counts, count_codons
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES

# Minimum gene length (nt) to include in concatenated codon counts.
_MIN_GENE_LENGTH = 240

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
    ffn_path: Path,
    cluster_gene_ids: set[str],
    min_length: int = _MIN_GENE_LENGTH,
) -> pd.Series:
    """Compute RSCU by concatenated pooling of codon counts across cluster genes.

    Pools raw codon counts from all genes in the cluster, then computes RSCU
    from the pooled counts.  This is the Sharp & Li (1987) standard: longer
    genes contribute proportionally more codons, which is correct because
    their per-codon frequencies are more precisely estimated.

    Args:
        ffn_path: Path to the nucleotide CDS FASTA containing all genes.
        cluster_gene_ids: Set of gene IDs belonging to the target cluster.
        min_length: Minimum sequence length (nt) to include.

    Returns:
        Series indexed by RSCU column names (e.g. 'Phe-UUU') with pooled
        RSCU values.
    """
    from collections import Counter

    total_counts: Counter = Counter()
    n_included = 0

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        if rec.id not in cluster_gene_ids:
            continue
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        total_counts += count_codons(seq)
        n_included += 1

    if not total_counts:
        logger.warning(
            "No codons pooled from cluster (%d IDs requested, %d passed length filter)",
            len(cluster_gene_ids), n_included,
        )
        return pd.Series(dtype=float)

    rscu_dict = compute_rscu_from_counts(total_counts)
    logger.info(
        "Cluster RSCU computed by concatenated pooling (%d genes, %d total codons)",
        n_included, sum(total_counts.values()),
    )

    # Return as a Series with only the standard RSCU column names
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_dict]
    return pd.Series({c: rscu_dict[c] for c in rscu_cols})


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


def _plot_pairwise_coa_matrix(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    rp_cluster: int,
    rp_gene_ids: set[str],
    n_axes: int,
    output_path: Path,
    sample_id: str,
) -> None:
    """Pairwise scatter matrix of all retained COA axes.

    Each off-diagonal panel shows a 2-D projection coloured by GMM cluster.
    Diagonal panels show per-axis density by cluster.  This reveals cluster
    separations that are invisible in the default Axis 1 vs Axis 2 view.
    """
    plt.rcParams.update(STYLE_PARAMS)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n = len(axis_cols)
    if n < 2:
        return

    genes = coa_coords["gene"].values
    V = coa_coords[axis_cols].values
    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    if n == 2:
        axes = np.array(axes).reshape(n, n)

    rp_idx = np.array([i for i, g in enumerate(genes) if g in rp_gene_ids])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row == col:
                # Diagonal: per-cluster density
                for k in range(n_clusters):
                    mask = labels == k
                    ax.hist(
                        V[mask, row], bins=30, alpha=0.4,
                        color=cmap(k), density=True,
                        edgecolor="none",
                    )
                ax.set_ylabel("Density" if col == 0 else "")
            else:
                # Off-diagonal: scatter
                colors = [cmap(l) for l in labels]
                ax.scatter(
                    V[:, col], V[:, row],
                    c=colors, alpha=0.25, s=6, edgecolors="none", rasterized=True,
                )
                # Highlight RP cluster
                rp_mask = labels == rp_cluster
                ax.scatter(
                    V[rp_mask, col], V[rp_mask, row],
                    c=[cmap(rp_cluster)] * rp_mask.sum(),
                    alpha=0.6, s=10, edgecolors="none",
                )
                # Mark RPs
                if len(rp_idx) > 0:
                    ax.scatter(
                        V[rp_idx, col], V[rp_idx, row],
                        facecolors="none", edgecolors="black",
                        s=20, linewidths=0.5, zorder=5,
                    )

            if row == n - 1:
                ax.set_xlabel(axis_cols[col])
            else:
                ax.set_xticklabels([])
            if col == 0 and row != col:
                ax.set_ylabel(axis_cols[row])
            elif col != 0:
                ax.set_yticklabels([])

    fig.suptitle(
        f"{sample_id}: pairwise COA axes (k={n_clusters})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_silhouette(
    labels: np.ndarray,
    rp_cluster: int,
    sample_silhouettes: np.ndarray,
    mean_score: float,
    output_path: Path,
    sample_id: str,
) -> None:
    """Per-cluster silhouette coefficient plot.

    Each cluster is shown as a horizontal bar of sorted per-gene silhouette
    values, coloured by cluster.  The RP cluster is annotated.
    """
    plt.rcParams.update(STYLE_PARAMS)
    n_clusters = int(labels.max()) + 1

    fig, ax = plt.subplots(figsize=(7, max(4, 0.5 * n_clusters)))
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))

    y_lower = 0
    for k in range(n_clusters):
        k_silhouettes = sample_silhouettes[labels == k]
        k_silhouettes.sort()
        n_k = len(k_silhouettes)
        y_upper = y_lower + n_k

        color = cmap(k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, k_silhouettes,
            facecolor=color, edgecolor=color, alpha=0.7,
        )

        # Cluster label
        label = f"C{k}"
        if k == rp_cluster:
            label += " (RP)"
        ax.text(-0.05, y_lower + 0.5 * n_k, label, fontsize=7, va="center", ha="right")
        y_lower = y_upper + 2  # gap between clusters

    ax.axvline(mean_score, color="#d7191c", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Genes (sorted within cluster)")
    ax.set_title(
        f"{sample_id}: silhouette analysis (mean={mean_score:.3f}, k={n_clusters})"
    )
    ax.set_yticks([])

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_coa_3d(
    coa_coords: pd.DataFrame,
    labels: np.ndarray,
    rp_cluster: int,
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
    inertia_pcts: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """3-D scatter of COA axes 1, 2, 3 coloured by GMM cluster.

    Provides depth context that the 2-D Axis 1 vs 2 plot misses, which is
    relevant when axis 3 carries meaningful compositional signal (strand
    bias, amino acid content).
    """
    plt.rcParams.update(STYLE_PARAMS)
    axis_cols = ["Axis1", "Axis2", "Axis3"]
    if not all(c in coa_coords.columns for c in axis_cols):
        return

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    genes = coa_coords["gene"].values
    V = coa_coords[axis_cols].values

    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))
    colors = [cmap(l) for l in labels]

    ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=colors, alpha=0.25, s=8, depthshade=True)

    # RP cluster overlay
    rp_mask = labels == rp_cluster
    ax.scatter(
        V[rp_mask, 0], V[rp_mask, 1], V[rp_mask, 2],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.6, s=14, depthshade=True,
        label=f"Cluster {rp_cluster} (RP)",
    )

    # Mark RP genes
    rp_idx = [i for i, g in enumerate(genes) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            V[rp_idx, 0], V[rp_idx, 1], V[rp_idx, 2],
            facecolors="none", edgecolors="black", s=30, linewidths=0.6,
            label=f"RPs (n={len(rp_idx)})", depthshade=False,
        )

    pct1, pct2, pct3 = inertia_pcts
    ax.set_xlabel(f"Axis 1 ({pct1:.1f}%)", labelpad=8)
    ax.set_ylabel(f"Axis 2 ({pct2:.1f}%)", labelpad=8)
    ax.set_zlabel(f"Axis 3 ({pct3:.1f}%)", labelpad=8)
    ax.set_title(f"{sample_id}: 3-D COA space (k={n_clusters})")
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=25, azim=135)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _compute_umap_embedding(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
) -> np.ndarray | None:
    """Compute UMAP embedding on the COA coordinate matrix.

    Returns None if umap-learn is not installed.
    """
    if not _HAS_UMAP:
        logger.warning("umap-learn not installed; skipping UMAP plots")
        return None

    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=min(n_neighbors, X.shape[0] - 1),
        min_dist=min_dist,
        metric="euclidean",
    )
    return reducer.fit_transform(X)


def _plot_umap_2d(
    embedding: np.ndarray,
    labels: np.ndarray,
    rp_cluster: int,
    gene_ids: list[str],
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
) -> None:
    """2-D UMAP embedding coloured by GMM cluster.

    UMAP preserves local neighbourhood structure, making cluster boundaries
    visible even when linear projections (COA axes) show heavy overlap.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))
    colors = [cmap(l) for l in labels]

    ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=colors, alpha=0.35, s=12, edgecolors="none", rasterized=True,
    )

    # RP cluster overlay
    rp_mask = labels == rp_cluster
    ax.scatter(
        embedding[rp_mask, 0], embedding[rp_mask, 1],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.7, s=18, edgecolors="none",
        label=f"Cluster {rp_cluster} (RP cluster)",
    )

    # Mark RP genes
    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            embedding[rp_idx, 0], embedding[rp_idx, 1],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"{sample_id}: UMAP of COA space (k={n_clusters})")
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_umap_3d(
    embedding_3d: np.ndarray,
    labels: np.ndarray,
    rp_cluster: int,
    gene_ids: list[str],
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
) -> None:
    """3-D UMAP embedding coloured by GMM cluster.

    Adds depth to the UMAP visualisation for clusters that sit behind each
    other in the 2-D projection.
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    n_clusters = int(labels.max()) + 1
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))
    colors = [cmap(l) for l in labels]

    ax.scatter(
        embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
        c=colors, alpha=0.25, s=8, depthshade=True,
    )

    rp_mask = labels == rp_cluster
    ax.scatter(
        embedding_3d[rp_mask, 0], embedding_3d[rp_mask, 1], embedding_3d[rp_mask, 2],
        c=[cmap(rp_cluster)] * rp_mask.sum(),
        alpha=0.6, s=14, depthshade=True,
        label=f"Cluster {rp_cluster} (RP)",
    )

    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            embedding_3d[rp_idx, 0], embedding_3d[rp_idx, 1], embedding_3d[rp_idx, 2],
            facecolors="none", edgecolors="black", s=30, linewidths=0.6,
            label=f"RPs (n={len(rp_idx)})", depthshade=False,
        )

    ax.set_xlabel("UMAP 1", labelpad=8)
    ax.set_ylabel("UMAP 2", labelpad=8)
    ax.set_zlabel("UMAP 3", labelpad=8)
    ax.set_title(f"{sample_id}: 3-D UMAP of COA space (k={n_clusters})")
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=25, azim=135)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Amino-acid property grouping (shared with plots.py, duplicated here to keep
# the module self-contained and avoid circular imports)
# ---------------------------------------------------------------------------

_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
for _base, _letter in [("Ser", "S"), ("Leu", "L"), ("Arg", "R")]:
    for _sfx in ("2", "4"):
        _THREE_TO_ONE[f"{_base}{_sfx}"] = _letter

_AA_GROUPS = {
    "Nonpolar": ["G", "A", "V", "L", "I", "P", "F", "M", "W"],
    "Polar":    ["S", "T", "C", "Y", "N", "Q"],
    "Positive": ["K", "R", "H"],
    "Negative": ["D", "E"],
}
_GROUP_COLORS = {
    "Nonpolar": "#5B8DBE",
    "Polar":    "#6AB187",
    "Positive": "#D4726A",
    "Negative": "#9B7FBF",
}
_AA_TO_GROUP = {}
for _grp, _aas in _AA_GROUPS.items():
    for _a in _aas:
        _AA_TO_GROUP[_a] = _grp
_AA_ORDER = list("ARNDC QEGHILKMFPSTWYV".replace(" ", ""))
_AA_NAMES = {v: k for k, v in _THREE_TO_ONE.items() if len(k) == 3}


def _plot_gmm_cluster_rscu_heatmap(
    cluster_rscu: pd.Series,
    output_path: Path,
    sample_id: str,
    cluster_label: int,
    n_genes: int,
) -> None:
    """Rounded-cell RSCU heatmap of the GMM cluster's pooled codon usage.

    Mirrors the visual style of ``plot_rscu_heatmap_rounded`` in plots.py
    but shows only the GMM RP-cluster reference RSCU (concatenated pooling).
    Amino acids are grouped by biochemical property; each cell shows the
    codon triplet and its RSCU value.
    """
    plt.rcParams.update(STYLE_PARAMS)

    if cluster_rscu is None or cluster_rscu.empty:
        return

    # Build a dataframe from the RSCU series (index = "AminoAcid-Codon")
    rows = []
    for label, rscu in cluster_rscu.items():
        if "-" not in label:
            continue
        aa_name, codon = label.rsplit("-", 1)
        one_letter = _THREE_TO_ONE.get(aa_name)
        if one_letter is None or one_letter in ("M", "W"):
            continue
        rows.append({"amino_acid": aa_name, "AA": one_letter, "codon": codon, "rscu": rscu})

    if not rows:
        return
    df = pd.DataFrame(rows)

    # Order amino acids by biochemical group
    group_order = ["Nonpolar", "Polar", "Positive", "Negative"]
    present = set(df["AA"].unique())
    aa_ordered = [aa for grp in group_order for aa in _AA_ORDER
                  if _AA_TO_GROUP.get(aa) == grp and aa in present]
    if not aa_ordered:
        return

    # Diverging colormap: blue (low) → white (1.0) → red (high)
    max_rscu = df["rscu"].max()
    vmax = min(max(max_rscu * 1.05, 3.0), 5.0)
    colors_below = plt.cm.Blues_r(np.linspace(0.0, 0.7, 128))
    colors_above = plt.cm.Reds(np.linspace(0.0, 0.75, 128))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rscu_diverging", np.vstack([colors_below, colors_above]),
    )
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1], wspace=0.03)
    ax = fig.add_subplot(gs[0])
    cax = fig.add_subplot(gs[1])

    cell_h, cell_w = 0.85, 0.85
    y_pos = 0
    y_positions = {}
    group_boundaries = []
    current_group = None

    for aa in aa_ordered:
        grp = _AA_TO_GROUP.get(aa, "")
        if grp != current_group:
            if current_group is not None:
                group_boundaries.append(y_pos - 0.15)
                y_pos += 0.3
            current_group = grp
        y_positions[aa] = y_pos
        y_pos += 1.0

    for aa in aa_ordered:
        sub = df[df["AA"] == aa].sort_values("codon")
        y = y_positions[aa]
        for j, (_, row) in enumerate(sub.iterrows()):
            rscu = row["rscu"]
            color = cmap(norm(rscu))
            rect = FancyBboxPatch(
                (j * 1.0 + 0.075, y + 0.075), cell_w, cell_h,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=1.2,
            )
            ax.add_patch(rect)

            text_color = "white" if (rscu > 2.5 or rscu < 0.2) else "#333333"
            pfx = [pe.withStroke(linewidth=0.3, foreground="white")] if rscu > 2.5 else []

            ax.text(
                j * 1.0 + 0.075 + cell_w / 2, y + 0.075 + cell_h * 0.62,
                row["codon"], ha="center", va="center",
                fontsize=8, fontweight="bold", fontfamily="monospace",
                color=text_color, path_effects=pfx,
            )
            ax.text(
                j * 1.0 + 0.075 + cell_w / 2, y + 0.075 + cell_h * 0.28,
                f"{rscu:.2f}", ha="center", va="center",
                fontsize=6.5, color=text_color, alpha=0.85,
            )

    for aa in aa_ordered:
        y = y_positions[aa]
        grp = _AA_TO_GROUP.get(aa, "")
        name = _AA_NAMES.get(aa, aa)
        ax.text(
            -0.3, y + 0.075 + cell_h / 2, f"{name} ({aa})",
            ha="right", va="center", fontsize=8.5, fontweight="bold",
            color=_GROUP_COLORS.get(grp, "#333333"),
        )

    for grp in group_order:
        aas_in_grp = [aa for aa in aa_ordered if _AA_TO_GROUP.get(aa) == grp]
        if aas_in_grp:
            y_start = y_positions[aas_in_grp[0]]
            y_end = y_positions[aas_in_grp[-1]] + 1.0
            y_mid = (y_start + y_end) / 2
            ax.text(
                -2.8, y_mid, grp, ha="center", va="center", fontsize=9,
                fontweight="bold", color=_GROUP_COLORS[grp], rotation=90,
            )
            ax.plot(
                [-2.2, -2.2], [y_start + 0.1, y_end - 0.1],
                color=_GROUP_COLORS[grp], linewidth=1.5, alpha=0.6, clip_on=False,
            )

    for yb in group_boundaries:
        ax.axhline(y=yb, color="#cccccc", linewidth=0.5, linestyle="-", alpha=0.5)

    ax.set_xlim(-3.2, 6 * 1.0 + 0.2)
    ax.set_ylim(-0.3, y_pos + 0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    title = f"GMM Cluster {cluster_label} RSCU ({n_genes} genes, concatenated pooling)"
    if sample_id:
        title += f"\n{sample_id}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("RSCU", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="-")
    cb.ax.text(
        1.6, 1.0, "= 1.0\n(no bias)", transform=cb.ax.get_yaxis_transform(),
        fontsize=7, va="center", ha="left", color="#555555",
    )

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_gmm_cluster_pergene_heatmap(
    rscu_gene_df: pd.DataFrame,
    cluster_gene_ids: set[str],
    labels: np.ndarray,
    gene_ids: list[str],
    rp_gene_ids: set[str],
    rp_cluster: int,
    output_path: Path,
    sample_id: str,
) -> None:
    """Clustered heatmap of per-gene RSCU for genes in the GMM RP cluster.

    Rows = genes in the RP cluster (hierarchically clustered), columns =
    codons (hierarchically clustered).  A colour sidebar marks ribosomal
    proteins vs non-RP co-clustered genes.
    """
    plt.rcParams.update(STYLE_PARAMS)
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        return

    # Filter to GMM cluster members only
    cluster_df = rscu_gene_df[rscu_gene_df["gene"].isin(cluster_gene_ids)].copy()
    if len(cluster_df) < 3:
        logger.info("Too few genes in GMM cluster for per-gene heatmap (%d)", len(cluster_df))
        return

    gene_col = cluster_df["gene"].values
    data = cluster_df[rscu_cols].values.copy()
    data = np.nan_to_num(data, nan=0.0)

    # Cap at 500 genes for readability
    if len(data) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(data), 500, replace=False)
        data = data[idx]
        gene_col = gene_col[idx]

    # Row colour sidebar: RP genes vs non-RP
    row_colors = pd.Series(
        ["#D4726A" if g in rp_gene_ids else "#5B8DBE" for g in gene_col],
        index=range(len(gene_col)),
        name="Gene type",
    )

    codon_labels = [c.split("-")[-1] for c in rscu_cols]
    data_df = pd.DataFrame(data, columns=codon_labels)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Clustering large matrix")
        g = sns.clustermap(
            data_df,
            cmap="RdYlBu_r",
            center=1.0,
            row_cluster=True,
            col_cluster=True,
            method="average",
            metric="euclidean",
            linewidths=0,
            xticklabels=True,
            yticklabels=False,
            row_colors=row_colors,
            figsize=(14, min(10, max(5, len(data) // 30))),
            cbar_kws={"label": "RSCU"},
        )

    # Legend for sidebar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D4726A", label="Ribosomal protein"),
        Patch(facecolor="#5B8DBE", label="Non-RP co-clustered"),
    ]
    g.ax_heatmap.legend(
        handles=legend_elements, loc="upper left",
        bbox_to_anchor=(1.06, 1.0), fontsize=8, framealpha=0.7,
    )

    title = f"Per-Gene RSCU — GMM Cluster {rp_cluster} ({len(data)} genes)"
    if sample_id:
        title += f" — {sample_id}"
    g.fig.suptitle(title, y=1.02, fontsize=12, fontweight="bold")

    for fmt in FORMATS:
        g.fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(g.fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gmm_clustering(
    rscu_gene_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    ffn_path: Path | None = None,
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
        ffn_path: Path to nucleotide CDS FASTA (for concatenated RSCU pooling).
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

    # ── Step 5: Compute RP-cluster RSCU (concatenated pooling) ──────
    if ffn_path and ffn_path.exists():
        cluster_rscu = _compute_cluster_rscu(ffn_path, cluster_gene_ids)
    else:
        # Fallback: per-gene mean if FFN not available (less accurate)
        logger.warning("FFN path not available; falling back to per-gene mean RSCU for cluster reference")
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
        cluster_df = rscu_gene_df[rscu_gene_df["gene"].isin(cluster_gene_ids)]
        cluster_rscu = cluster_df[rscu_cols].mean() if not cluster_df.empty else pd.Series(dtype=float)
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

    # Per-gene silhouette values (computed here so they go into the TSV)
    per_gene_silhouette = np.full(len(labels), np.nan)
    if best_k >= 2:
        try:
            per_gene_silhouette = silhouette_samples(X, labels)
        except Exception:
            pass

    # Gene-level cluster assignments
    cluster_df = pd.DataFrame({
        "gene": gene_ids,
        "gmm_cluster": labels,
        "gmm_rp_cluster_prob": probabilities[:, rp_cluster],
        "silhouette_score": per_gene_silhouette,
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

    # ── Step 7b: Compute silhouette score (before plots, for summary) ──
    mean_silhouette_score = np.nan
    if best_k >= 2:
        try:
            mean_silhouette_score = float(silhouette_score(X, labels))
        except Exception:
            pass

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
        "mean_silhouette_score": round(mean_silhouette_score, 4) if not np.isnan(mean_silhouette_score) else None,
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
        inertia_pcts_2 = (0.0, 0.0)
        inertia_pcts_3 = (0.0, 0.0, 0.0)
        if len(coa_inertia) >= 2:
            inertia_pcts_2 = (
                float(coa_inertia["pct_inertia"].iloc[0]),
                float(coa_inertia["pct_inertia"].iloc[1]),
            )
        if len(coa_inertia) >= 3:
            inertia_pcts_3 = (
                float(coa_inertia["pct_inertia"].iloc[0]),
                float(coa_inertia["pct_inertia"].iloc[1]),
                float(coa_inertia["pct_inertia"].iloc[2]),
            )

        coa_filtered = coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True)

        # BIC model selection curve
        _plot_bic_selection(
            bic_scores, best_k,
            gmm_dir / f"{sample_id}_gmm_bic",
            sample_id,
        )
        results["gmm_bic_plot"] = gmm_dir / f"{sample_id}_gmm_bic.png"

        # COA Axis 1 vs 2 scatter
        _plot_coa_clusters(
            coa_filtered,
            labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_clusters",
            sample_id,
            inertia_pcts=inertia_pcts_2,
        )
        results["gmm_coa_plot"] = gmm_dir / f"{sample_id}_gmm_coa_clusters.png"

        # RP cluster posterior probability histogram
        _plot_cluster_separation(
            coa_coords, labels, probabilities, rp_cluster,
            gmm_dir / f"{sample_id}_gmm_cluster_separation",
            sample_id,
        )
        results["gmm_separation_plot"] = gmm_dir / f"{sample_id}_gmm_cluster_separation.png"

        # Cosine-transformed COA scatter
        _plot_coa_clusters_cosine(
            coa_filtered, labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_cosine",
            sample_id,
            n_axes=n_axes,
        )
        results["gmm_coa_cosine_plot"] = gmm_dir / f"{sample_id}_gmm_coa_cosine.png"

        # Z-score-standardised COA scatter
        _plot_coa_clusters_zscore(
            coa_filtered, labels, rp_cluster, rp_gene_ids,
            gmm_dir / f"{sample_id}_gmm_coa_zscore",
            sample_id,
            n_axes=n_axes,
        )
        results["gmm_coa_zscore_plot"] = gmm_dir / f"{sample_id}_gmm_coa_zscore.png"

        # Pairwise scatter matrix across all retained COA axes
        _plot_pairwise_coa_matrix(
            coa_filtered, labels, rp_cluster, rp_gene_ids, n_axes,
            gmm_dir / f"{sample_id}_gmm_coa_pairwise",
            sample_id,
        )
        results["gmm_pairwise_plot"] = gmm_dir / f"{sample_id}_gmm_coa_pairwise.png"

        # Silhouette analysis (quantitative cluster separation)
        if best_k >= 2 and not np.all(np.isnan(per_gene_silhouette)):
            _plot_silhouette(
                labels, rp_cluster,
                per_gene_silhouette, mean_silhouette_score,
                gmm_dir / f"{sample_id}_gmm_silhouette",
                sample_id,
            )
            results["gmm_silhouette_plot"] = gmm_dir / f"{sample_id}_gmm_silhouette.png"
            results["gmm_mean_silhouette"] = mean_silhouette_score
            logger.info("Mean silhouette score: %.3f", mean_silhouette_score)

        # 3-D COA scatter (axes 1, 2, 3)
        if n_axes >= 3:
            _plot_coa_3d(
                coa_filtered, labels, rp_cluster, rp_gene_ids,
                gmm_dir / f"{sample_id}_gmm_coa_3d",
                sample_id,
                inertia_pcts=inertia_pcts_3,
            )
            results["gmm_coa_3d_plot"] = gmm_dir / f"{sample_id}_gmm_coa_3d.png"

        # UMAP embeddings (2-D and 3-D)
        if _HAS_UMAP and len(gene_ids) > 30:
            logger.info("Computing UMAP embeddings for %s", sample_id)

            emb_2d = _compute_umap_embedding(X, n_components=2)
            if emb_2d is not None:
                _plot_umap_2d(
                    emb_2d, labels, rp_cluster, gene_ids, rp_gene_ids,
                    gmm_dir / f"{sample_id}_gmm_umap_2d",
                    sample_id,
                )
                results["gmm_umap_2d_plot"] = gmm_dir / f"{sample_id}_gmm_umap_2d.png"

            emb_3d = _compute_umap_embedding(X, n_components=3)
            if emb_3d is not None:
                _plot_umap_3d(
                    emb_3d, labels, rp_cluster, gene_ids, rp_gene_ids,
                    gmm_dir / f"{sample_id}_gmm_umap_3d",
                    sample_id,
                )
                results["gmm_umap_3d_plot"] = gmm_dir / f"{sample_id}_gmm_umap_3d.png"
        elif not _HAS_UMAP:
            logger.info("umap-learn not installed; UMAP plots skipped")

        # GMM cluster RSCU rounded heatmap (amino acids × codons)
        _plot_gmm_cluster_rscu_heatmap(
            cluster_rscu, gmm_dir / f"{sample_id}_gmm_cluster_rscu_heatmap",
            sample_id, rp_cluster, n_cluster,
        )
        results["gmm_cluster_rscu_heatmap"] = gmm_dir / f"{sample_id}_gmm_cluster_rscu_heatmap.png"

        # GMM cluster per-gene RSCU heatmap (genes × codons, cluster members only)
        _plot_gmm_cluster_pergene_heatmap(
            rscu_gene_df, cluster_gene_ids, labels, gene_ids, rp_gene_ids,
            rp_cluster, gmm_dir / f"{sample_id}_gmm_cluster_pergene_heatmap",
            sample_id,
        )
        results["gmm_cluster_pergene_heatmap"] = gmm_dir / f"{sample_id}_gmm_cluster_pergene_heatmap.png"

    except Exception as e:
        logger.warning("GMM diagnostic plot generation failed: %s", e, exc_info=True)

    logger.info(
        "GMM clustering complete for %s: k=%d, RP cluster=%d (%d genes)",
        sample_id, best_k, rp_cluster, n_cluster,
    )

    return results
