"""Two-cluster Mahalanobis module: RP-anchor and density-core.

Identifies two clusters in COA (Correspondence Analysis) space:

1. **RP cluster** — anchored to the centroid of ribosomal protein genes,
   stabilised by bootstrap resampling.  Uses MinCovDet for robust
   covariance and a chi-squared boundary to define a tight ellipsoidal
   cluster.  Non-RP genes inside the boundary are included.

2. **Density-core cluster** — anchored to the genome-wide density peak
   (mode of a 2-D KDE on the first two COA axes), with the same
   chi-squared boundary methodology.

Both clusters use the **same** machinery:
  - MinCovDet covariance estimation (two-pass outlier removal)
  - Chi-squared threshold: sqrt(chi2.ppf(p, df=n_axes)), giving a
    principled, covariance-independent boundary
  - Distance-weighted RSCU pooling for the cluster reference
  - Soft membership via sigmoid on Mahalanobis distance

When RP genes split into distinct sub-populations (e.g. HGT, ribosome
specialisation), the module detects multiple sub-clusters via GMM+BIC and
runs independent fits for each.  The representative RP cluster is the one
containing the most RP genes.

Likewise, when the genome has multiple density peaks, each peak is fitted
independently and the representative density-core cluster is the one
containing the most genes.

The four-way gene classification (both / rp_only / dens_only / neither)
and translational selection strength assessment are preserved.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from Bio import SeqIO

from codonpipe.modules.advanced_analyses import compute_coa_on_rscu
from codonpipe.modules.rscu import compute_rscu_from_counts, count_codons
from codonpipe.plotting.utils import DPI, FORMATS, STYLE_PARAMS, apply_style, save_fig
from codonpipe.utils.codon_tables import (
    MIN_GENE_LENGTH, RSCU_COLUMN_NAMES, RSCU_COL_TO_CODON, AA_CODON_GROUPS_RSCU,
)

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_GENES_FOR_CLUSTERING = 50
_MIN_COA_AXES = 2
_MAX_COA_AXES = 8
_CUMULATIVE_INERTIA_TARGET = 0.80
_MIN_CLUSTER_SIZE = 10

# Robust covariance fitting
_RP_OUTLIER_ALPHA = 0.025          # Chi-squared alpha for RP outlier detection
_MIN_RP_FOR_ROBUST = 10            # Min genes for MinCovDet; else empirical

# Cluster boundaries
_CLUSTER_CHI2_P = 0.95             # Density cluster: chi-squared captures ~95%
_RP_EMPIRICAL_PCTL = 90            # RP cluster: 90th percentile of cleaned RP distances

# RP sub-cluster detection
_RP_SUBCLUSTER_MIN = 15
_RP_SUBCLUSTER_MAX_K = 4
_RP_SUBCLUSTER_MIN_FRAC = 0.12

# Density-peak detection
_DENSITY_KDE_BANDWIDTH = "scott"
_DENSITY_SEED_RADIUS_PCTL = 30
_DENSITY_MULTI_PEAK_MIN_FRAC = 0.10  # Min fraction of genes for a peak to qualify

# Bootstrap
_DEFAULT_N_BOOTSTRAPS = 200

# Translational selection strength
_TSS_COSINE_STRONG = 0.90
_TSS_COSINE_WEAK = 0.97
_TSS_RPONLY_FRAC_STRONG = 0.05
_TSS_RPONLY_FRAC_WEAK = 0.01

# Backward-compatible aliases (kept for cluster_stability imports)
_DISTANCE_MULTIPLIER = 2.0


# ---------------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------------

def _safe_inv(mat: np.ndarray) -> np.ndarray:
    """Invert a matrix, falling back to pseudo-inverse if singular."""
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        logger.warning("Singular covariance; using pseudo-inverse")
        return np.linalg.pinv(mat)


def _empirical_cov(X: np.ndarray) -> np.ndarray:
    """Compute empirical covariance, safe for very small n."""
    c = np.cov(X.T) if X.shape[0] > 1 else np.eye(X.shape[1])
    if c.ndim == 0:
        c = np.array([[float(c)]])
    elif c.ndim == 1:
        c = c.reshape(1, 1)
    return c


# ---------------------------------------------------------------------------
# COA axis selection
# ---------------------------------------------------------------------------

def _select_n_axes(coa_inertia: pd.DataFrame, max_axes: int = _MAX_COA_AXES) -> int:
    """Select number of COA axes to retain based on cumulative inertia."""
    if coa_inertia.empty:
        return _MIN_COA_AXES
    cum_pct = coa_inertia["cum_pct"].values
    target = _CUMULATIVE_INERTIA_TARGET * 100
    n = _MIN_COA_AXES
    for i, cp in enumerate(cum_pct):
        n = i + 1
        if cp >= target:
            break
    return max(_MIN_COA_AXES, min(n, max_axes, len(cum_pct)))


# ---------------------------------------------------------------------------
# Robust covariance fitting (shared by RP and density anchors)
# ---------------------------------------------------------------------------

def _fit_robust_rp_reference(
    X_rp: np.ndarray,
    n_axes: int,
    alpha: float = _RP_OUTLIER_ALPHA,
    min_rp: int = _MIN_RP_FOR_ROBUST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit robust covariance with two-pass outlier removal.

    Pass 1: MinCovDet → identify chi-squared outliers.
    Pass 2: Refit on inliers.

    Falls back to empirical covariance when the sample size is too small
    for MinCovDet to be numerically stable (< 2·n_axes + 3).

    Returns (centroid, cov, cov_inv, outlier_mask).
    """
    import warnings as _warnings

    n_rp = len(X_rp)

    # MinCovDet needs ~2×p samples for a p-dimensional problem to be
    # numerically stable.  Below that threshold, empirical covariance
    # is more reliable and avoids sklearn determinant-increase warnings.
    min_for_mcd = max(min_rp, 2 * n_axes + 3)

    if n_rp < max(min_rp, n_axes + 2):
        logger.info(
            "Only %d reference genes (< %d); using empirical covariance",
            n_rp, max(min_rp, n_axes + 2),
        )
        centroid = X_rp.mean(axis=0)
        cov = _empirical_cov(X_rp)
        return centroid, cov, _safe_inv(cov), np.zeros(n_rp, dtype=bool)

    def _try_mcd(X_data):
        """Attempt MinCovDet; return (location, covariance) or None."""
        n = len(X_data)
        if n < min_for_mcd:
            return None
        try:
            sf = max(0.5, min(0.9, (n - 2) / n))
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message="Determinant has increased")
                mcd = MinCovDet(random_state=42, support_fraction=sf).fit(X_data)
            return mcd.location_, mcd.covariance_
        except Exception as e:
            logger.debug("MinCovDet failed (%s); falling back", e)
            return None

    # Pass 1
    result_1 = _try_mcd(X_rp)
    if result_1 is not None:
        centroid_1, cov_1 = result_1
    else:
        centroid_1 = X_rp.mean(axis=0)
        cov_1 = _empirical_cov(X_rp)

    cov_1_inv = _safe_inv(cov_1)
    rp_dists = np.array([
        np.sqrt(max(0.0, (x - centroid_1) @ cov_1_inv @ (x - centroid_1)))
        for x in X_rp
    ])

    chi2_crit = np.sqrt(chi2.ppf(1 - alpha, df=n_axes))
    outlier_mask = rp_dists > chi2_crit
    n_outliers = int(outlier_mask.sum())
    logger.info(
        "Pass 1: %d/%d outliers (Mahalanobis > %.2f, alpha=%.3f)",
        n_outliers, n_rp, chi2_crit, alpha,
    )

    # Pass 2
    X_clean = X_rp[~outlier_mask]
    if len(X_clean) < max(2, n_axes + 1):
        logger.warning("Too few inliers (%d); using all genes", len(X_clean))
        X_clean = X_rp
        outlier_mask = np.zeros(n_rp, dtype=bool)

    result_2 = _try_mcd(X_clean)
    if result_2 is not None:
        centroid, cov = result_2
    else:
        centroid = X_clean.mean(axis=0)
        cov = _empirical_cov(X_clean)

    return centroid, cov, _safe_inv(cov), outlier_mask


# ---------------------------------------------------------------------------
# Mahalanobis distances
# ---------------------------------------------------------------------------

def _compute_mahalanobis_distances(
    X: np.ndarray,
    centroid: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance from each row of X to centroid."""
    diff = X - centroid
    left = diff @ cov_inv
    d_sq = np.maximum(np.sum(left * diff, axis=1), 0.0)
    return np.sqrt(d_sq)


def _chi2_threshold(n_axes: int, p: float = _CLUSTER_CHI2_P) -> float:
    """Chi-squared Mahalanobis distance threshold."""
    return float(np.sqrt(chi2.ppf(p, df=n_axes)))


def _distance_to_membership(
    distances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Sigmoid soft membership: (n_genes, 2) with col1 = P(in cluster)."""
    scale = max(threshold / 5.0, 1e-6)
    z = np.clip((distances - threshold) / scale, -500, 500)
    score = 1.0 / (1.0 + np.exp(z))
    return np.column_stack([1.0 - score, score])


# ---------------------------------------------------------------------------
# Cluster RSCU computation
# ---------------------------------------------------------------------------

def _compute_cluster_rscu(
    ffn_path: Path,
    cluster_gene_ids: set[str],
    min_length: int = MIN_GENE_LENGTH,
    gene_weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute RSCU by distance-weighted pooling of codon counts."""
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
            "No codons pooled from cluster (%d IDs requested, %d passed filter)",
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


def _compute_genome_wide_rscu(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> pd.Series:
    """Pool codon counts across all genes and compute genome-wide RSCU."""
    total_counts: Counter = Counter()
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        total_counts += count_codons(seq)
    if not total_counts:
        return pd.Series(dtype=float)
    rscu_dict = compute_rscu_from_counts(total_counts)
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_dict]
    return pd.Series({c: rscu_dict[c] for c in rscu_cols})


# ---------------------------------------------------------------------------
# Core-CAI and rare codons
# ---------------------------------------------------------------------------

def _rscu_to_adaptation_weights(cluster_rscu: pd.Series) -> dict[str, float]:
    """Convert core-cluster RSCU to per-codon adaptation weights (Sharp & Li 1987)."""
    weights: dict[str, float] = {}
    for family_label, codons in AA_CODON_GROUPS_RSCU.items():
        family_cols = []
        for col in RSCU_COLUMN_NAMES:
            codon = RSCU_COL_TO_CODON[col]
            if codon in codons:
                family_cols.append((col, codon))
        vals = {}
        for col, codon in family_cols:
            if col in cluster_rscu.index:
                vals[codon] = cluster_rscu[col]
        if not vals:
            continue
        max_rscu = max(vals.values())
        if max_rscu <= 0:
            continue
        for codon, rscu_val in vals.items():
            weights[codon] = max(rscu_val / max_rscu, 1e-6)
    return weights


def _compute_core_cai(
    ffn_path: Path,
    adaptation_weights: dict[str, float],
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, float]:
    """CAI of each gene relative to core-cluster codon preferences."""
    from codonpipe.modules.rscu import dna_to_rna
    gene_cai: dict[str, float] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        rna = dna_to_rna(seq)
        log_sum, n = 0.0, 0
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i + 3]
            if codon in adaptation_weights:
                log_sum += np.log(adaptation_weights[codon])
                n += 1
        if n > 0:
            gene_cai[rec.id] = float(np.exp(log_sum / n))
    return gene_cai


def _compute_gene_lengths_codons(
    ffn_path: Path, min_length: int = MIN_GENE_LENGTH,
) -> dict[str, int]:
    """Count coding codons per gene (excluding stop)."""
    gene_lengths: dict[str, int] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        gene_lengths[rec.id] = len(seq) // 3
    return gene_lengths


def _compute_rare_codon_burden(
    ffn_path: Path,
    adaptation_weights: dict[str, float],
    rarity_ceiling: float = 0.3,
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, float]:
    """Rare codon burden: length-normalized, rarity-weighted score.

    For each codon in a gene, if its adaptation weight w_i is below
    *rarity_ceiling*, the codon contributes (1 - w_i) to the burden
    sum.  The total is divided by the number of scored codons, giving
    a per-codon average.  Higher values mean more (and more severe)
    rare codons.

    This metric complements core_CAI by explicitly isolating the impact
    of disfavoured codons while ignoring well-adapted ones.  CAI weights
    all codons equally; burden focuses exclusively on the rare tail.
    """
    from codonpipe.modules.rscu import dna_to_rna
    gene_burden: dict[str, float] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        rna = dna_to_rna(seq)
        burden_sum, n = 0.0, 0
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i + 3]
            if codon in adaptation_weights:
                w = adaptation_weights[codon]
                n += 1
                if w < rarity_ceiling:
                    burden_sum += (1.0 - w)
        if n > 0:
            gene_burden[rec.id] = burden_sum / n
    return gene_burden


def _identify_rare_codons(rscu_series: pd.Series, threshold: float = 0.1) -> set[str]:
    """Codons with RSCU below threshold."""
    rare: set[str] = set()
    for col, val in rscu_series.items():
        if pd.isna(val) or val >= threshold:
            continue
        codon = RSCU_COL_TO_CODON.get(col)
        if codon:
            rare.add(codon)
    return rare


def _count_rare_codons_per_gene(
    ffn_path: Path, rare_codons: set[str], min_length: int = MIN_GENE_LENGTH,
) -> dict[str, int]:
    """Count rare-codon positions per gene."""
    from codonpipe.modules.rscu import dna_to_rna
    gene_counts: dict[str, int] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        rna = dna_to_rna(seq)
        n_rare = sum(1 for i in range(0, len(rna) - 2, 3) if rna[i:i + 3] in rare_codons)
        gene_counts[rec.id] = n_rare
    return gene_counts


# ---------------------------------------------------------------------------
# RP dense-core selection
# ---------------------------------------------------------------------------

def _select_rp_dense_core(
    X_rp: np.ndarray,
    rp_gene_ids_list: list[str],
    density_pctl: int = 50,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Select the dense core of RP genes for centroid/covariance fitting.

    Uses a 2-D KDE on the first two COA axes to evaluate each RP gene's
    local density, then retains only those above the *density_pctl*-th
    percentile.  This prevents compositionally drifted RP genes (which
    sit near the genome center) from pulling the centroid and inflating
    the covariance.

    Returns (X_core, core_gene_ids, core_mask) where *core_mask* is a
    boolean array over the original X_rp rows.
    """
    from scipy.stats import gaussian_kde

    n_rp = len(X_rp)
    if n_rp < 8:
        # Too few for a meaningful KDE; keep all
        return X_rp, rp_gene_ids_list, np.ones(n_rp, dtype=bool)

    X_2d = X_rp[:, :2]
    try:
        kde = gaussian_kde(X_2d.T, bw_method="scott")
        densities = kde(X_2d.T)
    except Exception:
        return X_rp, rp_gene_ids_list, np.ones(n_rp, dtype=bool)

    threshold = np.percentile(densities, density_pctl)
    core_mask = densities >= threshold

    # Guarantee at least n_axes+1 core genes (needed for covariance fitting)
    n_core = int(core_mask.sum())
    min_core = max(5, X_rp.shape[1] + 1)
    if n_core < min_core:
        # Fall back to keeping the top min_core by density
        top_idx = np.argsort(densities)[::-1][:min_core]
        core_mask = np.zeros(n_rp, dtype=bool)
        core_mask[top_idx] = True

    X_core = X_rp[core_mask]
    core_ids = [gid for gid, sel in zip(rp_gene_ids_list, core_mask) if sel]

    logger.info(
        "RP dense-core selection: %d/%d genes above %dth-pctl density threshold",
        int(core_mask.sum()), n_rp, density_pctl,
    )

    return X_core, core_ids, core_mask


# ---------------------------------------------------------------------------
# RP sub-cluster detection
# ---------------------------------------------------------------------------

def _detect_rp_subclusters(
    X_rp: np.ndarray,
    rp_gene_ids_list: list[str],
    max_k: int = _RP_SUBCLUSTER_MAX_K,
) -> list[dict]:
    """Detect RP sub-populations via GMM+BIC on first 2 COA axes.

    Returns a list of sub-cluster dicts, each with keys:
      X, gene_ids, label, n, avg_dist
    Sorted by n (largest first).  Returns a single-element list if
    no split is detected.
    """
    n_rp = len(X_rp)
    if n_rp < _RP_SUBCLUSTER_MIN:
        return [{"X": X_rp, "gene_ids": rp_gene_ids_list, "label": 0,
                 "n": n_rp, "avg_dist": 0.0}]

    X_2d = X_rp[:, :2] if X_rp.shape[1] >= 2 else X_rp

    best_k, best_bic = 1, np.inf
    gmm_best = None
    for k in range(1, min(max_k + 1, n_rp)):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42,
                                  covariance_type="full", n_init=3)
            gmm.fit(X_2d)
            bic = gmm.bic(X_2d)
            if bic < best_bic:
                best_bic, best_k, gmm_best = bic, k, gmm
        except Exception:
            pass

    if best_k <= 1 or gmm_best is None:
        return [{"X": X_rp, "gene_ids": rp_gene_ids_list, "label": 0,
                 "n": n_rp, "avg_dist": 0.0}]

    labels = gmm_best.predict(X_2d)
    min_members = max(3, int(_RP_SUBCLUSTER_MIN_FRAC * n_rp))
    subclusters = []

    for lab in sorted(set(labels)):
        mask = labels == lab
        X_sub = X_rp[mask]
        n_sub = len(X_sub)
        if n_sub < min_members:
            continue
        centroid_sub = X_sub.mean(axis=0)
        avg_dist = float(np.mean(np.linalg.norm(X_sub - centroid_sub, axis=1)))
        sub_ids = [gid for gid, sel in zip(rp_gene_ids_list, mask) if sel]
        subclusters.append({"X": X_sub, "gene_ids": sub_ids, "label": int(lab),
                            "n": n_sub, "avg_dist": avg_dist})

    if not subclusters:
        return [{"X": X_rp, "gene_ids": rp_gene_ids_list, "label": 0,
                 "n": n_rp, "avg_dist": 0.0}]

    # Sort by number of RP genes (largest first)
    subclusters.sort(key=lambda s: -s["n"])

    logger.info(
        "RP sub-cluster detection: BIC favours k=%d, %d qualifying sub-clusters "
        "(sizes: %s)",
        best_k, len(subclusters), [s["n"] for s in subclusters],
    )
    return subclusters


# ---------------------------------------------------------------------------
# Density-peak detection (supports multiple peaks)
# ---------------------------------------------------------------------------

def _detect_density_peaks(
    X: np.ndarray,
    gene_ids: list[str],
    n_axes: int,
    bandwidth: str = _DENSITY_KDE_BANDWIDTH,
    seed_radius_pctl: int = _DENSITY_SEED_RADIUS_PCTL,
) -> list[dict]:
    """Find one or more density peaks in COA space via 2-D KDE.

    Returns a list of peak dicts, each with:
      centroid, cov, cov_inv, seed_gene_ids, peak_xy, n_seed
    Sorted by seed size (largest first).
    """
    from scipy.stats import gaussian_kde
    from scipy.ndimage import label as ndimage_label, maximum_position

    # KDE on first 2 axes
    x2d = X[:, :2].T
    try:
        kde = gaussian_kde(x2d, bw_method=bandwidth)
    except Exception as e:
        logger.warning("Density KDE failed: %s; using median as single peak", e)
        return [_build_single_density_peak(X, gene_ids, n_axes, np.median(X[:, :2], axis=0),
                                           seed_radius_pctl)]

    # Evaluate on a grid
    grid_n = 200
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    pad_x, pad_y = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    gx = np.linspace(x_min - pad_x, x_max + pad_x, grid_n)
    gy = np.linspace(y_min - pad_y, y_max + pad_y, grid_n)
    gxx, gyy = np.meshgrid(gx, gy)
    grid_pts = np.vstack([gxx.ravel(), gyy.ravel()])
    density = kde(grid_pts).reshape(grid_n, grid_n)

    # Find multiple peaks: threshold at 50% of max density, label connected regions
    threshold_density = density.max() * 0.5
    binary = density > threshold_density
    labeled, n_regions = ndimage_label(binary)

    peaks_2d = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        # Peak is the max-density point within this region
        region_density = np.where(region_mask, density, 0.0)
        peak_idx = np.unravel_index(np.argmax(region_density), density.shape)
        peak_2d = np.array([gx[peak_idx[1]], gy[peak_idx[0]]])
        peaks_2d.append(peak_2d)

    if not peaks_2d:
        # Fallback: single global peak
        peak_idx = np.argmax(density)
        peak_2d = grid_pts[:, np.argmax(density.ravel())]
        peaks_2d = [peak_2d]

    # Build a cluster for each peak
    min_seed_genes = max(10, int(_DENSITY_MULTI_PEAK_MIN_FRAC * len(gene_ids)))
    results = []
    for peak_2d in peaks_2d:
        peak_info = _build_single_density_peak(
            X, gene_ids, n_axes, peak_2d, seed_radius_pctl,
        )
        if peak_info["n_seed"] >= min_seed_genes:
            results.append(peak_info)

    if not results:
        # Fall back to the global peak
        global_peak = grid_pts[:, np.argmax(density.ravel())]
        results = [_build_single_density_peak(X, gene_ids, n_axes, global_peak, seed_radius_pctl)]

    # Sort by seed size (most genes first)
    results.sort(key=lambda p: -p["n_seed"])

    logger.info(
        "Density-peak detection: %d qualifying peaks (seed sizes: %s)",
        len(results), [p["n_seed"] for p in results],
    )
    return results


def _build_single_density_peak(
    X: np.ndarray,
    gene_ids: list[str],
    n_axes: int,
    peak_2d: np.ndarray,
    seed_radius_pctl: int,
) -> dict:
    """Build centroid/covariance for a single density peak."""
    # Full-dimensional peak vector
    if n_axes > 2:
        peak_full = np.concatenate([peak_2d, np.median(X[:, 2:n_axes], axis=0)])
    else:
        peak_full = peak_2d.copy()

    # Seed genes within percentile radius
    dists = np.linalg.norm(X[:, :n_axes] - peak_full, axis=1)
    radius = float(np.percentile(dists, seed_radius_pctl))
    seed_mask = dists <= radius
    seed_ids = [gid for gid, s in zip(gene_ids, seed_mask) if s]
    X_seed = X[seed_mask][:, :n_axes]

    # Fit covariance on seed
    n_seed = len(X_seed)
    if n_seed >= _MIN_RP_FOR_ROBUST:
        try:
            sf = max(0.5, min(0.9, (n_seed - 2) / n_seed))
            mcd = MinCovDet(random_state=42, support_fraction=sf).fit(X_seed)
            centroid, cov = mcd.location_, mcd.covariance_
        except Exception:
            centroid, cov = X_seed.mean(axis=0), _empirical_cov(X_seed)
    else:
        centroid = X_seed.mean(axis=0) if n_seed > 0 else peak_full
        cov = _empirical_cov(X_seed) if n_seed > 1 else np.eye(n_axes)

    return {
        "centroid": centroid,
        "cov": cov,
        "cov_inv": _safe_inv(cov),
        "seed_gene_ids": seed_ids,
        "peak_xy": [float(peak_2d[0]), float(peak_2d[1])],
        "n_seed": n_seed,
        "seed_radius": float(radius),
    }


# ---------------------------------------------------------------------------
# Bootstrap RP centroid stabilisation
# ---------------------------------------------------------------------------

def _bootstrap_rp_centroid(
    X_rp: np.ndarray,
    n_axes: int,
    n_bootstraps: int = _DEFAULT_N_BOOTSTRAPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap the RP centroid and covariance for stability.

    Resamples RP genes with replacement B times, fits robust covariance
    each time, and returns the mean centroid, mean covariance, and
    its inverse.
    """
    n_rp = len(X_rp)
    if n_rp < 3:
        centroid = X_rp.mean(axis=0)
        cov = _empirical_cov(X_rp)
        return centroid, cov, _safe_inv(cov)

    rng = np.random.RandomState(42)
    centroids = []
    covs = []

    for b in range(n_bootstraps):
        boot_idx = rng.choice(n_rp, size=n_rp, replace=True)
        X_boot = X_rp[boot_idx]
        c, cv, _, _ = _fit_robust_rp_reference(X_boot, n_axes)
        centroids.append(c)
        covs.append(cv)

    mean_centroid = np.mean(centroids, axis=0)
    mean_cov = np.mean(covs, axis=0)

    logger.info(
        "Bootstrap RP centroid: %d replicates, centroid spread (std) = %.4f",
        n_bootstraps, float(np.std([np.linalg.norm(c - mean_centroid) for c in centroids])),
    )

    return mean_centroid, mean_cov, _safe_inv(mean_cov)


# ---------------------------------------------------------------------------
# Unified cluster fitting
# ---------------------------------------------------------------------------

def _fit_cluster(
    X: np.ndarray,
    gene_ids: list[str],
    centroid: np.ndarray,
    cov: np.ndarray,
    cov_inv: np.ndarray,
    n_axes: int,
    chi2_p: float,
    ffn_path: Path | None,
    rscu_gene_df: pd.DataFrame,
    anchor_gene_ids: set[str] | None = None,
    outlier_mask: np.ndarray | None = None,
    empirical_threshold: float | None = None,
) -> dict:
    """Fit a single Mahalanobis cluster.

    This is the shared engine for both RP and density clusters.

    When *empirical_threshold* is provided (RP cluster), it is used
    directly as the distance boundary.  Otherwise (density cluster),
    the chi-squared quantile at *chi2_p* is used.

    Returns a dict with: distances, threshold, cluster_gene_ids,
    cluster_rscu, probabilities, gene_weights, gene_core_cai,
    core_rare_per_gene, genome_rare_per_gene, n_cluster.
    """
    distances = _compute_mahalanobis_distances(X[:, :n_axes], centroid, cov_inv)
    threshold = empirical_threshold if empirical_threshold is not None else _chi2_threshold(n_axes, chi2_p)
    optimised_mask = distances <= threshold
    cluster_gene_ids = {gid for gid, opt in zip(gene_ids, optimised_mask) if opt}
    probabilities = _distance_to_membership(distances, threshold)

    # Distance-weighted RSCU
    gene_weights = {}
    for gid, d in zip(gene_ids, distances):
        if gid in cluster_gene_ids:
            w = max(1.0 - d / threshold, 0.0) if threshold > 0 else 1.0
            if w > 0:
                gene_weights[gid] = w

    cluster_rscu = pd.Series(dtype=float)
    if ffn_path and ffn_path.exists():
        cluster_rscu = _compute_cluster_rscu(ffn_path, cluster_gene_ids, gene_weights=gene_weights)
    elif cluster_gene_ids:
        rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
        cluster_df = rscu_gene_df[rscu_gene_df["gene"].isin(cluster_gene_ids)].copy()
        if not cluster_df.empty:
            w_arr = cluster_df["gene"].map(gene_weights).fillna(0.0).values
            w_sum = w_arr.sum()
            if w_sum > 0:
                cluster_rscu = cluster_df[rscu_cols].multiply(w_arr, axis=0).sum() / w_sum
            else:
                cluster_rscu = cluster_df[rscu_cols].mean()

    # Core-CAI and adaptation weights (reused for rare codon burden)
    gene_core_cai: dict[str, float] = {}
    adapt_w: dict[str, float] = {}
    if not cluster_rscu.empty and ffn_path and ffn_path.exists():
        adapt_w = _rscu_to_adaptation_weights(cluster_rscu)
        if adapt_w:
            gene_core_cai = _compute_core_cai(ffn_path, adapt_w)

    # Rare codons
    core_rare_per_gene: dict[str, int] = {}
    genome_rare_per_gene: dict[str, int] = {}
    gene_length_codons: dict[str, int] = {}
    rare_codon_burden: dict[str, float] = {}
    if not cluster_rscu.empty and ffn_path and ffn_path.exists():
        core_rare = _identify_rare_codons(cluster_rscu)
        if core_rare:
            core_rare_per_gene = _count_rare_codons_per_gene(ffn_path, core_rare)
        genome_rscu = _compute_genome_wide_rscu(ffn_path)
        if not genome_rscu.empty:
            genome_rare = _identify_rare_codons(genome_rscu)
            if genome_rare:
                genome_rare_per_gene = _count_rare_codons_per_gene(ffn_path, genome_rare)
        gene_length_codons = _compute_gene_lengths_codons(ffn_path)
        if adapt_w:
            rare_codon_burden = _compute_rare_codon_burden(ffn_path, adapt_w)

    # Anchor-gene counts (RP genes in RP cluster, or seed genes in density cluster)
    n_anchor_in_cluster = 0
    if anchor_gene_ids:
        n_anchor_in_cluster = len(cluster_gene_ids & anchor_gene_ids)

    return {
        "centroid": centroid,
        "cov": cov,
        "cov_inv": cov_inv,
        "distances": distances,
        "threshold": threshold,
        "cluster_gene_ids": cluster_gene_ids,
        "cluster_rscu": cluster_rscu,
        "probabilities": probabilities,
        "gene_weights": gene_weights,
        "gene_core_cai": gene_core_cai,
        "core_rare_per_gene": core_rare_per_gene,
        "genome_rare_per_gene": genome_rare_per_gene,
        "gene_length_codons": gene_length_codons,
        "rare_codon_burden": rare_codon_burden,
        "n_cluster": len(cluster_gene_ids),
        "n_anchor_in_cluster": n_anchor_in_cluster,
        "outlier_mask": outlier_mask if outlier_mask is not None else np.array([], dtype=bool),
    }


# ---------------------------------------------------------------------------
# Translational selection strength classification
# ---------------------------------------------------------------------------

def _classify_translational_selection(
    rp_vs_density_cosine: float | None,
    dual_anchor_categories: Counter | None,
    rp_centroid: np.ndarray | None,
    density_centroid: np.ndarray | None,
    gene_spread_median: float | None,
) -> dict:
    """Classify translational selection strength via three criteria."""
    evidence: dict[str, dict] = {}
    strong_votes, weak_votes = 0, 0

    # Criterion 1: RSCU cosine
    if rp_vs_density_cosine is not None and not np.isnan(rp_vs_density_cosine):
        if rp_vs_density_cosine < _TSS_COSINE_STRONG:
            strong_votes += 1; verdict = "strong"
        elif rp_vs_density_cosine > _TSS_COSINE_WEAK:
            weak_votes += 1; verdict = "weak"
        else:
            verdict = "moderate"
        evidence["rscu_cosine"] = {
            "value": round(rp_vs_density_cosine, 4),
            "strong_threshold": f"< {_TSS_COSINE_STRONG}",
            "weak_threshold": f"> {_TSS_COSINE_WEAK}",
            "verdict": verdict,
        }

    # Criterion 2: rp_only fraction
    if dual_anchor_categories is not None:
        n_rp_only = dual_anchor_categories.get("rp_only", 0)
        n_both = dual_anchor_categories.get("both", 0)
        denom = n_rp_only + n_both
        if denom > 0:
            frac = n_rp_only / denom
            if frac > _TSS_RPONLY_FRAC_STRONG:
                strong_votes += 1; verdict = "strong"
            elif frac < _TSS_RPONLY_FRAC_WEAK:
                weak_votes += 1; verdict = "weak"
            else:
                verdict = "moderate"
            evidence["rp_only_fraction"] = {
                "value": round(frac, 4), "n_rp_only": n_rp_only, "n_both": n_both,
                "strong_threshold": f"> {_TSS_RPONLY_FRAC_STRONG}",
                "weak_threshold": f"< {_TSS_RPONLY_FRAC_WEAK}",
                "verdict": verdict,
            }

    # Criterion 3: Centroid separation
    if (rp_centroid is not None and density_centroid is not None
            and gene_spread_median is not None and gene_spread_median > 0):
        min_len = min(len(rp_centroid), len(density_centroid))
        raw_dist = float(np.linalg.norm(rp_centroid[:min_len] - density_centroid[:min_len]))
        normed = raw_dist / gene_spread_median
        if normed > 0.5:
            strong_votes += 1; verdict = "strong"
        elif normed < 0.15:
            weak_votes += 1; verdict = "weak"
        else:
            verdict = "moderate"
        evidence["centroid_separation"] = {
            "raw_distance": round(raw_dist, 4), "normalised": round(normed, 4),
            "strong_threshold": "> 0.5", "weak_threshold": "< 0.15", "verdict": verdict,
        }

    # Overall
    if strong_votes >= 2:
        classification = "strong"
    elif weak_votes >= 2:
        classification = "weak"
    else:
        classification = "moderate"

    caveats = {
        "strong": None,
        "moderate": (
            "Translational selection signal is moderate. RP-based expression "
            "scores (MELP, CAI, Fop) should be interpreted with caution."
        ),
        "weak": (
            "Translational selection signal is weak. RP codon preferences are "
            "nearly indistinguishable from the genome-wide average, so "
            "RP-based expression scores (MELP, CAI, Fop) are unreliable. "
            "Consider alternative expression proxies or external data."
        ),
    }

    result = {
        "classification": classification, "evidence": evidence,
        "strong_votes": strong_votes, "weak_votes": weak_votes,
        "caveat": caveats[classification],
    }
    logger.info("Translational selection strength: %s (strong=%d, weak=%d)",
                classification.upper(), strong_votes, weak_votes)
    if result["caveat"]:
        logger.warning("TSS: %s", result["caveat"])
    return result


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
    """COA scatter colored by RSCU weight with threshold ellipse."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(8, 6))

    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values
    weights = np.clip(1.0 - distances / threshold, 0.0, 1.0) if threshold > 0 else np.ones_like(distances)
    opt_mask = distances <= threshold

    ax.scatter(x[~opt_mask], y[~opt_mask], c="#d0d0d0", alpha=0.25, s=8,
               edgecolors="none", rasterized=True,
               label=f"Non-optimized (n={int((~opt_mask).sum())})")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sc = ax.scatter(x[opt_mask], y[opt_mask], c=weights[opt_mask], cmap="YlGnBu",
                    norm=norm, alpha=0.7, s=14, edgecolors="none", rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("RSCU weight (1 - d/threshold)", fontsize=9)

    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(x[rp_idx], y[rp_idx], facecolors="none", edgecolors="black",
                   s=40, linewidths=0.8,
                   label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5)

    rp_idx_arr = np.array(rp_idx)
    if rp_outlier_mask.any() and len(rp_idx_arr) == len(rp_outlier_mask):
        outlier_idx = rp_idx_arr[rp_outlier_mask]
        if len(outlier_idx) > 0:
            ax.scatter(x[outlier_idx], y[outlier_idx], marker="x", color="#d7191c",
                       s=50, linewidths=1.5,
                       label=f"RP outliers (n={len(outlier_idx)})", zorder=6)

    # Draw convex hull of classified (optimised) genes
    n_opt = int(opt_mask.sum())
    if n_opt >= 3:
        try:
            from scipy.spatial import ConvexHull
            from scipy.interpolate import splprep, splev
            pts = np.column_stack([x[opt_mask], y[opt_mask]])
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            if len(hull.vertices) >= 4:
                tck, _ = splprep([hull_pts[:, 0], hull_pts[:, 1]],
                                 s=0, per=True, k=3)
                t_smooth = np.linspace(0, 1, 200)
                sx, sy = splev(t_smooth, tck)
            else:
                sx, sy = hull_pts[:, 0], hull_pts[:, 1]
            ax.plot(sx, sy, color="#d7191c", linewidth=1.5, linestyle="--",
                    alpha=0.7, zorder=4, label=f"Cluster boundary (n={n_opt})")
        except Exception:
            pass

    ax.scatter(centroid[0], centroid[1], marker="*", s=200, color="#d7191c",
               edgecolors="black", linewidths=0.5, zorder=7, label="RP centroid")

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
    """Histogram of Mahalanobis distances with weight function overlay."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x_max = min(np.percentile(distances, 99), threshold * 4)
    bins = np.linspace(0, x_max, 60)
    ax.hist(distances, bins=bins, color="#2c7bb6", alpha=0.45, label="All genes")
    ax.hist(rp_distances, bins=bins, color="#d7191c", alpha=0.55, label="RP genes")
    ax.axvline(threshold, color="#fdae61", linewidth=2, linestyle="--",
               label=f"Threshold (d={threshold:.2f})")
    ax.axvline(np.median(rp_distances), color="#abd9e9", linewidth=1.5, linestyle=":",
               label=f"Median RP (d={np.median(rp_distances):.2f})")

    n_opt = (distances <= threshold).sum()
    ax.set_xlabel("Mahalanobis distance from RP centroid")
    ax.set_ylabel("Number of genes")

    ax2 = ax.twinx()
    d_line = np.linspace(0, x_max, 300)
    w_line = np.clip(1.0 - d_line / threshold, 0.0, 1.0) if threshold > 0 else np.ones_like(d_line)
    ax2.plot(d_line, w_line, color="#2ca02c", linewidth=2, alpha=0.85, label="RSCU weight")
    ax2.fill_between(d_line, 0, w_line, color="#2ca02c", alpha=0.08)
    ax2.set_ylabel("RSCU weight (1 - d/threshold)", color="#2ca02c", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(-0.05, 1.1)

    ax.set_title(f"{sample_id}: distance distribution ({n_opt} genes in optimized set)")
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
    ax.scatter(x, y, c="#cccccc", alpha=0.15, s=6, edgecolors="none", rasterized=True)

    rp_idx = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids])
    if len(rp_idx) == 0:
        plt.close(fig); return

    if len(rp_idx) == len(rp_outlier_mask):
        non_outlier_rp = rp_idx[~rp_outlier_mask]
        outlier_rp = rp_idx[rp_outlier_mask]
    else:
        non_outlier_rp = rp_idx
        outlier_rp = np.array([])

    ax.scatter(x[non_outlier_rp], y[non_outlier_rp], c="#2c7bb6", s=30, alpha=0.8,
               edgecolors="black", linewidths=0.5,
               label=f"RP non-outlier (n={len(non_outlier_rp)})")
    if len(outlier_rp) > 0:
        ax.scatter(x[outlier_rp], y[outlier_rp], marker="x", c="#d7191c", s=60,
                   linewidths=1.5, label=f"RP outlier (n={len(outlier_rp)})", zorder=5)

    ax.set_xlabel("COA Axis 1"); ax.set_ylabel("COA Axis 2")
    ax.set_title(f"{sample_id}: RP outlier detection (chi2 crit={chi2_crit:.2f})")
    ax.legend(fontsize=8, framealpha=0.7)

    for fmt in FORMATS:
        fig.savefig(output_path.with_suffix(f".{fmt}"), format=fmt, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_dual_anchor_coa(
    coa_coords: pd.DataFrame,
    gene_ids: list[str],
    categories: list[str],
    rp_centroid: np.ndarray,
    density_centroid: np.ndarray,
    rp_cov: np.ndarray,
    density_cov: np.ndarray,
    rp_threshold: float,
    density_threshold: float,
    rp_gene_ids: set[str],
    output_path: Path,
    sample_id: str,
    inertia_pcts: tuple[float, float] = (0.0, 0.0),
) -> None:
    """COA scatter colored by dual-anchor category with both threshold ellipses."""
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(9, 7))
    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values

    cat_colors = {"both": "#1b9e77", "rp_only": "#d95f02",
                  "dens_only": "#7570b3", "neither": "#d0d0d0"}
    cat_labels = {"both": "Both clusters", "rp_only": "RP-cluster only",
                  "dens_only": "Density-cluster only", "neither": "Neither"}
    cat_alphas = {"both": 0.7, "rp_only": 0.7, "dens_only": 0.7, "neither": 0.35}
    cat_sizes = {"both": 14, "rp_only": 14, "dens_only": 14, "neither": 8}

    for cat in ["neither", "dens_only", "rp_only", "both"]:
        mask = np.array([c == cat for c in categories])
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(x[mask], y[mask], c=cat_colors[cat], alpha=cat_alphas[cat],
                   s=cat_sizes[cat], edgecolors="none", rasterized=True,
                   label=f"{cat_labels[cat]} (n={n})")

    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(x[rp_idx], y[rp_idx], facecolors="none", edgecolors="black",
                   s=40, linewidths=0.8,
                   label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5)

    def _draw_ellipse(c2d, cv2d, thresh, color, label, ls):
        try:
            eigvals, eigvecs = np.linalg.eigh(cv2d)
            if np.all(eigvals > 0):
                angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                w = 2 * thresh * np.sqrt(eigvals[1])
                h = 2 * thresh * np.sqrt(eigvals[0])
                ell = mpatches.Ellipse((c2d[0], c2d[1]), w, h, angle=angle,
                                       linewidth=1.8, edgecolor=color, facecolor="none",
                                       linestyle=ls, alpha=0.8, label=label)
                ax.add_patch(ell)
        except Exception:
            pass

    _draw_ellipse(rp_centroid[:2], rp_cov[:2, :2], rp_threshold,
                  "#d95f02", f"RP threshold (d={rp_threshold:.1f})", "--")
    _draw_ellipse(density_centroid[:2], density_cov[:2, :2], density_threshold,
                  "#7570b3", f"Density threshold (d={density_threshold:.1f})", ":")

    ax.scatter(rp_centroid[0], rp_centroid[1], marker="*", s=200, color="#d95f02",
               edgecolors="black", linewidths=0.5, zorder=7, label="RP centroid")
    ax.scatter(density_centroid[0], density_centroid[1], marker="D", s=80,
               color="#7570b3", edgecolors="black", linewidths=0.5, zorder=7,
               label="Density centroid")

    pct1, pct2 = inertia_pcts
    ax.set_xlabel(f"COA Axis 1 ({pct1:.1f}% inertia)")
    ax.set_ylabel(f"COA Axis 2 ({pct2:.1f}% inertia)")
    ax.set_title(f"{sample_id}: dual-anchor Mahalanobis clustering")
    ax.legend(fontsize=7, framealpha=0.7, loc="best")

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
    distance_multiplier: float = _DISTANCE_MULTIPLIER,
) -> dict:
    """Two-cluster Mahalanobis analysis: RP-anchor + density-core.

    Identifies:
      1. An RP cluster: tight cluster around bootstrap-stabilised RP centroid
      2. A density-core cluster: tight cluster around the genome density peak

    Both use chi-squared boundaries for principled, covariance-independent
    thresholds.  When multiple RP sub-populations or density peaks exist,
    the representative cluster is selected by count (most RPs / most genes).

    API and return dict are backward-compatible with the previous implementation.
    """
    mahal_dir = output_dir / "mahal_clustering"
    mahal_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # ── Load RP gene IDs ─────────────────────────────────────────────
    rp_gene_ids: set[str] = set()
    if rp_ids_file and rp_ids_file.exists():
        rp_gene_ids = {
            line.strip() for line in rp_ids_file.read_text().splitlines()
            if line.strip()
        }
    if not rp_gene_ids:
        logger.warning("No RP IDs for %s; clustering cannot proceed.", sample_id)
        return results

    # ── Validate ─────────────────────────────────────────────────────
    n_genes = len(rscu_gene_df)
    if n_genes < _MIN_GENES_FOR_CLUSTERING:
        logger.warning("Too few genes (%d < %d) in %s. Skipping.",
                        n_genes, _MIN_GENES_FOR_CLUSTERING, sample_id)
        return results

    # ── COA ──────────────────────────────────────────────────────────
    logger.info("Clustering: computing COA on %d genes for %s", n_genes, sample_id)
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df=expr_df)
    if not coa_results or "coa_coords" not in coa_results:
        logger.warning("COA failed for %s; skipping", sample_id)
        return results

    coa_coords = coa_results["coa_coords"]
    coa_inertia = coa_results.get("coa_inertia", pd.DataFrame())

    # ── Axis selection ───────────────────────────────────────────────
    n_axes = _select_n_axes(coa_inertia, _MAX_COA_AXES)
    axis_cols = [f"Axis{i+1}" for i in range(n_axes) if f"Axis{i+1}" in coa_coords.columns]
    n_axes = len(axis_cols)
    if n_axes < _MIN_COA_AXES:
        logger.warning("Insufficient COA axes (%d) for %s", n_axes, sample_id)
        return results

    gene_ids = coa_coords["gene"].astype(str).tolist()
    X = coa_coords[axis_cols].values
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() < _MIN_GENES_FOR_CLUSTERING:
        logger.warning("Too few valid genes after NaN removal in %s", sample_id)
        return results
    X = X[valid_mask]
    gene_ids = [g for g, v in zip(gene_ids, valid_mask) if v]

    logger.info("Using %d COA axes (%.1f%% cumulative inertia) for %d genes",
                n_axes,
                coa_inertia["cum_pct"].iloc[n_axes - 1] if len(coa_inertia) >= n_axes else 0,
                len(gene_ids))
    results["mahal_n_axes"] = n_axes

    # ── Extract RP genes in COA space ────────────────────────────────
    rp_indices = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids])
    n_rp_found = len(rp_indices)
    if n_rp_found < 3:
        logger.warning("Only %d RP genes in COA space for %s; need >= 3.", n_rp_found, sample_id)
        return results

    X_rp = X[rp_indices]
    rp_ids_list = [gene_ids[i] for i in rp_indices]
    logger.info("Found %d/%d RP genes in COA space", n_rp_found, len(rp_gene_ids))

    # ══════════════════════════════════════════════════════════════════
    # CLUSTER 1: RP CLUSTER
    # ══════════════════════════════════════════════════════════════════

    # Detect RP sub-populations
    rp_subclusters = _detect_rp_subclusters(X_rp, rp_ids_list)
    results["rp_subcluster_diagnostics"] = {
        "split_detected": len(rp_subclusters) > 1,
        "n_subclusters": len(rp_subclusters),
        "subcluster_sizes": [sc["n"] for sc in rp_subclusters],
        "all_subclusters": rp_subclusters,
    }

    # Fit each RP sub-cluster independently
    rp_fits = []
    for sc_idx, sc in enumerate(rp_subclusters):
        X_sc = sc["X"]
        sc_ids = set(sc["gene_ids"])

        # ── Dense-core selection ──────────────────────────────────────
        # RP genes with genome-average codon usage (near the density
        # centroid) don't carry a translational-selection signal.
        # Fitting centroid/covariance from *all* RP genes lets those
        # drifted members pull the centroid rightward and inflate the
        # covariance.  Instead, identify the dense core of RP genes
        # via 2-D KDE and fit only from those.
        X_core, core_ids, core_mask = _select_rp_dense_core(
            X_sc, sc["gene_ids"], density_pctl=50,
        )

        # Bootstrap-stabilise centroid using only the dense core
        centroid, cov, cov_inv = _bootstrap_rp_centroid(X_core, n_axes)

        # Two-pass outlier removal on the core genes
        _, _, _, outlier_mask_core = _fit_robust_rp_reference(X_core, n_axes)

        # Empirical threshold: compute Mahalanobis distances for the
        # core (non-outlier) RP genes and take the Nth percentile.
        # This gives a tight boundary centred on the densest part of
        # the RP cloud rather than a bloated ellipse stretched by
        # compositionally drifted genes.
        core_ids_set = set(core_ids)
        core_indices = np.array([i for i, g in enumerate(gene_ids) if g in core_ids_set])
        rp_dists_pre = _compute_mahalanobis_distances(X[:, :n_axes], centroid, cov_inv)
        rp_dists_core = rp_dists_pre[core_indices]
        rp_dists_clean = (
            rp_dists_core[~outlier_mask_core]
            if len(outlier_mask_core) == len(rp_dists_core)
            else rp_dists_core
        )
        if len(rp_dists_clean) == 0:
            rp_dists_clean = rp_dists_core
        emp_threshold = float(np.percentile(rp_dists_clean, _RP_EMPIRICAL_PCTL))
        logger.info(
            "  RP sub-cluster %d: empirical threshold=%.3f "
            "(%dth percentile of %d core RP distances, %d/%d core genes)",
            sc_idx, emp_threshold, _RP_EMPIRICAL_PCTL, len(rp_dists_clean),
            int(core_mask.sum()), len(X_sc),
        )

        # Outlier mask for _fit_cluster should span the full sub-cluster
        # (non-core genes are not outliers per se, but aren't used for
        # threshold computation).  We pass the core outlier mask for
        # diagnostic plots.
        outlier_mask = np.zeros(len(X_sc), dtype=bool)
        outlier_mask[core_mask] = outlier_mask_core

        sc_indices = np.array([i for i, g in enumerate(gene_ids) if g in sc_ids])

        fit = _fit_cluster(
            X=X, gene_ids=gene_ids,
            centroid=centroid, cov=cov, cov_inv=cov_inv,
            n_axes=n_axes, chi2_p=_CLUSTER_CHI2_P,
            ffn_path=ffn_path, rscu_gene_df=rscu_gene_df,
            anchor_gene_ids=sc_ids, outlier_mask=outlier_mask,
            empirical_threshold=emp_threshold,
        )
        fit["rp_gene_ids"] = sc_ids
        fit["rp_indices"] = sc_indices
        fit["rp_dists_all"] = fit["distances"][sc_indices]
        fit["n_rp_in_cluster"] = len(fit["cluster_gene_ids"] & sc_ids)
        fit["subcluster_index"] = sc_idx

        rp_fits.append(fit)
        logger.info(
            "  RP sub-cluster %d: %d RPs, cluster=%d genes (%d RPs in cluster)",
            sc_idx, sc["n"], fit["n_cluster"], fit["n_rp_in_cluster"],
        )

    results["rp_subclusters"] = rp_fits

    # Select representative: most RP genes in cluster
    primary = max(rp_fits, key=lambda f: f["n_rp_in_cluster"])

    # Unpack primary RP cluster
    centroid = primary["centroid"]
    cov = primary["cov"]
    cov_inv = primary["cov_inv"]
    rp_outlier_mask = primary["outlier_mask"]
    rp_indices = primary["rp_indices"]
    rp_gene_ids = primary["rp_gene_ids"]
    distances = primary["distances"]
    threshold = primary["threshold"]
    rp_dists_all = primary["rp_dists_all"]
    cluster_gene_ids = primary["cluster_gene_ids"]
    cluster_rscu = primary["cluster_rscu"]
    probabilities = primary["probabilities"]
    gene_weights = primary["gene_weights"]
    gene_core_cai = primary["gene_core_cai"]
    core_rare_per_gene = primary["core_rare_per_gene"]
    genome_rare_per_gene = primary["genome_rare_per_gene"]
    gene_length_codons = primary["gene_length_codons"]
    rare_codon_burden = primary["rare_codon_burden"]
    n_cluster = primary["n_cluster"]
    n_rp_in_cluster = primary["n_rp_in_cluster"]

    median_rp_dist = float(np.median(rp_dists_all[~rp_outlier_mask])) if rp_outlier_mask.any() and len(rp_dists_all) == len(rp_outlier_mask) else float(np.median(rp_dists_all))

    # Backward-compatible threshold diagnostics
    threshold_diag = {
        "method": "chi2",
        "chi2_p": _CLUSTER_CHI2_P,
        "chi2_df": n_axes,
        "threshold": round(threshold, 4),
        "effective_multiplier": round(threshold / median_rp_dist, 4) if median_rp_dist > 0 else 0.0,
        "projection": "full",
    }
    results["threshold_diagnostics"] = threshold_diag

    # RP-vs-cluster cosine similarity
    rp_cosine_sim = np.nan
    if rp_rscu_df is not None and not rp_rscu_df.empty:
        rscu_cols = [c for c in RSCU_COLUMN_NAMES
                     if c in rp_rscu_df.columns and c in cluster_rscu.index]
        if rscu_cols:
            rp_only_rscu = rp_rscu_df[rscu_cols].mean()
            rp_cosine_sim = 1.0 - cosine_dist(
                cluster_rscu[rscu_cols].fillna(0).values,
                rp_only_rscu.fillna(0).values,
            )

    logger.info(
        "RP cluster: threshold=%.2f (chi2, p=%.2f, df=%d), "
        "%d genes (%d RPs + %d non-RP)",
        threshold, _CLUSTER_CHI2_P, n_axes,
        n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster,
    )

    if n_cluster < _MIN_CLUSTER_SIZE:
        logger.warning("RP cluster is small (%d genes); expression scoring may be unreliable", n_cluster)

    # Store RP-anchor geometry
    results["rp_centroid"] = centroid
    results["rp_cov"] = cov
    results["rp_threshold"] = threshold
    results["rp_gene_ids"] = rp_gene_ids

    results["mahal_cluster_gene_ids"] = cluster_gene_ids
    results["mahal_rp_cluster"] = 1
    optimised_mask = distances <= threshold
    labels = optimised_mask.astype(int)
    results["mahal_labels"] = labels
    results["mahal_best_k"] = 2
    results["mahal_bic_scores"] = []
    results["mahal_probabilities"] = probabilities
    results["mahal_cluster_rscu"] = cluster_rscu
    results["gene_core_cai"] = gene_core_cai
    results["mahal_rp_cosine_sim"] = rp_cosine_sim

    # ── Save RP cluster outputs ──────────────────────────────────────
    # Compute length-normalized rare codon frequency
    _rare_freq = {}
    for gid in gene_ids:
        n_rare = core_rare_per_gene.get(gid, 0)
        n_codons = gene_length_codons.get(gid, 0)
        _rare_freq[gid] = n_rare / n_codons if n_codons > 0 else np.nan

    cluster_df = pd.DataFrame({
        "gene": gene_ids,
        "mahalanobis_distance": distances,
        "core_CAI": [gene_core_cai.get(gid, np.nan) for gid in gene_ids],
        "gene_length_codons": [gene_length_codons.get(gid, 0) for gid in gene_ids],
        "n_core_rare_codons": [core_rare_per_gene.get(gid, 0) for gid in gene_ids],
        "n_genome_rare_codons": [genome_rare_per_gene.get(gid, 0) for gid in gene_ids],
        "rare_codon_freq": [round(_rare_freq.get(gid, np.nan), 5) for gid in gene_ids],
        "rare_codon_burden": [round(rare_codon_burden.get(gid, np.nan), 5) for gid in gene_ids],
        "membership_score": probabilities[:, 1],
        "in_optimized_set": [gid in cluster_gene_ids for gid in gene_ids],
        "is_ribosomal_protein": [gid in rp_gene_ids for gid in gene_ids],
    })
    cluster_path = mahal_dir / f"{sample_id}_mahal_clusters.tsv"
    cluster_df.to_csv(cluster_path, sep="\t", index=False)
    results["mahal_clusters_path"] = cluster_path

    rscu_path = mahal_dir / f"{sample_id}_mahal_cluster_rscu.tsv"
    cluster_rscu.to_frame("RSCU").to_csv(rscu_path, sep="\t")
    results["mahal_cluster_rscu_path"] = rscu_path

    ids_path = mahal_dir / f"{sample_id}_mahal_cluster_ids.txt"
    ids_path.write_text("\n".join(sorted(cluster_gene_ids)) + "\n")
    results["mahal_cluster_ids_path"] = ids_path

    # Summary stats
    mean_weight = float(np.mean(list(gene_weights.values()))) if gene_weights else 0.0
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
        "rp_outliers_removed": int(rp_outlier_mask.sum()) if len(rp_outlier_mask) > 0 else 0,
        "mahalanobis_threshold": round(threshold, 4),
        "threshold_method": "chi2",
        "threshold_projection": "full",
        "threshold_effective_multiplier": threshold_diag["effective_multiplier"],
        "threshold_kde_valley": None,
        "threshold_2d": None,
        "threshold_n_genes_inside_2d": None,
        "threshold_otsu": None,
        "median_rp_distance": round(median_rp_dist, 4),
        "rscu_pooling": "distance-weighted",
        "mean_rscu_weight": round(mean_weight, 4),
        "rp_cosine_similarity": round(rp_cosine_sim, 4) if not np.isnan(rp_cosine_sim) else None,
    }
    summary_df = pd.DataFrame([summary])
    summary_path = mahal_dir / f"{sample_id}_mahal_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    results["mahal_summary_path"] = summary_path

    # COA coords with assignments
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
    results["mahal_coa_inertia"] = coa_inertia
    results["mahal_gene_distances"] = pd.Series(distances, index=gene_ids, name="mahalanobis_distance")

    # ── RP cluster diagnostic plots ──────────────────────────────────
    try:
        inertia_pcts_2 = (0.0, 0.0)
        if len(coa_inertia) >= 2:
            inertia_pcts_2 = (float(coa_inertia["pct_inertia"].iloc[0]),
                              float(coa_inertia["pct_inertia"].iloc[1]))

        coa_filtered = coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True)

        _plot_coa_mahalanobis(
            coa_filtered, distances, gene_ids, rp_gene_ids,
            rp_indices, rp_outlier_mask, threshold, centroid, cov,
            mahal_dir / f"{sample_id}_mahal_coa_mahalanobis", sample_id,
            inertia_pcts=inertia_pcts_2,
        )
        results["mahal_coa_plot"] = mahal_dir / f"{sample_id}_mahal_coa_mahalanobis.png"

        _plot_distance_histogram(
            distances, rp_dists_all, threshold,
            mahal_dir / f"{sample_id}_mahal_distance_histogram", sample_id,
        )
        results["mahal_separation_plot"] = mahal_dir / f"{sample_id}_mahal_distance_histogram.png"

        chi2_crit = np.sqrt(chi2.ppf(1 - _RP_OUTLIER_ALPHA, df=n_axes))
        _plot_rp_outlier_detection(
            coa_filtered, rp_gene_ids, gene_ids, rp_outlier_mask,
            rp_indices, chi2_crit,
            mahal_dir / f"{sample_id}_mahal_rp_outliers", sample_id,
        )
        results["mahal_rp_outlier_plot"] = mahal_dir / f"{sample_id}_mahal_rp_outliers.png"
    except Exception as e:
        logger.warning("RP diagnostic plots failed: %s", e, exc_info=True)

    # ══════════════════════════════════════════════════════════════════
    # CLUSTER 2: DENSITY-CORE CLUSTER
    # ══════════════════════════════════════════════════════════════════

    try:
        density_peaks = _detect_density_peaks(X, gene_ids, n_axes)

        # Fit each density peak
        density_fits = []
        for dp_idx, dp in enumerate(density_peaks):
            dp_fit = _fit_cluster(
                X=X, gene_ids=gene_ids,
                centroid=dp["centroid"], cov=dp["cov"], cov_inv=dp["cov_inv"],
                n_axes=n_axes, chi2_p=_CLUSTER_CHI2_P,
                ffn_path=ffn_path, rscu_gene_df=rscu_gene_df,
                anchor_gene_ids=set(dp["seed_gene_ids"]),
            )
            dp_fit["peak_xy"] = dp["peak_xy"]
            dp_fit["n_seed"] = dp["n_seed"]
            dp_fit["seed_radius"] = dp["seed_radius"]
            dp_fit["seed_gene_ids"] = dp["seed_gene_ids"]
            density_fits.append(dp_fit)
            logger.info(
                "  Density peak %d: seed=%d, cluster=%d genes",
                dp_idx, dp["n_seed"], dp_fit["n_cluster"],
            )

        # Select representative: most genes in cluster
        density_primary = max(density_fits, key=lambda f: f["n_cluster"])

        density_centroid = density_primary["centroid"]
        density_cov = density_primary["cov"]
        density_cov_inv = density_primary["cov_inv"]
        density_distances = density_primary["distances"]
        density_threshold = density_primary["threshold"]
        density_cluster_gene_ids = density_primary["cluster_gene_ids"]
        density_probabilities = density_primary["probabilities"]

        results["density_anchor_diagnostics"] = {
            "density_peak_xy": density_primary["peak_xy"],
            "n_seed_genes": density_primary["n_seed"],
            "seed_radius": density_primary["seed_radius"],
            "bandwidth": str(_DENSITY_KDE_BANDWIDTH),
            "n_peaks_detected": len(density_fits),
            "peak_cluster_sizes": [f["n_cluster"] for f in density_fits],
        }

        density_thresh_diag = {
            "method": "chi2",
            "chi2_alpha": _CLUSTER_CHI2_P,
            "chi2_df": n_axes,
            "chi2_critical_value": round(density_threshold, 4),
        }
        results["density_threshold_diagnostics"] = density_thresh_diag

        logger.info("Density-core cluster: threshold=%.3f (chi2, p=%.2f, df=%d), %d genes",
                     density_threshold, _CLUSTER_CHI2_P, n_axes, len(density_cluster_gene_ids))

        results["density_cluster_gene_ids"] = density_cluster_gene_ids
        results["density_distances"] = density_distances
        results["density_threshold"] = density_threshold
        results["density_centroid"] = density_centroid
        results["density_cov"] = density_cov
        results["density_probabilities"] = density_probabilities

        # ── Dual-anchor comparison ───────────────────────────────────
        rp_set = cluster_gene_ids
        dens_set = density_cluster_gene_ids

        categories = []
        for gid in gene_ids:
            in_rp = gid in rp_set
            in_dens = gid in dens_set
            if in_rp and in_dens:
                categories.append("both")
            elif in_rp:
                categories.append("rp_only")
            elif in_dens:
                categories.append("dens_only")
            else:
                categories.append("neither")

        dual_df = pd.DataFrame({
            "gene": gene_ids,
            "rp_mahal_distance": distances,
            "rp_membership": probabilities[:, 1],
            "in_rp_cluster": [gid in rp_set for gid in gene_ids],
            "density_mahal_distance": density_distances,
            "density_membership": density_probabilities[:, 1],
            "in_density_cluster": [gid in dens_set for gid in gene_ids],
            "dual_category": categories,
            "is_ribosomal_protein": [gid in rp_gene_ids for gid in gene_ids],
        })

        cat_counts = Counter(categories)
        logger.info("Dual-anchor: both=%d, rp_only=%d, dens_only=%d, neither=%d",
                     cat_counts.get("both", 0), cat_counts.get("rp_only", 0),
                     cat_counts.get("dens_only", 0), cat_counts.get("neither", 0))

        dual_path = mahal_dir / f"{sample_id}_dual_anchor_comparison.tsv"
        dual_df.to_csv(dual_path, sep="\t", index=False)
        results["dual_anchor_path"] = dual_path
        results["dual_anchor_df"] = dual_df
        results["dual_anchor_categories"] = cat_counts

        # Density cluster RSCU and cosine with RP cluster
        density_cluster_rscu = density_primary["cluster_rscu"]
        if not density_cluster_rscu.empty:
            density_rscu_path = mahal_dir / f"{sample_id}_density_cluster_rscu.tsv"
            density_cluster_rscu.to_frame("RSCU").to_csv(density_rscu_path, sep="\t")
            results["density_cluster_rscu"] = density_cluster_rscu
            results["density_cluster_rscu_path"] = density_rscu_path

            shared_cols = [c for c in cluster_rscu.index if c in density_cluster_rscu.index]
            if shared_cols and not cluster_rscu.empty:
                anchor_cosine = 1.0 - cosine_dist(
                    cluster_rscu[shared_cols].fillna(0).values,
                    density_cluster_rscu[shared_cols].fillna(0).values,
                )
                results["rp_vs_density_rscu_cosine"] = float(anchor_cosine)
                logger.info("RP-vs-density RSCU cosine: %.4f", anchor_cosine)

        # Density summary
        _seed_mask = np.array([gid in density_primary["seed_gene_ids"] for gid in gene_ids])
        density_median = float(np.median(density_distances[_seed_mask])) if _seed_mask.any() else 0.0
        density_summary = {
            "density_peak_axis1": density_primary["peak_xy"][0],
            "density_peak_axis2": density_primary["peak_xy"][1],
            "density_seed_genes": density_primary["n_seed"],
            "density_seed_radius": density_primary["seed_radius"],
            "density_threshold": round(density_threshold, 4),
            "density_median_seed_dist": round(density_median, 4),
            "density_cluster_size": len(density_cluster_gene_ids),
            "n_both": cat_counts.get("both", 0),
            "n_rp_only": cat_counts.get("rp_only", 0),
            "n_dens_only": cat_counts.get("dens_only", 0),
            "n_neither": cat_counts.get("neither", 0),
            "rp_centroid_to_density_centroid": round(
                float(np.linalg.norm(
                    centroid[:min(len(centroid), len(density_centroid))]
                    - density_centroid[:min(len(centroid), len(density_centroid))]
                )), 4),
        }
        pd.DataFrame([density_summary]).to_csv(
            mahal_dir / f"{sample_id}_density_anchor_summary.tsv", sep="\t", index=False)
        results["density_summary_path"] = mahal_dir / f"{sample_id}_density_anchor_summary.tsv"

        # ── Translational selection strength ─────────────────────────
        _gene_centre = X[:, :n_axes].mean(axis=0)
        _gene_dists = np.linalg.norm(X[:, :n_axes] - _gene_centre, axis=1)
        _gene_spread_median = float(np.median(_gene_dists))

        tss = _classify_translational_selection(
            rp_vs_density_cosine=results.get("rp_vs_density_rscu_cosine"),
            dual_anchor_categories=cat_counts,
            rp_centroid=centroid,
            density_centroid=density_centroid,
            gene_spread_median=_gene_spread_median,
        )
        results["translational_selection_strength"] = tss

        # Append TSS to summary
        if summary_path.exists():
            try:
                s_df = pd.read_csv(summary_path, sep="\t")
                s_df["translational_selection"] = tss["classification"]
                s_df["tss_caveat"] = tss["caveat"] if tss["caveat"] else ""
                s_df.to_csv(summary_path, sep="\t", index=False)
            except Exception:
                pass

        tss_path = mahal_dir / f"{sample_id}_translational_selection_strength.tsv"
        tss_flat = {
            "sample_id": sample_id, "classification": tss["classification"],
            "strong_votes": tss["strong_votes"], "weak_votes": tss["weak_votes"],
            "caveat": tss["caveat"] if tss["caveat"] else "",
        }
        for criterion, info in tss["evidence"].items():
            for k, v in info.items():
                tss_flat[f"{criterion}_{k}"] = v
        pd.DataFrame([tss_flat]).to_csv(tss_path, sep="\t", index=False)
        results["tss_path"] = tss_path

        # ── Dual-anchor plot ─────────────────────────────────────────
        try:
            _plot_dual_anchor_coa(
                coa_coords=coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True),
                gene_ids=gene_ids, categories=categories,
                rp_centroid=centroid, density_centroid=density_centroid,
                rp_cov=cov, density_cov=density_cov,
                rp_threshold=threshold, density_threshold=density_threshold,
                rp_gene_ids=rp_gene_ids,
                output_path=mahal_dir / f"{sample_id}_dual_anchor_coa",
                sample_id=sample_id,
                inertia_pcts=(
                    (float(coa_inertia["pct_inertia"].iloc[0]),
                     float(coa_inertia["pct_inertia"].iloc[1]))
                    if len(coa_inertia) >= 2 else (0.0, 0.0)
                ),
            )
            results["dual_anchor_plot"] = mahal_dir / f"{sample_id}_dual_anchor_coa.png"
        except Exception as e:
            logger.warning("Dual-anchor plot failed: %s", e, exc_info=True)

    except Exception as e:
        logger.warning("Density-core cluster failed for %s: %s", sample_id, e, exc_info=True)

    logger.info(
        "Clustering complete for %s: RP cluster=%d genes (%d RPs + %d non-RP), threshold=%.2f",
        sample_id, n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster, threshold,
    )

    return results
