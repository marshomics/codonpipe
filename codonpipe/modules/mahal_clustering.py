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

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import chi2
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import MinCovDet
from sklearn.metrics import silhouette_score

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

_MIN_GENES_FOR_CLUSTERING = 50       # Minimum genes to attempt clustering
_MIN_COA_AXES = 2             # Minimum COA axes required
_MAX_COA_AXES = 8             # Maximum COA axes to use
_CUMULATIVE_INERTIA_TARGET = 0.80  # Target cumulative inertia for axis selection
_MIN_CLUSTER_SIZE = 10        # Warn if optimized set has fewer genes than this

# RP-anchored Mahalanobis approach
_RP_OUTLIER_ALPHA = 0.025     # Chi-squared alpha for RP outlier detection
_DISTANCE_MULTIPLIER = 2.0    # Default threshold = multiplier x median RP distance (fallback)
_MIN_RP_FOR_ROBUST = 10       # Min RP genes for MinCovDet; else empirical covariance
_ADAPTIVE_MULT_MIN = 1.2      # Floor for adaptive multiplier (relative to median RP dist)
_ADAPTIVE_MULT_MAX = 3.0      # Ceiling for adaptive multiplier
_ADAPTIVE_KDE_N_POINTS = 512  # Resolution for 1-D KDE on distance distribution
_ADAPTIVE_MIN_GENES = 100     # Minimum genes to attempt adaptive threshold

# RP sub-cluster detection
_RP_SUBCLUSTER_MIN = 15       # Minimum RP genes to attempt sub-cluster detection
_RP_SUBCLUSTER_MAX_K = 4      # Maximum sub-clusters to test
_RP_SUBCLUSTER_SIL_THRESH = 0.45  # Silhouette score above which we accept a split
_RP_SUBCLUSTER_MIN_FRAC = 0.20    # Sub-cluster must contain >= 20% of total RPs

# Density-anchor mode
_DENSITY_KDE_BANDWIDTH = "scott"  # KDE bandwidth for density peak detection
_DENSITY_SEED_RADIUS_PCTL = 30   # Percentile of distances for initial seed set
_DENSITY_MULTIPLIER = 2.5         # Threshold = multiplier × median seed distance

# Translational selection strength classification
# Thresholds are applied in order: "strong" requires ALL strong criteria,
# "weak" if ANY weak criterion is met, otherwise "moderate".
_TSS_COSINE_STRONG = 0.90   # RP-vs-density RSCU cosine BELOW this → strong selection
_TSS_COSINE_WEAK = 0.97     # ABOVE this → weak selection
_TSS_RPONLY_FRAC_STRONG = 0.05   # rp_only/(rp_only+both) ABOVE this → strong
_TSS_RPONLY_FRAC_WEAK = 0.01     # BELOW this → weak
_TSS_CENTROID_DIST_PCTL = 50     # Centroid distance normalised by median gene spread


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


def _select_rp_subcluster(
    X_rp: np.ndarray,
    rp_gene_ids_list: list[str],
    max_k: int = _RP_SUBCLUSTER_MAX_K,
    sil_threshold: float = _RP_SUBCLUSTER_SIL_THRESH,
) -> tuple[np.ndarray, list[str], dict]:
    """Detect and select the tightest RP sub-cluster when RPs are split.

    Uses agglomerative clustering (Ward linkage) to test whether the RP set
    in COA space is better described by k=2..max_k sub-clusters than a single
    group.  If the best silhouette score exceeds ``sil_threshold``, the
    sub-cluster with the smallest average within-cluster distance (i.e. the
    densest, most tightly-clustered group) is selected as the anchor for
    Mahalanobis fitting.

    Args:
        X_rp: (n_rp, n_axes) RP gene coordinates in COA space.
        rp_gene_ids_list: Gene IDs corresponding to rows of X_rp.
        max_k: Maximum number of sub-clusters to evaluate.
        sil_threshold: Minimum silhouette score to accept a split.

    Returns:
        X_rp_selected: Coordinates of selected sub-cluster.
        selected_ids: Gene IDs in the selected sub-cluster.
        diagnostics: Dict with detection details for logging/export.
    """
    n_rp = len(X_rp)
    diag: dict = {
        "attempted": False,
        "split_detected": False,
        "best_k": 1,
        "best_silhouette": -1.0,
        "selected_subcluster": -1,
        "n_original": n_rp,
        "n_selected": n_rp,
        "subcluster_sizes": [n_rp],
    }

    if n_rp < _RP_SUBCLUSTER_MIN:
        logger.debug(
            "Too few RPs (%d < %d) for sub-cluster detection; using all",
            n_rp, _RP_SUBCLUSTER_MIN,
        )
        return X_rp, rp_gene_ids_list, diag

    diag["attempted"] = True

    # Test k=2..max_k with agglomerative (Ward) clustering
    best_k = 1
    best_sil = -1.0
    best_labels = None

    for k in range(2, min(max_k + 1, n_rp)):
        try:
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = agg.fit_predict(X_rp)
            sil = silhouette_score(X_rp, labels)
            if sil > best_sil:
                best_sil = sil
                best_k = k
                best_labels = labels
        except Exception as e:
            logger.debug("Agglomerative k=%d failed: %s", k, e)

    diag["best_k"] = best_k
    diag["best_silhouette"] = round(float(best_sil), 4)

    if best_sil < sil_threshold or best_labels is None:
        logger.info(
            "RP sub-cluster detection: best silhouette=%.3f (k=%d) below "
            "threshold %.2f; RPs form a single coherent group",
            best_sil, best_k, sil_threshold,
        )
        return X_rp, rp_gene_ids_list, diag

    # Split detected — select the best anchor sub-cluster
    diag["split_detected"] = True
    unique_labels = sorted(set(best_labels))
    subcluster_sizes = [int((best_labels == lab).sum()) for lab in unique_labels]
    diag["subcluster_sizes"] = subcluster_sizes

    # For each sub-cluster that meets the minimum size requirement,
    # compute a density score = avg_dist_to_centroid / sqrt(n_members).
    # Dividing by sqrt(n) penalises tiny clusters whose low spread is
    # simply an artifact of having few points.  The sub-cluster with the
    # lowest score (tightest per its size) is selected.
    min_members = max(3, int(_RP_SUBCLUSTER_MIN_FRAC * n_rp))
    best_sub = -1
    best_score = np.inf
    best_density = np.inf
    for lab in unique_labels:
        mask = best_labels == lab
        X_sub = X_rp[mask]
        n_sub = len(X_sub)
        if n_sub < min_members:
            continue
        centroid_sub = X_sub.mean(axis=0)
        avg_dist = float(np.mean(np.linalg.norm(X_sub - centroid_sub, axis=1)))
        score = avg_dist / np.sqrt(n_sub)
        if score < best_score:
            best_score = score
            best_density = avg_dist
            best_sub = lab

    if best_sub < 0:
        logger.warning("No sub-cluster had >= 3 members; using all RPs")
        return X_rp, rp_gene_ids_list, diag

    selected_mask = best_labels == best_sub
    X_selected = X_rp[selected_mask]
    selected_ids = [gid for gid, sel in zip(rp_gene_ids_list, selected_mask) if sel]

    diag["selected_subcluster"] = int(best_sub)
    diag["n_selected"] = len(selected_ids)

    excluded_ids = [gid for gid, sel in zip(rp_gene_ids_list, selected_mask) if not sel]

    logger.info(
        "RP sub-cluster split detected (silhouette=%.3f, k=%d): "
        "selected densest sub-cluster (%d RPs, avg dist=%.3f), "
        "excluded %d RPs in %d other sub-cluster(s). Sizes: %s",
        best_sil, best_k, len(selected_ids), best_density,
        len(excluded_ids), best_k - 1, subcluster_sizes,
    )

    return X_selected, selected_ids, diag


def _find_density_peak_anchor(
    X: np.ndarray,
    gene_ids: list[str],
    n_axes: int,
    bandwidth: str = _DENSITY_KDE_BANDWIDTH,
    seed_radius_pctl: int = _DENSITY_SEED_RADIUS_PCTL,
    min_rp: int = _MIN_RP_FOR_ROBUST,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Detect the genome density peak in COA space and fit covariance on nearby genes.

    Uses a 2-D KDE on the first two COA axes to locate the mode of the
    genome-wide gene distribution.  Genes within the *seed_radius_pctl*
    percentile distance from that peak are used as the seed set for a
    robust covariance fit (MinCovDet when seed is large enough, else
    empirical).  The resulting centroid, covariance, and inverse are
    returned for downstream Mahalanobis distance computation.

    Args:
        X: (n_genes, n_axes) COA coordinates for all genes.
        gene_ids: Gene IDs corresponding to rows of X.
        n_axes: Number of COA axes (used for covariance fitting).
        bandwidth: KDE bandwidth method (passed to scipy gaussian_kde).
        seed_radius_pctl: Percentile of distances from peak that defines
            the seed neighbourhood.  Lower values give a tighter seed.
        min_rp: Minimum seed genes for MinCovDet; else empirical.

    Returns:
        centroid: (n_axes,) density-peak centroid (fitted on seed genes).
        cov: (n_axes, n_axes) covariance matrix of the seed.
        cov_inv: (n_axes, n_axes) inverse covariance.
        seed_gene_ids: Gene IDs in the seed set.
        diagnostics: Dict with peak location, seed size, etc.
    """
    from scipy.stats import gaussian_kde

    diag: dict = {
        "density_peak_xy": None,
        "n_seed_genes": 0,
        "seed_radius": 0.0,
        "bandwidth": str(bandwidth),
    }

    # --- KDE on first 2 axes to find the peak ---
    x2d = X[:, :2].T  # (2, n_genes)
    try:
        kde = gaussian_kde(x2d, bw_method=bandwidth)
    except Exception as e:
        logger.warning("Density KDE failed: %s", e)
        # Fallback: use the coordinate-wise median as the peak
        peak_2d = np.median(X[:, :2], axis=0)
        logger.info("Falling back to median as density peak: %s", peak_2d)
        kde = None

    if kde is not None:
        # Evaluate KDE on a grid to find the global maximum
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        pad_x = (x_max - x_min) * 0.05
        pad_y = (y_max - y_min) * 0.05
        grid_n = 200
        gx = np.linspace(x_min - pad_x, x_max + pad_x, grid_n)
        gy = np.linspace(y_min - pad_y, y_max + pad_y, grid_n)
        gxx, gyy = np.meshgrid(gx, gy)
        grid_pts = np.vstack([gxx.ravel(), gyy.ravel()])
        density = kde(grid_pts)
        peak_idx = np.argmax(density)
        peak_2d = grid_pts[:, peak_idx]
    else:
        peak_2d = np.median(X[:, :2], axis=0)

    diag["density_peak_xy"] = [round(float(peak_2d[0]), 6),
                                round(float(peak_2d[1]), 6)]

    logger.info(
        "Density peak located at (%.4f, %.4f) in COA Axis1-Axis2 space",
        peak_2d[0], peak_2d[1],
    )

    # --- Build full-dimensional peak vector ---
    # For axes beyond 2, use the coordinate-wise median of all genes
    if n_axes > 2:
        peak_full = np.concatenate([peak_2d, np.median(X[:, 2:n_axes], axis=0)])
    else:
        peak_full = peak_2d.copy()

    # --- Select seed genes within the radius percentile ---
    dists_from_peak = np.linalg.norm(X[:, :n_axes] - peak_full, axis=1)
    radius = float(np.percentile(dists_from_peak, seed_radius_pctl))
    seed_mask = dists_from_peak <= radius

    diag["seed_radius"] = round(radius, 6)
    diag["n_seed_genes"] = int(seed_mask.sum())

    logger.info(
        "Density seed: %d genes within %.4f of peak (%dth percentile radius)",
        seed_mask.sum(), radius, seed_radius_pctl,
    )

    seed_gene_ids = [gid for gid, s in zip(gene_ids, seed_mask) if s]
    X_seed = X[seed_mask][:, :n_axes]

    # --- Fit covariance on seed genes ---
    def _safe_inv(mat):
        try:
            return np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(mat)

    def _empirical_cov(Xm):
        c = np.cov(Xm.T) if Xm.shape[0] > 1 else np.eye(Xm.shape[1])
        if c.ndim == 0:
            c = np.array([[float(c)]])
        elif c.ndim == 1:
            c = c.reshape(1, 1)
        return c

    n_seed = len(X_seed)
    if n_seed < min_rp:
        logger.warning(
            "Density seed too small (%d < %d) for MinCovDet; using empirical",
            n_seed, min_rp,
        )
        centroid = X_seed.mean(axis=0)
        cov = _empirical_cov(X_seed)
    else:
        try:
            support_frac = max(0.5, min(0.9, (n_seed - 2) / n_seed))
            mcd = MinCovDet(random_state=42, support_fraction=support_frac).fit(X_seed)
            centroid = mcd.location_
            cov = mcd.covariance_
        except Exception as e:
            logger.warning("MinCovDet on density seed failed (%s); empirical fallback", e)
            centroid = X_seed.mean(axis=0)
            cov = _empirical_cov(X_seed)

    cov_inv = _safe_inv(cov)

    logger.info(
        "Density-anchor covariance fitted on %d seed genes (centroid shift from "
        "raw peak: %.4f)",
        n_seed,
        float(np.linalg.norm(centroid[:2] - peak_2d)),
    )

    return centroid, cov, cov_inv, seed_gene_ids, diag


def _classify_translational_selection(
    rp_vs_density_cosine: float | None,
    dual_anchor_categories: Counter | None,
    rp_centroid: np.ndarray | None,
    density_centroid: np.ndarray | None,
    gene_spread_median: float | None,
) -> dict:
    """Classify the strength of translational selection signal.

    Uses three independent lines of evidence from the dual-anchor comparison:

    1. **RSCU cosine similarity** between the RP-cluster and density-cluster.
       High similarity means RP codon preferences are barely distinguishable
       from the genome average.

    2. **rp_only fraction** — the proportion of RP-cluster genes that are NOT
       in the density cluster.  These genes have been pulled away from the
       genome average by selection.  A near-zero fraction means the RP
       cluster sits on top of the genome peak.

    3. **Centroid separation** — Euclidean distance between the RP and density
       centroids, normalised by the median gene-to-genome-centre spread.  A
       small ratio means the anchors overlap.

    Classification:
        "strong"   — at least 2 of the 3 strong-signal indicators are met
        "weak"     — at least 2 of the 3 weak-signal indicators are met
        "moderate" — everything else

    Returns:
        Dict with 'classification', 'evidence' (per-criterion results),
        and 'caveat' (human-readable warning for weak/moderate).
    """
    evidence: dict[str, dict] = {}
    strong_votes = 0
    weak_votes = 0

    # --- Criterion 1: RSCU cosine ---
    if rp_vs_density_cosine is not None and not np.isnan(rp_vs_density_cosine):
        if rp_vs_density_cosine < _TSS_COSINE_STRONG:
            strong_votes += 1
            verdict = "strong"
        elif rp_vs_density_cosine > _TSS_COSINE_WEAK:
            weak_votes += 1
            verdict = "weak"
        else:
            verdict = "moderate"
        evidence["rscu_cosine"] = {
            "value": round(rp_vs_density_cosine, 4),
            "strong_threshold": f"< {_TSS_COSINE_STRONG}",
            "weak_threshold": f"> {_TSS_COSINE_WEAK}",
            "verdict": verdict,
        }

    # --- Criterion 2: rp_only fraction ---
    if dual_anchor_categories is not None:
        n_rp_only = dual_anchor_categories.get("rp_only", 0)
        n_both = dual_anchor_categories.get("both", 0)
        denom = n_rp_only + n_both
        if denom > 0:
            frac = n_rp_only / denom
            if frac > _TSS_RPONLY_FRAC_STRONG:
                strong_votes += 1
                verdict = "strong"
            elif frac < _TSS_RPONLY_FRAC_WEAK:
                weak_votes += 1
                verdict = "weak"
            else:
                verdict = "moderate"
            evidence["rp_only_fraction"] = {
                "value": round(frac, 4),
                "n_rp_only": n_rp_only,
                "n_both": n_both,
                "strong_threshold": f"> {_TSS_RPONLY_FRAC_STRONG}",
                "weak_threshold": f"< {_TSS_RPONLY_FRAC_WEAK}",
                "verdict": verdict,
            }

    # --- Criterion 3: Centroid separation ---
    if (rp_centroid is not None and density_centroid is not None
            and gene_spread_median is not None and gene_spread_median > 0):
        min_len = min(len(rp_centroid), len(density_centroid))
        raw_dist = float(np.linalg.norm(
            rp_centroid[:min_len] - density_centroid[:min_len]
        ))
        normed = raw_dist / gene_spread_median
        # Strong: centroids clearly separated (normed > 0.5)
        # Weak: centroids nearly overlapping (normed < 0.15)
        if normed > 0.5:
            strong_votes += 1
            verdict = "strong"
        elif normed < 0.15:
            weak_votes += 1
            verdict = "weak"
        else:
            verdict = "moderate"
        evidence["centroid_separation"] = {
            "raw_distance": round(raw_dist, 4),
            "normalised": round(normed, 4),
            "strong_threshold": "> 0.5",
            "weak_threshold": "< 0.15",
            "verdict": verdict,
        }

    # --- Overall classification (majority vote) ---
    if strong_votes >= 2:
        classification = "strong"
    elif weak_votes >= 2:
        classification = "weak"
    else:
        classification = "moderate"

    # Build caveat string
    caveats = {
        "strong": None,
        "moderate": (
            "Translational selection signal is moderate. RP-based expression "
            "scores (MELP, CAI, Fop) should be interpreted with caution — the "
            "RP codon preferences are only partially distinguishable from the "
            "genome-wide average."
        ),
        "weak": (
            "Translational selection signal is weak. RP codon preferences are "
            "nearly indistinguishable from the genome-wide average, so "
            "RP-based expression scores (MELP, CAI, Fop) are unreliable for "
            "this genome. Consider using alternative expression proxies or "
            "external transcriptomic data."
        ),
    }

    result = {
        "classification": classification,
        "evidence": evidence,
        "strong_votes": strong_votes,
        "weak_votes": weak_votes,
        "caveat": caveats[classification],
    }

    logger.info(
        "Translational selection strength: %s (strong_votes=%d, weak_votes=%d)",
        classification.upper(), strong_votes, weak_votes,
    )
    if result["caveat"]:
        logger.warning("⚠ %s", result["caveat"])

    return result


def _find_adaptive_threshold(
    distances: np.ndarray,
    rp_distances_clean: np.ndarray,
    median_rp_dist: float,
    default_multiplier: float = _DISTANCE_MULTIPLIER,
    min_mult: float = _ADAPTIVE_MULT_MIN,
    max_mult: float = _ADAPTIVE_MULT_MAX,
    n_kde_points: int = _ADAPTIVE_KDE_N_POINTS,
    min_genes: int = _ADAPTIVE_MIN_GENES,
) -> tuple[float, dict]:
    """Find a data-driven Mahalanobis distance threshold.

    Attempts two methods in order:

    1. **KDE valley detection** — fits a 1-D Gaussian KDE to the full distance
       distribution and searches for the first local minimum after the RP peak.
       The RP peak is defined as the KDE mode in the range [0, 2 × median_rp].
       The valley must lie between the RP peak and the next local maximum
       (the genome-bulk peak) to be accepted.

    2. **Otsu's method** — if no valley is found, binarises the distance
       histogram to maximise between-class variance.  This is robust to
       unimodal distributions but produces a less precise boundary.

    The result is clamped to [min_mult × median_rp, max_mult × median_rp]
    so it can't collapse to near-zero or explode on pathological inputs.

    If the genome has fewer than *min_genes* genes, or the median RP distance
    is zero, the default multiplier is returned.

    Args:
        distances: (n_genes,) Mahalanobis distances for all genes.
        rp_distances_clean: Distances for non-outlier RP genes.
        median_rp_dist: Median of rp_distances_clean.
        default_multiplier: Fallback multiplier if adaptive methods fail.
        min_mult / max_mult: Bounds on the effective multiplier.
        n_kde_points: Number of evaluation points for the 1-D KDE.
        min_genes: Minimum genes required to attempt adaptive threshold.

    Returns:
        threshold: The chosen distance threshold.
        diagnostics: Dict with method used, valley/Otsu location, effective
            multiplier, and any intermediate values.
    """
    from scipy.stats import gaussian_kde as gkde

    diag: dict = {
        "method": "default",
        "threshold": default_multiplier * median_rp_dist,
        "effective_multiplier": default_multiplier,
        "kde_valley": None,
        "kde_rp_peak": None,
        "kde_bulk_peak": None,
        "otsu_threshold": None,
    }

    floor = min_mult * median_rp_dist
    ceiling = max_mult * median_rp_dist

    if len(distances) < min_genes or median_rp_dist <= 0:
        logger.info(
            "Adaptive threshold: too few genes (%d) or zero median RP dist; "
            "using default %.1f×",
            len(distances), default_multiplier,
        )
        diag["threshold"] = np.clip(diag["threshold"], floor, ceiling)
        diag["effective_multiplier"] = diag["threshold"] / median_rp_dist
        return diag["threshold"], diag

    # Evaluation range: 0 to 99th percentile (ignore extreme outliers)
    d_max = float(np.percentile(distances, 99))
    d_eval = np.linspace(0, d_max, n_kde_points)

    # ── Method 1: KDE valley detection ───────────────────────────────
    try:
        kde = gkde(distances, bw_method="scott")
        density = kde(d_eval)

        # Find local extrema
        # A local min at index i: density[i] < density[i-1] and density[i] < density[i+1]
        local_mins = []
        local_maxs = []
        for i in range(1, len(density) - 1):
            if density[i] < density[i - 1] and density[i] < density[i + 1]:
                local_mins.append((d_eval[i], density[i]))
            if density[i] > density[i - 1] and density[i] > density[i + 1]:
                local_maxs.append((d_eval[i], density[i]))

        # RP peak: the highest local max in [0, 2 × median_rp_dist]
        rp_peak_limit = 2.0 * median_rp_dist
        rp_peaks = [(d, v) for d, v in local_maxs if d <= rp_peak_limit]
        if not rp_peaks:
            # Fallback: global max of density in [0, rp_peak_limit]
            mask = d_eval <= rp_peak_limit
            if mask.any():
                idx = np.argmax(density[mask])
                rp_peaks = [(d_eval[mask][idx], density[mask][idx])]

        if rp_peaks:
            rp_peak_d = max(rp_peaks, key=lambda x: x[1])[0]
            diag["kde_rp_peak"] = round(float(rp_peak_d), 4)

            # Find the first valley AFTER the RP peak
            valid_valleys = [
                (d, v) for d, v in local_mins
                if d > rp_peak_d and d > median_rp_dist
            ]

            if valid_valleys:
                valley_d = valid_valleys[0][0]

                # Sanity: there should be a bulk peak after the valley
                bulk_peaks = [
                    (d, v) for d, v in local_maxs if d > valley_d
                ]
                if bulk_peaks:
                    diag["kde_bulk_peak"] = round(float(bulk_peaks[0][0]), 4)

                valley_clamped = float(np.clip(valley_d, floor, ceiling))
                diag["kde_valley"] = round(float(valley_d), 4)
                diag["method"] = "kde_valley"
                diag["threshold"] = valley_clamped
                diag["effective_multiplier"] = round(
                    valley_clamped / median_rp_dist, 4
                )

                logger.info(
                    "Adaptive threshold (KDE valley): raw=%.3f, clamped=%.3f "
                    "(%.2f× median RP dist). RP peak=%.3f, valley=%.3f",
                    valley_d, valley_clamped,
                    valley_clamped / median_rp_dist,
                    rp_peak_d, valley_d,
                )
                return valley_clamped, diag

        logger.debug("KDE valley detection: no valid valley found after RP peak")

    except Exception as e:
        logger.debug("KDE valley detection failed: %s", e)

    # ── Method 2: Otsu's method ──────────────────────────────────────
    try:
        # Histogram the distances
        n_bins = 256
        hist_range = (0, d_max)
        counts, bin_edges = np.histogram(distances, bins=n_bins, range=hist_range)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        total = counts.sum()
        if total == 0:
            raise ValueError("Empty histogram")

        # Otsu: maximise between-class variance
        best_t = default_multiplier * median_rp_dist
        best_var = -1.0

        cum_sum = 0.0
        cum_count = 0
        total_sum = float((counts * bin_centres).sum())

        for i in range(n_bins):
            cum_count += counts[i]
            if cum_count == 0:
                continue
            bg_count = total - cum_count
            if bg_count == 0:
                break

            cum_sum += counts[i] * bin_centres[i]
            mean_bg = (total_sum - cum_sum) / bg_count
            mean_fg = cum_sum / cum_count

            var_between = cum_count * bg_count * (mean_fg - mean_bg) ** 2
            if var_between > best_var:
                best_var = var_between
                best_t = bin_centres[i]

        # Only accept Otsu threshold if it's above the median RP distance
        # (otherwise it's splitting within the RP peak, which is wrong)
        if best_t > median_rp_dist:
            otsu_clamped = float(np.clip(best_t, floor, ceiling))
            diag["otsu_threshold"] = round(float(best_t), 4)
            diag["method"] = "otsu"
            diag["threshold"] = otsu_clamped
            diag["effective_multiplier"] = round(
                otsu_clamped / median_rp_dist, 4
            )

            logger.info(
                "Adaptive threshold (Otsu): raw=%.3f, clamped=%.3f "
                "(%.2f× median RP dist)",
                best_t, otsu_clamped, otsu_clamped / median_rp_dist,
            )
            return otsu_clamped, diag

        logger.debug(
            "Otsu threshold (%.3f) is below median RP distance (%.3f); "
            "rejecting", best_t, median_rp_dist,
        )

    except Exception as e:
        logger.debug("Otsu threshold failed: %s", e)

    # ── Fallback: default multiplier ─────────────────────────────────
    fallback = float(np.clip(default_multiplier * median_rp_dist, floor, ceiling))
    diag["threshold"] = fallback
    diag["effective_multiplier"] = round(fallback / median_rp_dist, 4)

    logger.info(
        "Adaptive threshold: no valley or Otsu found; using default "
        "%.1f× = %.3f", default_multiplier, fallback,
    )
    return fallback, diag


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


def _rscu_to_adaptation_weights(cluster_rscu: pd.Series) -> dict[str, float]:
    """Convert core-cluster RSCU values to per-codon adaptation weights.

    For each synonymous family, w_codon = RSCU_codon / max(RSCU) within
    the family. This follows Sharp & Li (1987): the weight for the most
    preferred codon is 1.0, and all others are scaled relative to it.

    Codons belonging to amino acids not represented in the cluster RSCU
    (or with max RSCU = 0) are omitted, so they won't contribute to CAI.

    Returns:
        Dict mapping RNA codon (e.g. 'GCU') to its adaptation weight in (0, 1].
    """
    # Build a mapping from RSCU column name → codon
    weights: dict[str, float] = {}
    for family_label, codons in AA_CODON_GROUPS_RSCU.items():
        # Find the RSCU column names for this family
        family_cols = []
        for col in RSCU_COLUMN_NAMES:
            codon = RSCU_COL_TO_CODON[col]
            if codon in codons:
                family_cols.append((col, codon))

        # Get RSCU values present in the cluster reference
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
            w = rscu_val / max_rscu
            # Floor at a small epsilon to avoid log(0) in geometric mean
            weights[codon] = max(w, 1e-6)

    return weights


def _compute_core_cai(
    ffn_path: Path,
    adaptation_weights: dict[str, float],
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, float]:
    """Compute CAI of each gene relative to the core-cluster codon preferences.

    CAI (Sharp & Li 1987) is the geometric mean of the per-codon adaptation
    weights across all sense codons in a gene.  Codons not present in the
    weight table (stops, Met, Trp, or missing families) are skipped.

    Args:
        ffn_path: Nucleotide CDS FASTA.
        adaptation_weights: Codon → weight from _rscu_to_adaptation_weights().
        min_length: Minimum sequence length (nt).

    Returns:
        Dict mapping gene ID → core-relative CAI (0–1 scale).
    """
    from codonpipe.modules.rscu import dna_to_rna

    gene_cai: dict[str, float] = {}

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue

        rna = dna_to_rna(seq)
        log_sum = 0.0
        n = 0
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i : i + 3]
            if codon in adaptation_weights:
                log_sum += np.log(adaptation_weights[codon])
                n += 1

        if n > 0:
            gene_cai[rec.id] = float(np.exp(log_sum / n))
        # else: gene has no scoreable codons, omit from dict

    return gene_cai


def _identify_rare_codons(
    rscu_series: pd.Series,
    threshold: float = 0.1,
) -> set[str]:
    """Return the set of RNA codons whose RSCU falls below *threshold*.

    These are codons that the reference gene pool (core cluster or whole
    genome) essentially never uses.  A codon with RSCU < 0.1 in a family
    of 4 synonymous codons means it accounts for <2.5% of usage for that
    amino acid.

    Args:
        rscu_series: RSCU values indexed by column names like 'Ala-GCU'.
        threshold: RSCU below this value flags the codon as rare.

    Returns:
        Set of RNA codon triplets (e.g. {'CUA', 'AGA', ...}).
    """
    rare: set[str] = set()
    for col, val in rscu_series.items():
        if pd.isna(val):
            continue
        if val < threshold:
            codon = RSCU_COL_TO_CODON.get(col)
            if codon:
                rare.add(codon)
    return rare


def _count_rare_codons_per_gene(
    ffn_path: Path,
    rare_codons: set[str],
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, int]:
    """Count how many sense-codon positions in each gene use a rare codon.

    Args:
        ffn_path: Nucleotide CDS FASTA.
        rare_codons: Set of RNA triplets considered rare.
        min_length: Minimum sequence length (nt).

    Returns:
        Dict mapping gene ID → count of rare-codon positions.
    """
    from codonpipe.modules.rscu import dna_to_rna

    gene_counts: dict[str, int] = {}
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        rna = dna_to_rna(seq)
        n_rare = 0
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i : i + 3]
            if codon in rare_codons:
                n_rare += 1
        gene_counts[rec.id] = n_rare
    return gene_counts


def _compute_genome_wide_rscu(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> pd.Series:
    """Pool codon counts across all genes and compute genome-wide RSCU.

    Args:
        ffn_path: Nucleotide CDS FASTA.
        min_length: Minimum sequence length (nt).

    Returns:
        Series indexed by RSCU column names with genome-wide RSCU values.
    """
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
    except Exception as e:
        logger.debug("Could not draw threshold ellipse: %s", e)

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
    """COA scatter colored by dual-anchor category with both threshold ellipses.

    Four categories are rendered in distinct colours:
      both      — teal
      rp_only   — coral red
      dens_only — steel blue
      neither   — light grey
    """
    plt.rcParams.update(STYLE_PARAMS)
    fig, ax = plt.subplots(figsize=(9, 7))

    x = coa_coords["Axis1"].values
    y = coa_coords["Axis2"].values

    cat_colors = {
        "both": "#1b9e77",
        "rp_only": "#d95f02",
        "dens_only": "#7570b3",
        "neither": "#d0d0d0",
    }
    cat_labels = {
        "both": "Both clusters",
        "rp_only": "RP-cluster only",
        "dens_only": "Density-cluster only",
        "neither": "Neither",
    }
    cat_alphas = {
        "both": 0.7, "rp_only": 0.7, "dens_only": 0.7, "neither": 0.15,
    }
    cat_sizes = {
        "both": 14, "rp_only": 14, "dens_only": 14, "neither": 6,
    }

    # Plot by category, "neither" first so it's behind
    for cat in ["neither", "dens_only", "rp_only", "both"]:
        mask = np.array([c == cat for c in categories])
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(
            x[mask], y[mask],
            c=cat_colors[cat], alpha=cat_alphas[cat], s=cat_sizes[cat],
            edgecolors="none", rasterized=True,
            label=f"{cat_labels[cat]} (n={n})",
        )

    # RP markers
    rp_idx = [i for i, g in enumerate(gene_ids) if g in rp_gene_ids]
    if rp_idx:
        ax.scatter(
            x[rp_idx], y[rp_idx],
            facecolors="none", edgecolors="black", s=40, linewidths=0.8,
            label=f"Ribosomal proteins (n={len(rp_idx)})", zorder=5,
        )

    # Draw ellipses for both anchors
    def _draw_ellipse(centroid_2d, cov_2d, thresh, color, label, ls):
        try:
            eigvals, eigvecs = np.linalg.eigh(cov_2d)
            if np.all(eigvals > 0):
                angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
                width = 2 * thresh * np.sqrt(eigvals[1])
                height = 2 * thresh * np.sqrt(eigvals[0])
                ell = mpatches.Ellipse(
                    (centroid_2d[0], centroid_2d[1]), width, height, angle=angle,
                    linewidth=1.8, edgecolor=color, facecolor="none",
                    linestyle=ls, alpha=0.8, label=label,
                )
                ax.add_patch(ell)
        except Exception as e:
            logger.debug("Could not draw %s ellipse: %s", label, e)

    _draw_ellipse(
        rp_centroid[:2], rp_cov[:2, :2], rp_threshold,
        "#d95f02", f"RP threshold (d={rp_threshold:.1f})", "--",
    )
    _draw_ellipse(
        density_centroid[:2], density_cov[:2, :2], density_threshold,
        "#7570b3", f"Density threshold (d={density_threshold:.1f})", ":",
    )

    # Centroids
    ax.scatter(
        rp_centroid[0], rp_centroid[1], marker="*", s=200,
        color="#d95f02", edgecolors="black", linewidths=0.5,
        zorder=7, label="RP centroid",
    )
    ax.scatter(
        density_centroid[0], density_centroid[1], marker="D", s=80,
        color="#7570b3", edgecolors="black", linewidths=0.5,
        zorder=7, label="Density centroid",
    )

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
    """RP-anchored Mahalanobis distance clustering for translational optimization.

    Replaces GMM-based clustering. Uses the known RP genes as an anchor to
    define a Mahalanobis distance threshold in COA space, then classifies
    all genes within that threshold as translationally optimized.

    Steps:
        1. Compute COA on per-gene RSCU (reuses advanced_analyses.compute_coa_on_rscu)
        2. Select top COA axes by cumulative inertia
        3. Fit robust covariance on RP genes (MinCovDet + two-pass outlier removal)
        4. Compute Mahalanobis distance from cleaned RP centroid to all genes
        5. Set threshold at *distance_multiplier* × median RP Mahalanobis distance
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
        distance_multiplier: Threshold = multiplier × median RP Mahalanobis
            distance.  Default 2.0.  Lower values (e.g. 1.5) produce a
            tighter cluster retaining only the most strongly optimised genes;
            higher values (e.g. 3.0) are more permissive.

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
    rp_ids_list = [gene_ids[i] for i in rp_indices]
    logger.info(
        "Found %d/%d RP genes in COA space for %s",
        n_rp_found, len(rp_gene_ids), sample_id,
    )

    # ── Step 3b: RP sub-cluster detection ────────────────────────────
    # When RPs are split into distinct sub-populations in COA space
    # (e.g. due to HGT, ribosome specialisation, or compositional
    # heterogeneity), the centroid of *all* RPs falls in empty space
    # between the groups, inflating the covariance and producing a
    # meaningless cluster.  Detect this and select the densest,
    # most tightly-clustered RP sub-group as the anchor.
    X_rp, rp_ids_list, subcluster_diag = _select_rp_subcluster(
        X_rp, rp_ids_list,
    )
    results["rp_subcluster_diagnostics"] = subcluster_diag

    # Update RP indices and gene_ids to match the (possibly reduced) set
    rp_gene_ids_active = set(rp_ids_list)
    rp_indices = np.array([i for i, g in enumerate(gene_ids) if g in rp_gene_ids_active])

    if len(rp_indices) < 3:
        logger.warning(
            "After sub-cluster selection, only %d RP genes remain for %s; "
            "need >= 3. Skipping.", len(rp_indices), sample_id,
        )
        return results

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

    # ── Step 5b: Adaptive threshold via 2-D projection ───────────────
    # The bimodal gap between RP-like genes and the genome bulk is most
    # visible in the first 2 COA axes (which carry the most inertia).
    # Higher dimensions add chi-squared noise that washes out the valley.
    # Strategy: find the valley in 2-D Mahalanobis space, identify which
    # genes fall below it, then set the full-dimensional threshold to the
    # max full-dimensional distance among those genes.
    if n_axes >= 2:
        centroid_2d = centroid[:2]
        cov_2d = cov[:2, :2]
        try:
            cov_2d_inv = np.linalg.inv(cov_2d)
        except np.linalg.LinAlgError:
            cov_2d_inv = np.linalg.pinv(cov_2d)

        distances_2d = _compute_mahalanobis_distances(
            X[:, :2], centroid_2d, cov_2d_inv,
        )
        rp_dists_2d_all = distances_2d[rp_indices]
        rp_dists_2d_clean = rp_dists_2d_all[~rp_outlier_mask]
        if len(rp_dists_2d_clean) == 0:
            rp_dists_2d_clean = rp_dists_2d_all
        median_rp_2d = float(np.median(rp_dists_2d_clean))

        threshold_2d, threshold_diag = _find_adaptive_threshold(
            distances_2d, rp_dists_2d_clean, median_rp_2d,
            default_multiplier=distance_multiplier,
        )
        threshold_diag["projection"] = "2d"

        # Translate 2-D threshold to full-dimensional space:
        # find genes inside the 2-D boundary, then set the full-dim
        # threshold to the maximum full-dim distance among them (plus a
        # small margin to avoid clipping boundary genes).
        genes_inside_2d = distances_2d <= threshold_2d
        if genes_inside_2d.any():
            full_dists_inside = distances[genes_inside_2d]
            # Use 95th percentile of the full-dim distances of 2-D-selected
            # genes as the threshold (avoids a single outlier inflating it)
            threshold = float(np.percentile(full_dists_inside, 95))
            threshold_diag["threshold_2d"] = round(threshold_2d, 4)
            threshold_diag["n_genes_inside_2d"] = int(genes_inside_2d.sum())
            threshold_diag["threshold"] = round(threshold, 4)
            threshold_diag["effective_multiplier"] = round(
                threshold / median_rp_dist, 4
            )
            logger.info(
                "2-D adaptive threshold: valley at %.2f in 2-D → %d genes → "
                "full-dim threshold %.2f (%.2f× median RP dist)",
                threshold_2d, genes_inside_2d.sum(),
                threshold, threshold / median_rp_dist,
            )
        else:
            # Fallback if no genes inside 2-D boundary (shouldn't happen)
            threshold = distance_multiplier * median_rp_dist
            threshold_diag["threshold"] = round(threshold, 4)
            logger.warning(
                "No genes inside 2-D adaptive boundary; using default %.1f×",
                distance_multiplier,
            )
    else:
        # Only 2 axes — no projection needed, run directly on full distances
        threshold, threshold_diag = _find_adaptive_threshold(
            distances, rp_dists_clean, median_rp_dist,
            default_multiplier=distance_multiplier,
        )
        threshold_diag["projection"] = "full"

    results["threshold_diagnostics"] = threshold_diag

    logger.info(
        "Mahalanobis threshold: %.2f (method=%s, effective multiplier=%.2f×, "
        "median RP dist=%.2f)",
        threshold, threshold_diag["method"],
        threshold_diag["effective_multiplier"], median_rp_dist,
    )

    # Store RP-anchor geometry for downstream plotting
    results["rp_centroid"] = centroid
    results["rp_cov"] = cov
    results["rp_threshold"] = threshold

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

    # ── Step 7b: Core-relative CAI ───────────────────────────────────
    # Derive per-codon adaptation weights from the core cluster RSCU,
    # then score every gene's CAI against those weights.
    gene_core_cai: dict[str, float] = {}
    if not cluster_rscu.empty and ffn_path and ffn_path.exists():
        adapt_weights = _rscu_to_adaptation_weights(cluster_rscu)
        if adapt_weights:
            gene_core_cai = _compute_core_cai(ffn_path, adapt_weights)
            logger.info(
                "Core-relative CAI computed for %d genes (mean=%.4f, median=%.4f)",
                len(gene_core_cai),
                float(np.mean(list(gene_core_cai.values()))) if gene_core_cai else 0.0,
                float(np.median(list(gene_core_cai.values()))) if gene_core_cai else 0.0,
            )
        else:
            logger.warning("Could not derive adaptation weights from cluster RSCU; skipping core CAI")
    results["gene_core_cai"] = gene_core_cai

    # ── Step 7c: Rare codon counts ───────────────────────────────────
    # Identify codons with RSCU ≈ 0 in (a) the core cluster and (b) the
    # whole genome, then count per gene how many codon positions use them.
    _rare_codon_threshold = 0.1
    core_rare_per_gene: dict[str, int] = {}
    genome_rare_per_gene: dict[str, int] = {}

    if not cluster_rscu.empty and ffn_path and ffn_path.exists():
        core_rare_codons = _identify_rare_codons(cluster_rscu, threshold=_rare_codon_threshold)
        if core_rare_codons:
            core_rare_per_gene = _count_rare_codons_per_gene(ffn_path, core_rare_codons)
            logger.info(
                "Core-rare codons: %d codons with RSCU < %.2f in core cluster "
                "(e.g. %s)",
                len(core_rare_codons), _rare_codon_threshold,
                ", ".join(sorted(core_rare_codons)[:5]),
            )

        genome_rscu = _compute_genome_wide_rscu(ffn_path)
        if not genome_rscu.empty:
            genome_rare_codons = _identify_rare_codons(genome_rscu, threshold=_rare_codon_threshold)
            if genome_rare_codons:
                genome_rare_per_gene = _count_rare_codons_per_gene(ffn_path, genome_rare_codons)
                logger.info(
                    "Genome-rare codons: %d codons with RSCU < %.2f genome-wide "
                    "(e.g. %s)",
                    len(genome_rare_codons), _rare_codon_threshold,
                    ", ".join(sorted(genome_rare_codons)[:5]),
                )

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
        "core_CAI": [gene_core_cai.get(gid, np.nan) for gid in gene_ids],
        "n_core_rare_codons": [core_rare_per_gene.get(gid, 0) for gid in gene_ids],
        "n_genome_rare_codons": [genome_rare_per_gene.get(gid, 0) for gid in gene_ids],
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
        "threshold_method": threshold_diag["method"],
        "threshold_projection": threshold_diag.get("projection", "full"),
        "threshold_effective_multiplier": threshold_diag["effective_multiplier"],
        "threshold_kde_valley": threshold_diag.get("kde_valley"),
        "threshold_2d": threshold_diag.get("threshold_2d"),
        "threshold_n_genes_inside_2d": threshold_diag.get("n_genes_inside_2d"),
        "threshold_otsu": threshold_diag.get("otsu_threshold"),
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
    results["mahal_coa_inertia"] = coa_inertia
    results["mahal_gene_distances"] = pd.Series(distances, index=gene_ids, name="mahalanobis_distance")

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

    # ── Step 11: Density-anchor mode ────────────────────────────────────
    # A second Mahalanobis clustering anchored to the genome-wide density
    # peak in COA space.  This captures the "compositional baseline" — the
    # codon preferences of the average gene — in contrast to the RP-anchor
    # which captures translational selection.
    #
    # Comparing the two clusters yields four gene categories:
    #   both     — genes in both clusters (core genome, near average AND near RP)
    #   rp_only  — in RP-cluster but not density-cluster (translationally
    #              selected away from the genome average)
    #   dens_only — in density-cluster but not RP-cluster (typical genome
    #               composition, not translationally optimised)
    #   neither  — outside both clusters (HGT/outlier candidates)

    try:
        density_centroid, density_cov, density_cov_inv, density_seed_ids, density_diag = (
            _find_density_peak_anchor(
                X, gene_ids, n_axes,
                bandwidth=_DENSITY_KDE_BANDWIDTH,
                seed_radius_pctl=_DENSITY_SEED_RADIUS_PCTL,
            )
        )
        results["density_anchor_diagnostics"] = density_diag

        # Compute density-based Mahalanobis distances
        density_distances = _compute_mahalanobis_distances(
            X[:, :n_axes], density_centroid, density_cov_inv,
        )

        # Threshold: multiplier × median distance of seed genes
        seed_set = set(density_seed_ids)
        seed_dists = np.array([
            d for gid, d in zip(gene_ids, density_distances)
            if gid in seed_set
        ])
        if len(seed_dists) == 0:
            seed_dists = density_distances
        density_median = float(np.median(seed_dists))
        density_threshold = _DENSITY_MULTIPLIER * density_median

        logger.info(
            "Density-anchor threshold: %.2f × %.2f = %.2f",
            _DENSITY_MULTIPLIER, density_median, density_threshold,
        )

        density_optimized_mask = density_distances <= density_threshold
        density_cluster_gene_ids = {
            gid for gid, opt in zip(gene_ids, density_optimized_mask) if opt
        }

        # Soft membership for density cluster
        density_probabilities = _distance_to_membership(density_distances, density_threshold)

        # Store density-anchor results
        results["density_cluster_gene_ids"] = density_cluster_gene_ids
        results["density_distances"] = density_distances
        results["density_threshold"] = density_threshold
        results["density_centroid"] = density_centroid
        results["density_cov"] = density_cov
        results["density_probabilities"] = density_probabilities

        # ── Step 12: Dual-anchor comparison ──────────────────────────────
        rp_set = cluster_gene_ids        # from Step 6
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

        # Category counts
        cat_counts = Counter(categories)
        logger.info(
            "Dual-anchor categories for %s: both=%d, rp_only=%d, "
            "dens_only=%d, neither=%d",
            sample_id,
            cat_counts.get("both", 0),
            cat_counts.get("rp_only", 0),
            cat_counts.get("dens_only", 0),
            cat_counts.get("neither", 0),
        )

        # Save dual-anchor table
        dual_path = mahal_dir / f"{sample_id}_dual_anchor_comparison.tsv"
        dual_df.to_csv(dual_path, sep="\t", index=False)
        results["dual_anchor_path"] = dual_path
        results["dual_anchor_df"] = dual_df
        results["dual_anchor_categories"] = cat_counts

        # Summary stats for the density anchor
        density_summary = {
            "density_peak_axis1": density_diag["density_peak_xy"][0] if density_diag["density_peak_xy"] else None,
            "density_peak_axis2": density_diag["density_peak_xy"][1] if density_diag["density_peak_xy"] else None,
            "density_seed_genes": density_diag["n_seed_genes"],
            "density_seed_radius": density_diag["seed_radius"],
            "density_threshold": round(density_threshold, 4),
            "density_median_seed_dist": round(density_median, 4),
            "density_cluster_size": len(density_cluster_gene_ids),
            "n_both": cat_counts.get("both", 0),
            "n_rp_only": cat_counts.get("rp_only", 0),
            "n_dens_only": cat_counts.get("dens_only", 0),
            "n_neither": cat_counts.get("neither", 0),
            "rp_centroid_to_density_centroid": round(
                float(np.linalg.norm(centroid[:min(len(centroid), len(density_centroid))]
                                     - density_centroid[:min(len(centroid), len(density_centroid))])),
                4,
            ),
        }
        density_summary_df = pd.DataFrame([density_summary])
        density_summary_path = mahal_dir / f"{sample_id}_density_anchor_summary.tsv"
        density_summary_df.to_csv(density_summary_path, sep="\t", index=False)
        results["density_summary_path"] = density_summary_path

        # ── Step 13: Density-anchor RSCU ─────────────────────────────────
        # Compute RSCU for the density cluster using the same distance-
        # weighted pooling approach, but with density distances/threshold.
        if ffn_path and ffn_path.exists() and density_cluster_gene_ids:
            density_gene_weights = {}
            for gid, d in zip(gene_ids, density_distances):
                if gid in density_cluster_gene_ids:
                    w = max(1.0 - d / density_threshold, 0.0) if density_threshold > 0 else 1.0
                    if w > 0:
                        density_gene_weights[gid] = w

            density_cluster_rscu = _compute_cluster_rscu(
                ffn_path, density_cluster_gene_ids, gene_weights=density_gene_weights,
            )
            if not density_cluster_rscu.empty:
                density_rscu_path = mahal_dir / f"{sample_id}_density_cluster_rscu.tsv"
                density_cluster_rscu.to_frame("RSCU").to_csv(density_rscu_path, sep="\t")
                results["density_cluster_rscu"] = density_cluster_rscu
                results["density_cluster_rscu_path"] = density_rscu_path

                # Cosine similarity between RP-cluster and density-cluster RSCU
                shared_cols = [c for c in cluster_rscu.index if c in density_cluster_rscu.index]
                if shared_cols and not cluster_rscu.empty:
                    anchor_cosine = 1.0 - cosine_dist(
                        cluster_rscu[shared_cols].fillna(0).values,
                        density_cluster_rscu[shared_cols].fillna(0).values,
                    )
                    results["rp_vs_density_rscu_cosine"] = float(anchor_cosine)
                    logger.info(
                        "RP-cluster vs density-cluster RSCU cosine similarity: %.4f "
                        "(high = RP preferences match genome average; "
                        "low = strong translational selection signal)",
                        anchor_cosine,
                    )

        # ── Step 14: Classify translational selection strength ────────────
        # Compute median gene spread for centroid distance normalisation
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

        # Append to the Mahalanobis summary TSV
        summary_path = mahal_dir / f"{sample_id}_mahal_summary.tsv"
        if summary_path.exists():
            try:
                s_df = pd.read_csv(summary_path, sep="\t")
                s_df["translational_selection"] = tss["classification"]
                s_df["tss_caveat"] = tss["caveat"] if tss["caveat"] else ""
                s_df.to_csv(summary_path, sep="\t", index=False)
            except Exception as e:
                logger.debug("Could not append TSS to summary: %s", e)

        # Write dedicated TSS diagnostics file
        tss_path = mahal_dir / f"{sample_id}_translational_selection_strength.tsv"
        tss_flat = {
            "sample_id": sample_id,
            "classification": tss["classification"],
            "strong_votes": tss["strong_votes"],
            "weak_votes": tss["weak_votes"],
            "caveat": tss["caveat"] if tss["caveat"] else "",
        }
        for criterion, info in tss["evidence"].items():
            for k, v in info.items():
                tss_flat[f"{criterion}_{k}"] = v
        pd.DataFrame([tss_flat]).to_csv(tss_path, sep="\t", index=False)
        results["tss_path"] = tss_path

        # ── Step 15: Dual-anchor diagnostic plot ─────────────────────────
        try:
            _plot_dual_anchor_coa(
                coa_coords=coa_coords[coa_coords["gene"].isin(gene_ids)].reset_index(drop=True),
                gene_ids=gene_ids,
                categories=categories,
                rp_centroid=centroid,
                density_centroid=density_centroid,
                rp_cov=cov,
                density_cov=density_cov,
                rp_threshold=threshold,
                density_threshold=density_threshold,
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
            logger.warning("Dual-anchor COA plot failed: %s", e, exc_info=True)

    except Exception as e:
        logger.warning(
            "Density-anchor mode failed for %s: %s", sample_id, e,
            exc_info=True,
        )

    logger.info(
        "RP-anchored clustering complete for %s: "
        "threshold=%.2f, optimized set=%d genes (%d RPs + %d non-RP)",
        sample_id, threshold, n_cluster, n_rp_in_cluster, n_cluster - n_rp_in_cluster,
    )

    return results
