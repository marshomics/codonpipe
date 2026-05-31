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
    _select_rp_dense_core,
    _detect_rp_subclusters,
    _compute_cluster_rscu,
    _chi2_threshold,
    _MAX_COA_AXES,
    _MIN_COA_AXES,
    _MIN_GENES_FOR_CLUSTERING,
    _RP_OUTLIER_ALPHA,
    _MIN_RP_FOR_ROBUST,
    _CLUSTER_CHI2_P,
)

# ── RP sub-cluster stability validation (Hennig 2007) ───────────────────────
# When a genome's ribosomal-protein genes form two or more distinct
# populations in COA space (e.g. a canonical RP cohort plus an HGT- or
# paralogy-derived secondary cohort with non-canonical codon usage), the
# bootstrap consensus must NOT pool them — pooling produces a centroid and
# covariance averaged across biologically different references. We therefore
# (1) detect candidate sub-clusters once on the full RP set via GMM+BIC, then
# (2) validate each candidate by clusterwise bootstrap stability (the Jaccard
# resampling of Hennig 2007, "Cluster-wise assessment of cluster stability",
# the method behind R's fpc::clusterboot). A candidate counts as a genuine
# cluster only if its mean maximum-Jaccard to the best-matching bootstrap
# sub-cluster meets _CLUSTER_STABLE_JACCARD; below _CLUSTER_DISSOLVED_JACCARD
# it is treated as dissolved (noise). The matching step is what makes this
# robust to label switching across resamples.
_CLUSTER_STABLE_JACCARD = 0.75      # Hennig: >=0.75 = valid, stable cluster
_CLUSTER_DISSOLVED_JACCARD = 0.60   # Hennig: <0.60 = dissolved / not a real cluster
_CLUSTER_VALIDATION_BOOTSTRAPS = 100

# ── Gap-based RP-cluster boundary ───────────────────────────────────────────
# A fixed chi-squared quantile (the old _CLUSTER_CHI2_P=0.95 boundary) is too
# tight for a well-separated RP cohort: it cuts through the cohort and excludes
# legitimate members, while the genome's distant RP outliers sit far beyond.
# The RP genes' own Mahalanobis-distance distribution, however, typically shows
# a clear *gap* between the compact translational-optimum cohort and the
# scattered outliers (verified on real data: B. subtilis RP distances run
# ~0.5-3.9 then jump to 5.6+). We therefore cut at the largest gap in the
# cleaned RP distances, with guards:
#   floor   = chi2 p90  -> never cut inside the dense core
#   ceiling = chi2 p999 -> never run out into the genomic bulk ...
#   EXCEPT a single wide, clean gap may push the boundary past the ceiling, up
#   to the gap, because a wide separation is itself strong evidence of a real
#   cohort/outlier boundary (this is what recovers the full B. subtilis cohort).
_GAP_FLOOR_CHI2_P = 0.90
_GAP_CEIL_CHI2_P = 0.999
# A gap qualifies as "wide/clean" (allowed to override the chi-squared ceiling)
# when it is at least this multiple of the median spacing between consecutive
# RP distances below it — i.e. a genuine discontinuity, not ordinary scatter.
_GAP_OVERRIDE_RATIO = 3.0
# Absolute safety cap on the override, as a chi-squared quantile: even a huge
# gap cannot push the boundary past this (prevents a pathological RP set from
# producing a genome-spanning cluster).
_GAP_OVERRIDE_HARD_CEIL_CHI2_P = 0.99999


def _gap_boundary(rp_distances, n_axes):
    """Largest-gap RP-cluster boundary with chi-squared guards.

    Cuts the cleaned RP Mahalanobis distances at their largest interior gap so
    the boundary sits between the compact RP cohort and the distant outliers,
    rather than at an arbitrary fixed quantile. Returns (threshold, info).

    The search for the largest gap is restricted to the band
    [chi2 p90, chi2 p999]; a sufficiently wide, clean gap (>= _GAP_OVERRIDE_RATIO
    times the local median spacing) is allowed to extend the boundary above the
    p999 ceiling, up to the gap itself but never past chi2 p99999.
    """
    d = np.sort(np.asarray(rp_distances, float))
    d = d[np.isfinite(d)]
    floor = _chi2_threshold(n_axes, _GAP_FLOOR_CHI2_P)
    ceil = _chi2_threshold(n_axes, _GAP_CEIL_CHI2_P)
    hard_ceil = _chi2_threshold(n_axes, _GAP_OVERRIDE_HARD_CEIL_CHI2_P)
    if d.size < 4:
        return float(min(ceil, max(floor, _chi2_threshold(n_axes, _CLUSTER_CHI2_P)))), {
            "method": "chi2_fallback_small_rp", "n_rp": int(d.size)}

    spacings = np.diff(d)
    med_spacing = float(np.median(spacings)) if spacings.size else 0.0

    # Candidate cut points: lower edge of each gap, considered if it lies in the
    # band OR forms a wide/clean gap (which may sit at/above the ceiling).
    best_cut, best_gap, overrode = None, -1.0, False
    for i in range(d.size - 1):
        lo = float(d[i])
        gap = float(d[i + 1] - lo)
        wide = med_spacing > 0 and gap >= _GAP_OVERRIDE_RATIO * med_spacing
        in_band = floor <= lo <= ceil
        # allow override only for a genuinely wide gap whose lower edge is past
        # the floor (i.e. outside the dense core) and not already deep in noise
        override_ok = wide and lo >= floor and lo <= hard_ceil
        if (in_band or override_ok) and gap > best_gap:
            best_gap, best_cut = gap, lo
            overrode = override_ok and lo > ceil

    if best_cut is None:
        # No qualifying gap: fall back to the chi-squared p99 boundary, bounded.
        thr = float(np.clip(_chi2_threshold(n_axes, 0.99), floor, ceil))
        return thr, {"method": "chi2_p99_no_gap", "n_rp": int(d.size),
                     "floor": round(floor, 3), "ceil": round(ceil, 3)}

    thr = float(min(best_cut, hard_ceil))
    return thr, {
        "method": "largest_gap_override" if overrode else "largest_gap_in_band",
        "n_rp": int(d.size), "gap_width": round(best_gap, 3),
        "median_spacing": round(med_spacing, 4),
        "floor": round(floor, 3), "ceil": round(ceil, 3),
        "overrode_ceiling": bool(overrode),
        "rp_below_threshold": int((d <= thr).sum()),
    }


def _jaccard_index(a: set, b: set) -> float:
    """Jaccard similarity |a∩b| / |a∪b| (0 when both empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# Margin below the weakest core RP gene's cosine that a rescued gene is still
# allowed to sit at, so the rescue threshold is not hostage to a single odd
# core member. 0.0 = exactly the weakest core member; a small margin lets a
# gene that is essentially as RP-like also qualify.
_RESCUE_COSINE_MARGIN = 0.005
# Geometric sanity cap on the rescue: a rescued gene must sit within this
# multiple of the cluster's Mahalanobis boundary. RSCU cosine alone is too
# permissive because all ribosomal proteins share amino-acid composition and
# therefore look somewhat RP-like even when scattered far from the cluster in
# COA space (observed: B. subtilis RP genes at distance 13-18 near the genome
# bulk score high cosine). Requiring BOTH RP-like codon usage AND geometric
# proximity recovers the cohort's borderline tail (distance just past the cut)
# without admitting the far-flung RP genes. The cohort tail and the scattered
# RP genes are well separated in distance (B. subtilis: tail ends ~5.8, next
# RP gene at ~8.5), so 2x the boundary cleanly divides them.
_RESCUE_DISTANCE_MULT = 2.0


def _rescue_rp_by_rscu(core_ids, rp_gene_ids, ffn_path, rp_rscu_df,
                       distances=None, gene_ids=None, boundary=None):
    """Rescue annotated RP genes whose codon usage is as ribosomal-like as the
    weakest RP gene already in the core AND which sit within a sanity distance
    of the cluster.

    The geometric (Mahalanobis) boundary can clip legitimate RP genes when the
    cohort's distances trail off without a clean gap. Membership in the
    translational-optimum core is fundamentally about ribosomal-protein-like
    codon usage, so we compare each gene's RSCU to the RP consensus by cosine
    similarity and admit any out-of-core RP gene whose similarity is at least
    that of the weakest in-core RP gene (minus a small margin) AND whose
    Mahalanobis distance is within ``_RESCUE_DISTANCE_MULT`` x the cluster
    boundary. The cosine test keeps the rescue biologically grounded; the
    distance cap prevents compositionally-RP-but-geometrically-distant genes
    from being pulled in. The rescue is restricted to annotated RP genes, so it
    can never add non-RP bulk genes.

    Returns the set of rescued gene IDs (possibly empty). Needs the per-gene
    nucleotide FASTA (to compute each gene's RSCU) and the RP-only RSCU table
    (for the consensus); the distance cap is applied only when ``distances``,
    ``gene_ids`` and ``boundary`` are supplied. Returns an empty set when the
    FASTA or RP RSCU table is unavailable.
    """
    if not core_ids or not rp_gene_ids:
        return set()
    if ffn_path is None or not Path(ffn_path).exists():
        return set()
    if rp_rscu_df is None or rp_rscu_df.empty:
        return set()

    # Optional geometric sanity cap: max distance a rescued gene may sit at.
    dist_cap = None
    gid_to_dist = None
    if distances is not None and gene_ids is not None and boundary is not None:
        dist_cap = float(boundary) * _RESCUE_DISTANCE_MULT
        gid_to_dist = {g: float(d) for g, d in zip(gene_ids, np.asarray(distances, float))}

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rp_rscu_df.columns]
    if not rscu_cols:
        return set()
    consensus = rp_rscu_df[rscu_cols].mean()
    cvec = np.nan_to_num(consensus.to_numpy(dtype=float), nan=0.0)
    cnorm = np.linalg.norm(cvec)
    if cnorm == 0:
        return set()

    # Per-gene RSCU for the union of the current core and all RP genes (so we
    # can score both the weakest core member and every rescue candidate).
    need_ids = set(core_ids) | set(rp_gene_ids)
    gene_rscu = _compute_per_gene_rscu(ffn_path, need_ids)
    if gene_rscu is None or gene_rscu.empty:
        return set()
    cols_present = [c for c in rscu_cols if c in gene_rscu.columns]
    if not cols_present:
        return set()
    cvec = np.nan_to_num(consensus[cols_present].to_numpy(dtype=float), nan=0.0)
    cnorm = np.linalg.norm(cvec)
    if cnorm == 0:
        return set()

    def _cos(gid):
        if gid not in gene_rscu.index:
            return None
        v = np.nan_to_num(gene_rscu.loc[gid, cols_present].to_numpy(dtype=float), nan=0.0)
        vn = np.linalg.norm(v)
        if vn == 0:
            return None
        return float(v @ cvec / (vn * cnorm))

    # Weakest cosine among RP genes already in the core defines the bar.
    core_rp = [g for g in core_ids if g in rp_gene_ids]
    core_cos = [c for c in (_cos(g) for g in core_rp) if c is not None]
    if not core_cos:
        return set()
    bar = float(min(core_cos)) - _RESCUE_COSINE_MARGIN

    rescued = set()
    for g in rp_gene_ids:
        if g in core_ids:
            continue
        # Geometric sanity cap: skip genes too far from the cluster, even if
        # their codon usage is RP-like.
        if dist_cap is not None and gid_to_dist is not None:
            d = gid_to_dist.get(g)
            if d is None or not np.isfinite(d) or d > dist_cap:
                continue
        c = _cos(g)
        if c is not None and c >= bar:
            rescued.add(g)
    return rescued


def _compute_per_gene_rscu(ffn_path, gene_ids):
    """Per-gene RSCU table (genes × codon columns) for a set of gene IDs.

    Thin wrapper over the project's per-gene RSCU computation so the rescue
    helper stays decoupled from the exact RSCU implementation. Returns a
    DataFrame indexed by gene ID, or None if it cannot be computed.
    """
    try:
        from codonpipe.modules.rscu import compute_rscu_per_gene
    except Exception:
        return None
    try:
        df = compute_rscu_per_gene(Path(ffn_path))
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if "gene" in df.columns:
        df = df.set_index("gene")
    keep = [g for g in gene_ids if g in df.index]
    return df.loc[keep] if keep else None


def _validate_rp_subclusters(
    X_rp: np.ndarray,
    rp_ids_list: list[str],
    n_boot: int = _CLUSTER_VALIDATION_BOOTSTRAPS,
    seed: int = 0,
) -> list[dict]:
    """Validate RP sub-clusters by clusterwise Jaccard bootstrap (Hennig 2007).

    Reference partition is the GMM+BIC split from ``_detect_rp_subclusters``.
    For each of ``n_boot`` resamples of the RP genes (with replacement), the
    RP set is re-clustered and every *reference* sub-cluster is matched to the
    bootstrap sub-cluster with which it shares the largest Jaccard index; that
    maximum Jaccard is recorded. A reference cluster's stability is the mean of
    these recorded maxima. Clusters with mean Jaccard >= _CLUSTER_STABLE_JACCARD
    are genuine; this distinguishes two (or more) real RP clusters from one
    cluster plus outliers, because an outlier pocket does not recur coherently
    across resamples and dissolves (mean Jaccard < _CLUSTER_DISSOLVED_JACCARD).

    Returns the reference sub-cluster list with three keys added to each:
      mean_jaccard, jaccard_sd, stability_label ("stable"/"uncertain"/"dissolved").
    Sorted by descending mean_jaccard, then size.
    """
    ref = _detect_rp_subclusters(X_rp, rp_ids_list)
    if len(ref) <= 1:
        # No split: single population. Still report its self-stability so the
        # downstream record is uniform, but no matching is needed.
        for c in ref:
            c["mean_jaccard"] = 1.0
            c["jaccard_sd"] = 0.0
            c["stability_label"] = "stable"
        return ref

    ref_sets = [set(c["gene_ids"]) for c in ref]
    per_cluster_jacc = [[] for _ in ref]
    id_to_row = {gid: i for i, gid in enumerate(rp_ids_list)}
    n_rp = len(rp_ids_list)
    rng = np.random.RandomState(seed)

    for _ in range(n_boot):
        boot_rows = rng.choice(n_rp, size=n_rp, replace=True)
        uniq = np.unique(boot_rows)
        X_b = X_rp[uniq]
        ids_b = [rp_ids_list[i] for i in uniq]
        boot_sub = _detect_rp_subclusters(X_b, ids_b)
        boot_sets = [set(c["gene_ids"]) for c in boot_sub]
        for ci, rset in enumerate(ref_sets):
            # Best Jaccard between this reference cluster (restricted to genes
            # present in the resample, the standard Hennig restriction) and any
            # bootstrap cluster.
            rset_in = rset & set(ids_b)
            best = max((_jaccard_index(rset_in, bset) for bset in boot_sets),
                       default=0.0)
            per_cluster_jacc[ci].append(best)

    for ci, c in enumerate(ref):
        arr = np.array(per_cluster_jacc[ci]) if per_cluster_jacc[ci] else np.array([0.0])
        mj = float(arr.mean())
        c["mean_jaccard"] = round(mj, 4)
        c["jaccard_sd"] = round(float(arr.std()), 4)
        c["stability_label"] = (
            "stable" if mj >= _CLUSTER_STABLE_JACCARD
            else "dissolved" if mj < _CLUSTER_DISSOLVED_JACCARD
            else "uncertain"
        )

    ref.sort(key=lambda c: (-c["mean_jaccard"], -c["n"]))
    return ref


def _select_rp_anchor(
    validated: list[dict],
    ffn_path: Path | None,
    rp_rscu_df: pd.DataFrame | None,
) -> tuple[dict, str]:
    """Choose which validated RP sub-cluster anchors the optimized core.

    Among sub-clusters that passed stability validation (stability_label !=
    "dissolved"), the anchor is the one whose codon usage most resembles the
    canonical ribosomal-protein RSCU consensus (highest cosine similarity),
    with RP-gene count as the tie-breaker. This mirrors the single-shot
    run_mahal_clustering rule and is the biologically defensible criterion:
    "the real translational-optimum cluster looks like the host's ribosomal
    proteins", not merely "the biggest RP cohort" — a secondary HGT/paralog RP
    cohort can be large but compositionally aberrant. If no RSCU reference is
    available, falls back to the largest stable cluster.

    Returns (anchor_subcluster, selection_reason).
    """
    candidates = [c for c in validated if c.get("stability_label") != "dissolved"]
    if not candidates:
        candidates = list(validated)  # all dissolved: keep the best-Jaccard one
    if len(candidates) == 1:
        return candidates[0], "single_validated_cluster"

    # RSCU cosine similarity to the RP consensus for each candidate.
    have_ref = (
        ffn_path is not None and ffn_path.exists()
        and rp_rscu_df is not None and not rp_rscu_df.empty
    )
    if have_ref:
        rscu_cols_all = [c for c in RSCU_COLUMN_NAMES if c in rp_rscu_df.columns]
        rp_consensus = rp_rscu_df[rscu_cols_all].mean() if rscu_cols_all else None
        for c in candidates:
            sim = float("nan")
            try:
                if rp_consensus is not None and len(c["gene_ids"]) >= 3:
                    sub_rscu = _compute_cluster_rscu(ffn_path, set(c["gene_ids"]))
                    shared = [col for col in rscu_cols_all if col in sub_rscu.index]
                    if shared:
                        sim = 1.0 - cosine_dist(
                            sub_rscu[shared].fillna(0).values,
                            rp_consensus[shared].fillna(0).values,
                        )
            except Exception:
                sim = float("nan")
            c["rp_cosine_sim"] = round(sim, 4) if sim == sim else None
        if any(c.get("rp_cosine_sim") is not None for c in candidates):
            anchor = max(
                candidates,
                key=lambda c: (
                    c["rp_cosine_sim"] if c.get("rp_cosine_sim") is not None else float("-inf"),
                    c["n"],
                ),
            )
            return anchor, "rscu_consensus_similarity"

    # No usable RSCU reference: fall back to the largest stable cluster.
    anchor = max(candidates, key=lambda c: c["n"])
    return anchor, "largest_stable_cluster_fallback"
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

# Weights for the composite cluster-stability score. The score combines
# three complementary signals:
#   - Jaccard similarity across bootstrap resamples (set overlap of the
#     RP cluster identity) — weighted highest because identity is the
#     primary thing we want stable.
#   - Mean per-gene membership frequency in the RP cluster (does each
#     RP gene reliably end up in the cluster?).
#   - Cosine similarity of cluster mean RSCU across bootstraps (does
#     the cluster's codon-usage signature stay constant?).
# The 0.4 / 0.3 / 0.3 weighting is heuristic, not derived from a fit.
# The Jaccard term is given a slight edge because it summarises the
# binary "is this gene in / out" agreement that matters most for
# downstream HGT calls; the other two are diagnostic of softer
# instability modes. Tune these constants if you have a benchmark set
# of stable vs unstable clusters and want to recalibrate.
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
) -> tuple[set[str], float]:
    """One bootstrap replicate using chi-squared threshold.

    Returns (cluster_gene_ids, oob_recovery_rate).

    The bootstrap leaves on average ~37% of RP genes out-of-bag (OOB)
    because the resample is taken with replacement. Those OOB genes
    are an honest hold-out: the centroid that classifies them was fit
    without seeing them. ``oob_recovery_rate`` is the fraction of OOB
    RP genes that the bootstrap centroid still places inside the
    chi-squared cluster boundary. A high recovery rate means the RP
    cluster is *predictively* stable; a low rate means the cluster
    fits its training resample well but doesn't generalise to held-out
    RP genes.

    This complements the existing across-replicate Jaccard similarity,
    which measures internal consistency (do replicates agree on which
    genome-wide genes go where) but does not test whether held-out RP
    members would be recovered.

    The multiplier parameter is accepted for backward compatibility but
    ignored — the threshold is always chi-squared based.
    """
    rng = np.random.RandomState(seed)
    n_rp = len(rp_indices)
    boot_rp_idx = rng.choice(rp_indices, size=n_rp, replace=True)
    X_rp = X[boot_rp_idx]

    # Dense-core selection BEFORE fitting covariance — matches the single-shot
    # RP path (run_mahal_clustering -> _select_rp_dense_core). Without it, a few
    # compositionally drifted RP genes inflate the resampled covariance; in
    # low-dimensional COA spaces (e.g. 2 axes) the resulting chi-squared ellipse
    # is so large it swallows most of the genome, and the membership-frequency
    # consensus inherits that bloat (observed: B. subtilis 61-83% of genes in
    # the "RP" cluster). Fitting from the dense core keeps the boundary on the
    # translationally-optimized RP cloud. Resampling can duplicate rows, so the
    # 2-D KDE is run on the unique resampled points to stay well-conditioned.
    uniq_idx = np.unique(boot_rp_idx)
    X_rp_uniq = X[uniq_idx]
    _ids_uniq = [str(i) for i in uniq_idx]
    try:
        _, core_id_strs, _ = _select_rp_dense_core(X_rp_uniq, _ids_uniq, density_pctl=50)
        core_set = set(core_id_strs)
        X_fit = np.array([X[i] for i in uniq_idx if str(i) in core_set])
        if len(X_fit) < max(3, n_axes + 1):
            X_fit = X_rp  # fall back to full resample if the core is too small
    except Exception:
        X_fit = X_rp

    centroid, cov, cov_inv, _ = _fit_robust_rp_reference(
        X_fit, n_axes, alpha=_RP_OUTLIER_ALPHA, min_rp=_MIN_RP_FOR_ROBUST,
    )

    distances = _compute_mahalanobis_distances(X, centroid, cov_inv)
    # Largest-gap boundary on this replicate's RP-gene distances, rather than a
    # fixed chi-squared quantile. The cut is computed from the unique resampled
    # (in-bag) RP genes — the cohort the centroid was actually fit on — so the
    # boundary lands in the gap between the compact RP cohort and the distant
    # outliers, recovering legitimate cohort members the p95 cut excluded while
    # still keeping the far outliers out.
    rp_dist_inbag = distances[np.unique(boot_rp_idx)]
    threshold, _ = _gap_boundary(rp_dist_inbag, n_axes)
    opt_mask = distances <= threshold

    # Out-of-bag recovery: which RP indices were NEVER drawn this round?
    in_bag = set(int(i) for i in boot_rp_idx)
    oob_rp_indices = [int(i) for i in rp_indices if int(i) not in in_bag]
    if oob_rp_indices:
        oob_inside = sum(1 for i in oob_rp_indices if opt_mask[i])
        oob_recovery = oob_inside / len(oob_rp_indices)
    else:
        # Pathological: every RP gene was sampled at least once. Treat
        # as 1.0 by convention (no held-out evidence to disagree with).
        oob_recovery = float("nan")

    cluster = {gid for gid, inside in zip(gene_ids, opt_mask) if inside}
    return cluster, oob_recovery


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

    # ── RP sub-cluster identification + stability validation ─────────
    # If the RP genes form more than one population in COA space, validate
    # each candidate by clusterwise Jaccard bootstrap (Hennig 2007) and anchor
    # the optimized core on the single stable cluster whose codon usage best
    # matches the ribosomal-protein RSCU consensus. This prevents the
    # membership-frequency bootstrap below from pooling biologically distinct
    # RP cohorts (e.g. canonical RP genes + an HGT/paralog cohort) into one
    # averaged reference. With a single RP population this is a no-op: the
    # full RP index set is used, exactly as before.
    X_rp_all = X[rp_indices]
    rp_ids_in_coa = [coa_gene_ids[i] for i in rp_indices]
    anchor_rp_indices = rp_indices
    subcluster_records: list[dict] = []
    anchor_reason = "single_rp_population"
    try:
        validated = _validate_rp_subclusters(X_rp_all, rp_ids_in_coa)
        n_real = sum(1 for c in validated if c.get("stability_label") == "stable")
        subcluster_records = [
            {
                "label": c.get("label"),
                "n_rp": c["n"],
                "mean_jaccard": c.get("mean_jaccard"),
                "jaccard_sd": c.get("jaccard_sd"),
                "stability_label": c.get("stability_label"),
                "rp_cosine_sim": c.get("rp_cosine_sim"),
            }
            for c in validated
        ]
        if len(validated) > 1 and n_real >= 2:
            anchor, anchor_reason = _select_rp_anchor(validated, ffn_path, rp_rscu_df)
            anchor_ids = set(anchor["gene_ids"])
            _row_of = {gid: i for i, gid in enumerate(coa_gene_ids)}
            anchor_rp_indices = np.array(
                [_row_of[g] for g in anchor_ids if g in _row_of], dtype=int
            )
            for c in validated:
                c["is_anchor"] = (c is anchor)
            logger.info(
                "RP cluster identification: %d candidate sub-cluster(s), %d stable "
                "(Jaccard>=%.2f); anchored on %d-gene cluster (reason=%s, "
                "cosine=%s). Sub-cluster sizes/Jaccard: %s",
                len(validated), n_real, _CLUSTER_STABLE_JACCARD,
                len(anchor_rp_indices), anchor_reason,
                anchor.get("rp_cosine_sim"),
                [(c["n"], c.get("mean_jaccard"), c.get("stability_label")) for c in validated],
            )
        elif len(validated) > 1:
            logger.info(
                "RP sub-cluster split detected (%d candidates) but only %d passed "
                "stability validation (Jaccard>=%.2f); treating RP genes as a single "
                "population (no confident second cluster). Sizes/Jaccard: %s",
                len(validated), n_real, _CLUSTER_STABLE_JACCARD,
                [(c["n"], c.get("mean_jaccard"), c.get("stability_label")) for c in validated],
            )
            anchor_reason = "split_not_stable_pooled"
    except Exception as e:
        logger.warning(
            "RP sub-cluster validation failed (%s); using full RP set as anchor.", e
        )
        anchor_rp_indices = rp_indices

    if len(anchor_rp_indices) < 3:
        anchor_rp_indices = rp_indices  # safety: never anchor on < 3 genes

    results["rp_subcluster_stability"] = subcluster_records
    results["rp_anchor_reason"] = anchor_reason
    results["rp_anchor_n_genes"] = int(len(anchor_rp_indices))

    # ── Bootstrap ───────────────────────────────────────────────────
    # Each iteration is independent and the result (a set of gene IDs) is
    # cheap to pickle, so this is a textbook joblib loop. We pass the
    # iteration index as the per-iteration seed to keep bit-identical
    # results against the previous serial path; the parallel-aware
    # helper still spawns child rngs for any callee that asks for one.
    from codonpipe.utils._parallel import parallel_perm

    def _boot_iter(rng: np.random.Generator, idx: int) -> tuple[set[str], float]:
        # _bootstrap_rp_reference takes a deterministic ``seed`` integer.
        # Pass ``idx`` so the per-iteration seed is identical to the
        # serial code's loop variable — reproducibility against pre-
        # parallelisation runs is preserved exactly. Returns (cluster,
        # oob_recovery) so the outer loop can report predictive
        # stability alongside internal consistency.
        return _bootstrap_rp_reference(
            X, coa_gene_ids, anchor_rp_indices, n_axes,
            multiplier=0.0,  # unused, chi-squared threshold
            seed=idx,
        )

    boot_results: list[tuple[set[str], float]] = parallel_perm(
        n_bootstraps,
        _boot_iter,
        master_seed=0,  # individual seeds come from idx; master unused
        desc="cluster-stability-bootstrap",
    )
    boot_clusters: list[set[str]] = [r[0] for r in boot_results]
    oob_recoveries = np.array(
        [r[1] for r in boot_results if not np.isnan(r[1])]
    )
    mean_oob_recovery = (
        float(oob_recoveries.mean()) if oob_recoveries.size else float("nan")
    )

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

    # ── RSCU-similarity rescue of borderline RP genes ──────────────
    # The membership-frequency boundary is geometric (Mahalanobis distance in
    # COA space). On genomes where the RP cohort's distances trail off smoothly
    # with no clean gap (e.g. B. subtilis), that boundary clips legitimate
    # ribosomal-protein genes whose codon usage is still strongly RP-like — the
    # distance is marginally beyond the cut, but the gene IS ribosomal. Because
    # the optimized core is meant to be "genes with ribosomal-protein-like codon
    # usage", we rescue any annotated RP gene whose RSCU cosine to the RP
    # consensus is at least as high as the weakest RP gene already in the core.
    # This recovers genuine members while still excluding the truly drifted RP
    # outliers (whose codon usage has diverged from the consensus). The rescue
    # is RP-only and threshold-data-driven (no magic constant), so it cannot
    # pull in non-RP bulk genes.
    rescued_ids: set = set()
    try:
        # Representative per-gene distances + boundary for the geometric sanity
        # cap: fit the anchor RP centroid once (deterministically, no resample)
        # on its dense core and take the gap boundary, mirroring the per-replicate
        # geometry without the bootstrap noise.
        _rescue_dist = None
        _rescue_boundary = None
        try:
            _anchor_X = X[anchor_rp_indices]
            _au = np.unique(anchor_rp_indices)
            _ids_u = [str(i) for i in _au]
            _, _core_strs, _ = _select_rp_dense_core(X[_au], _ids_u, density_pctl=50)
            _cset = set(_core_strs)
            _Xfit = np.array([X[i] for i in _au if str(i) in _cset])
            if len(_Xfit) < max(3, n_axes + 1):
                _Xfit = _anchor_X
            _cen, _cov, _cinv, _ = _fit_robust_rp_reference(
                _Xfit, n_axes, alpha=_RP_OUTLIER_ALPHA, min_rp=_MIN_RP_FOR_ROBUST,
            )
            _rescue_dist = _compute_mahalanobis_distances(X, _cen, _cinv)
            _rescue_boundary, _ = _gap_boundary(_rescue_dist[anchor_rp_indices], n_axes)
        except Exception as _e:
            logger.debug("Rescue distance prep failed (%s); cosine-only rescue", _e)

        rescued_ids = _rescue_rp_by_rscu(
            core_ids, rp_gene_ids, ffn_path, rp_rscu_df,
            distances=_rescue_dist, gene_ids=coa_gene_ids, boundary=_rescue_boundary,
        )
        if rescued_ids:
            core_ids = core_ids | rescued_ids
            core_mask = np.array([gid in core_ids for gid in all_gene_ids])
            n_core = int(core_mask.sum())
            logger.info(
                "RSCU-similarity rescue: added %d borderline RP gene(s) whose "
                "codon usage is as ribosomal-like as the weakest core member "
                "(core %d -> %d genes)",
                len(rescued_ids), n_core - len(rescued_ids), n_core,
            )
    except Exception as e:
        logger.debug("RSCU-similarity rescue skipped (%s)", e)

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
        # Predictive (hold-out) stability: across replicates, what fraction
        # of out-of-bag RP genes does the bootstrap centroid still place
        # inside the cluster? High mean_jaccard with low mean_oob_recovery
        # would mean replicates agree with each other but generalise
        # poorly to held-out RP genes — a sign of overfitting to the
        # specific RP set.
        "mean_oob_recovery": (
            round(mean_oob_recovery, 4) if not np.isnan(mean_oob_recovery) else None
        ),
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
        "Stability: Jaccard=%.3f, OOB recovery=%.3f, core=%d genes "
        "(%.0f%% RP coverage), composite=%.3f",
        mean_jaccard,
        mean_oob_recovery if not np.isnan(mean_oob_recovery) else 0.0,
        n_core, rp_coverage * 100, composite,
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

    # Persist RP sub-cluster identification + stability (Hennig Jaccard) so the
    # anchor decision is auditable. One row per candidate RP sub-cluster.
    if subcluster_records:
        sc_df = pd.DataFrame(subcluster_records)
        sc_df.insert(0, "sample_id", sample_id)
        sc_df["anchor_reason"] = anchor_reason
        sc_path = stab_dir / f"{sample_id}_rp_subcluster_stability.tsv"
        sc_df.to_csv(sc_path, sep="\t", index=False)
        results["rp_subcluster_stability_path"] = sc_path

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
