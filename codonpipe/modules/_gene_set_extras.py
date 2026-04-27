"""Additional GOI analyses and publication-ready plots.

Three deliverables, each with both a tabular output and a figure:

1. Expression-tier breakdown
   <sid>_goi_expression_tier_tests.tsv  — per (metric, tier) Fisher's exact
   <sid>_goi_expression_tiers.png/.svg  — paired stacked bars, GOI vs background

2. HGT and codon-usage anomaly map
   <sid>_goi_anomaly_map.png/.svg       — mahalanobis vs GC3-deviation scatter
                                          with GOIs highlighted by partition,
                                          plus per-flag count panel

3. Within-GOI clustering + driver analysis
   <sid>_goi_internal_clusters.tsv      — gene → internal cluster id
   <sid>_goi_internal_drivers.tsv       — codons / metrics that distinguish each cluster
   <sid>_goi_internal_structure.png/.svg — dendrogram + heatmap + cluster scatter

Defensible-by-design:
  * Internal clustering operates on CLR-Δ (gene minus genome-mean) so the
    clusters reflect codon-preference shifts, not absolute RSCU magnitudes
    that would be confounded by gene length.
  * Pairwise distance is Aitchison (CLR + Euclidean), the standard for
    compositional comparison (Aitchison 1986).
  * Number of clusters is auto-selected via silhouette score with a min
    cluster size guard, so we don't fabricate structure on small GOI sets.
  * Driver tests use Mann-Whitney + BH FDR; reported with effect sizes
    (Cliff's delta for codons, Kruskal-Wallis H for scalars).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform

from codonpipe.modules.gene_set import (
    _drop_redundant_codon_per_family,
    clr_transform,
)
from codonpipe.utils.codon_tables import COL_GENE, RSCU_COLUMN_NAMES
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Expression-tier breakdown
# ──────────────────────────────────────────────────────────────────────────────


_EXPRESSION_METRICS = ("MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop")
_TIERS = ("high", "medium", "low")


def compute_expression_tier_breakdown(
    summary_df: pd.DataFrame,
    base_df: pd.DataFrame,
    metrics: tuple[str, ...] = _EXPRESSION_METRICS,
) -> pd.DataFrame:
    """Per (metric, tier) GOI vs background composition with Fisher's exact.

    Args:
        summary_df: Per-GOI table containing ``<metric>_class`` columns
            (e.g. MELP_class, CAI_class, ...). Tiers expected: 'high',
            'medium', 'low', plus possibly 'unknown' for genes without
            scores. 'unknown' is excluded from the test.
        base_df: Per-gene table for the *full* genome with the same class
            columns. The function compares GOI tier counts against the
            *rest of genome* (genes not in summary_df).
        metrics: Which metrics to test. Skips any not present in either table.

    Returns:
        DataFrame with columns: metric, tier, n_goi, n_background,
        frac_goi, frac_background, odds_ratio, p_value, p_adjusted, significant.
        BH FDR is applied across the full (metric, tier) panel.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    goi_genes = set(summary_df[COL_GENE]) if COL_GENE in summary_df.columns else set()
    bg = base_df[~base_df[COL_GENE].isin(goi_genes)] if goi_genes else base_df

    rows = []
    for metric in metrics:
        col = f"{metric}_class"
        if col not in summary_df.columns or col not in bg.columns:
            continue
        # Drop 'unknown' from both sides — those genes have no score, so
        # they don't belong in a high/medium/low test.
        goi_known = summary_df[summary_df[col].isin(_TIERS)]
        bg_known = bg[bg[col].isin(_TIERS)]
        if len(goi_known) < 5 or len(bg_known) < 30:
            continue
        n_goi_total = len(goi_known)
        n_bg_total = len(bg_known)
        for tier in _TIERS:
            a = int((goi_known[col] == tier).sum())
            b = int((bg_known[col] == tier).sum())
            c = n_goi_total - a
            d = n_bg_total - b
            try:
                odds, p = sp_stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except ValueError:
                odds, p = float("nan"), float("nan")
            rows.append({
                "metric": metric,
                "tier": tier,
                "n_goi": a,
                "n_background": b,
                "frac_goi": a / n_goi_total if n_goi_total else float("nan"),
                "frac_background": b / n_bg_total if n_bg_total else float("nan"),
                "odds_ratio": float(odds) if not np.isnan(odds) else float("nan"),
                "p_value": float(p) if not np.isnan(p) else float("nan"),
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p_adjusted"] = benjamini_hochberg(df["p_value"].values)
    df["significant"] = df["p_adjusted"] < 0.05
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Within-GOI clustering + driver analysis
# ──────────────────────────────────────────────────────────────────────────────


def _build_clr_delta_matrix(
    rscu_gene_df: pd.DataFrame,
    rscu_genome: dict[str, float] | None,
    rscu_cols_indep: list[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build the per-gene CLR-Δ matrix for *rscu_gene_df*.

    Returns (matrix, gene_ids, codon_columns) where matrix has shape
    (n_genes, n_codons_independent) and entries are
    CLR(rscu_gene)_i − CLR(rscu_genome)_i.
    """
    if rscu_genome is not None and len(rscu_genome) > 0:
        genome_vec = np.array([rscu_genome.get(c, np.nan) for c in rscu_cols_indep])
    else:
        genome_vec = rscu_gene_df[rscu_cols_indep].mean().values
    if np.any(np.isnan(genome_vec)):
        fallback = rscu_gene_df[rscu_cols_indep].mean().values
        nan_mask = np.isnan(genome_vec)
        genome_vec[nan_mask] = fallback[nan_mask]
    clr_genome = clr_transform(genome_vec)

    gene_mat = rscu_gene_df[rscu_cols_indep].values
    delta_mat = np.empty_like(gene_mat, dtype=float)
    for i in range(gene_mat.shape[0]):
        delta_mat[i, :] = clr_transform(gene_mat[i, :]) - clr_genome
    gene_ids = list(rscu_gene_df[COL_GENE].values)
    return delta_mat, gene_ids, rscu_cols_indep


def _auto_select_n_clusters(
    distance_matrix: np.ndarray,
    linkage_mat: np.ndarray,
    min_cluster_size: int = 3,
    k_min: int = 2,
    k_max: int = 8,
) -> int:
    """Pick the number of clusters by silhouette score.

    Constrains every candidate K to produce clusters all of size >=
    *min_cluster_size*; otherwise that K is skipped. Returns 1 if the
    distance matrix is too small for any valid K.
    """
    n = distance_matrix.shape[0]
    if n < max(2 * min_cluster_size, 4):
        return 1
    from sklearn.metrics import silhouette_score
    best_k = 1
    best_score = -1.0
    for k in range(k_min, min(k_max, n - 1) + 1):
        labels = fcluster(linkage_mat, t=k, criterion="maxclust")
        # Reject if any cluster too small
        sizes = np.bincount(labels)
        if (sizes[1:] < min_cluster_size).any():
            continue
        if len(set(labels)) < 2:
            continue
        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _cluster_drivers(
    delta_mat: np.ndarray,
    gene_ids: list[str],
    codon_cols: list[str],
    cluster_labels: np.ndarray,
    summary_df: pd.DataFrame | None = None,
    scalar_metrics: tuple[str, ...] = (
        "CAI", "MELP", "Fop", "ENC", "GC3", "ENCprime", "MILC",
        "mahalanobis_dist", "mahal_cluster_distance", "membership_score",
        "cbi_rp", "cbi_mahal", "length",
    ),
) -> pd.DataFrame:
    """Identify codons + scalar metrics that drive each within-GOI cluster.

    For each cluster, runs Mann-Whitney U comparing its members against all
    other clusters combined, for every codon (CLR-Δ) and every available
    scalar metric. Reports test statistic, two-sided p, BH-corrected p
    (across all (cluster, feature) tests), and Cliff's delta as effect size.
    """
    n_clusters = len(set(cluster_labels))
    if n_clusters < 2:
        return pd.DataFrame()

    rows = []
    # Codon-level tests: CLR-Δ per codon, per cluster vs the rest
    delta_df = pd.DataFrame(delta_mat, columns=codon_cols)
    delta_df["__cluster"] = cluster_labels
    delta_df[COL_GENE] = gene_ids

    for cluster_id in sorted(set(cluster_labels)):
        in_cluster = delta_df["__cluster"] == cluster_id
        if in_cluster.sum() < 3 or (~in_cluster).sum() < 3:
            continue
        for codon in codon_cols:
            x = delta_df.loc[in_cluster, codon].dropna().values
            y = delta_df.loc[~in_cluster, codon].dropna().values
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
            except ValueError:
                continue
            delta_eff = _cliffs_delta_simple(x, y)
            rows.append({
                "cluster_id": int(cluster_id),
                "feature_type": "codon",
                "feature": codon,
                "median_in": float(np.median(x)),
                "median_out": float(np.median(y)),
                "U_statistic": float(u),
                "p_value": float(p),
                "cliffs_delta": float(delta_eff),
            })

    # Scalar metric tests: pull from summary_df if provided
    if summary_df is not None and not summary_df.empty:
        sm_df = summary_df[[COL_GENE] + [
            m for m in scalar_metrics if m in summary_df.columns
        ]].copy()
        sm_df["__cluster"] = sm_df[COL_GENE].map({
            g: c for g, c in zip(gene_ids, cluster_labels)
        })
        sm_df = sm_df.dropna(subset=["__cluster"])
        for cluster_id in sorted(set(cluster_labels)):
            in_cluster = sm_df["__cluster"] == cluster_id
            if in_cluster.sum() < 3 or (~in_cluster).sum() < 3:
                continue
            for metric in scalar_metrics:
                if metric not in sm_df.columns:
                    continue
                x = sm_df.loc[in_cluster, metric].dropna().values
                y = sm_df.loc[~in_cluster, metric].dropna().values
                if len(x) < 3 or len(y) < 3:
                    continue
                try:
                    u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
                except ValueError:
                    continue
                delta_eff = _cliffs_delta_simple(x, y)
                rows.append({
                    "cluster_id": int(cluster_id),
                    "feature_type": "scalar",
                    "feature": metric,
                    "median_in": float(np.median(x)),
                    "median_out": float(np.median(y)),
                    "U_statistic": float(u),
                    "p_value": float(p),
                    "cliffs_delta": float(delta_eff),
                })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["p_adjusted"] = benjamini_hochberg(out["p_value"].values)
    out["significant"] = out["p_adjusted"] < 0.05
    out["abs_effect"] = out["cliffs_delta"].abs()
    return out.sort_values(
        ["cluster_id", "abs_effect"], ascending=[True, False],
    ).reset_index(drop=True)


def _cliffs_delta_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Compact Cliff's delta (matches gene_set._cliffs_delta but doesn't import)."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (nx * ny))


def cluster_within_goi(
    goi_rscu_df: pd.DataFrame,
    rscu_genome: dict[str, float] | None,
    summary_df: pd.DataFrame | None = None,
    n_clusters: int | None = None,
    min_cluster_size: int = 3,
) -> dict:
    """Hierarchical clustering of GOIs by their CLR-Δ codon profile.

    Returns a dict with:
      delta_mat        — (n_goi, 38) CLR-Δ matrix
      gene_ids         — list of GOI gene ids in row order
      codon_cols       — 38 independent codon column names
      distance_matrix  — square pairwise Aitchison distance matrix
      linkage          — scipy linkage matrix (Ward)
      cluster_id       — pd.Series (gene → integer cluster id, 1-indexed)
      n_clusters       — number of clusters chosen
      drivers_df       — output of _cluster_drivers (codon + scalar drivers)
    """
    if goi_rscu_df is None or len(goi_rscu_df) < 2 * min_cluster_size:
        return {}

    rscu_cols_full = [c for c in RSCU_COLUMN_NAMES if c in goi_rscu_df.columns]
    rscu_cols = _drop_redundant_codon_per_family(rscu_cols_full)
    if len(rscu_cols) < 5:
        return {}

    # Drop GOIs with NaN RSCU (rare but breaks CLR)
    valid_mask = ~goi_rscu_df[rscu_cols].isna().any(axis=1)
    goi_clean = goi_rscu_df[valid_mask].reset_index(drop=True)
    if len(goi_clean) < 2 * min_cluster_size:
        return {}

    delta_mat, gene_ids, codon_cols = _build_clr_delta_matrix(
        goi_clean, rscu_genome, rscu_cols,
    )

    # Pairwise Aitchison distance = Euclidean on CLR-Δ vectors
    distance_matrix = squareform(pdist(delta_mat, metric="euclidean"))
    # Ward linkage
    link = linkage(pdist(delta_mat, metric="euclidean"), method="ward")

    if n_clusters is None:
        n_clusters = _auto_select_n_clusters(
            distance_matrix, link, min_cluster_size=min_cluster_size,
        )
    cluster_labels = fcluster(link, t=n_clusters, criterion="maxclust") if n_clusters > 1 \
        else np.ones(len(gene_ids), dtype=int)

    cluster_series = pd.Series(cluster_labels, index=gene_ids, name="cluster_id")

    drivers_df = pd.DataFrame()
    if n_clusters >= 2:
        drivers_df = _cluster_drivers(
            delta_mat, gene_ids, codon_cols, cluster_labels,
            summary_df=summary_df,
        )

    return {
        "delta_mat": delta_mat,
        "gene_ids": gene_ids,
        "codon_cols": codon_cols,
        "distance_matrix": distance_matrix,
        "linkage": link,
        "cluster_id": cluster_series,
        "n_clusters": int(n_clusters),
        "drivers_df": drivers_df,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figure rendering
# ──────────────────────────────────────────────────────────────────────────────


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        logger.warning("matplotlib unavailable; skipping extra figures (%s)", e)
        return None


def render_expression_tier_figure(
    output_dir: Path,
    sample_id: str,
    tier_table: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    """Paired stacked bars per metric: top = GOI, bottom = background.

    Significance asterisks above each tier segment from the BH-corrected
    Fisher tests.
    """
    plt = _ensure_matplotlib()
    if plt is None or tier_table is None or tier_table.empty:
        return (None, None)

    metrics = list(tier_table["metric"].unique())
    fig, axes = plt.subplots(
        nrows=len(metrics), ncols=1,
        figsize=(11, 1.4 * len(metrics) + 1.5),
        constrained_layout=True,
        sharex=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    tier_colors = {"high": "#d62728", "medium": "#7f7f7f", "low": "#1f77b4"}

    for ax, metric in zip(axes, metrics):
        sub = tier_table[tier_table["metric"] == metric].set_index("tier")
        # Re-order to high/medium/low for consistent reading
        sub = sub.reindex(_TIERS)

        # Two horizontal stacked bars: GOI fractions on top, background on bottom
        cumulative_goi = 0.0
        cumulative_bg = 0.0
        for tier in _TIERS:
            if tier not in sub.index:
                continue
            row = sub.loc[tier]
            f_goi = row.get("frac_goi", 0)
            f_bg = row.get("frac_background", 0)
            color = tier_colors[tier]
            ax.barh(1, f_goi, left=cumulative_goi, height=0.5,
                    color=color, edgecolor="white", linewidth=1)
            ax.barh(0, f_bg, left=cumulative_bg, height=0.5,
                    color=color, edgecolor="white", linewidth=1)

            # Significance star at the centre of each GOI segment
            p_adj = row.get("p_adjusted", float("nan"))
            star = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            label = f"{tier} ({int(row['n_goi'])})"
            if star:
                label += f" {star}"
            if f_goi > 0.06:
                ax.text(cumulative_goi + f_goi / 2, 1, label,
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
            if f_bg > 0.06:
                ax.text(cumulative_bg + f_bg / 2, 0, f"{tier} ({int(row['n_background'])})",
                        ha="center", va="center", fontsize=7, color="white")

            cumulative_goi += f_goi
            cumulative_bg += f_bg

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["bg", "GOI"], fontsize=9)
        ax.set_xlim(0, 1.0)
        ax.set_title(metric, fontsize=10, loc="left")
        ax.set_xlabel("")
        if ax is axes[-1]:
            ax.set_xlabel("Fraction of genes")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    fig.suptitle(
        f"GOI vs background — expression-tier composition  ({sample_id})\n"
        "Significance: * BH-p<0.05, ** <0.01, *** <0.001",
        fontsize=11, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_goi_expression_tiers.png"
    svg = out_dir / f"{sample_id}_goi_expression_tiers.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


def render_anomaly_map_figure(
    output_dir: Path,
    sample_id: str,
    summary_df: pd.DataFrame,
    base_df: pd.DataFrame,
    partition_per_gene: pd.DataFrame | None = None,
) -> tuple[Path | None, Path | None]:
    """Two-panel anomaly visualization.

    Left: scatter of mahalanobis_dist (genome centroid) vs gc3_deviation
        with all genome genes as background and GOIs overlaid coloured by
        partition (mahal_cluster / bulk / outlier). 95th-percentile lines
        on each axis.

    Right: per-flag GOI bar chart — counts of GOIs flagged by
        hgt_flag_combined / hgt_flag_fdr / gc3_outlier, with the
        background prevalence as a horizontal reference.
    """
    plt = _ensure_matplotlib()
    if plt is None or summary_df is None or summary_df.empty:
        return (None, None)

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.05])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    has_mahal = "mahalanobis_dist" in base_df.columns and \
                "gc3_deviation" in base_df.columns
    if not has_mahal:
        axL.text(0.5, 0.5, "Mahalanobis / GC3 deviation unavailable",
                 ha="center", va="center", transform=axL.transAxes)
    else:
        bg = base_df.dropna(subset=["mahalanobis_dist", "gc3_deviation"])
        goi_genes = set(summary_df[COL_GENE]) if COL_GENE in summary_df.columns else set()
        bg_only = bg[~bg[COL_GENE].isin(goi_genes)]
        goi = bg[bg[COL_GENE].isin(goi_genes)].copy()

        # Background scatter
        axL.scatter(
            bg_only["mahalanobis_dist"], bg_only["gc3_deviation"],
            s=4, alpha=0.18, color="#888888", label=f"background (n={len(bg_only)})",
        )

        # Overlay GOIs coloured by partition if provided
        partition_colors = {
            "mahal_cluster": "#2ca02c",
            "bulk": "#9467bd",
            "outlier": "#d62728",
            "unknown": "#cccccc",
        }
        if partition_per_gene is not None and not partition_per_gene.empty:
            goi_p = goi.merge(partition_per_gene, on=COL_GENE, how="left")
            for cat, color in partition_colors.items():
                sub = goi_p[goi_p["partition"] == cat]
                if not sub.empty:
                    axL.scatter(
                        sub["mahalanobis_dist"], sub["gc3_deviation"],
                        s=42, color=color, edgecolor="black", linewidth=0.5,
                        label=f"GOI: {cat} (n={len(sub)})", zorder=4,
                    )
        else:
            axL.scatter(
                goi["mahalanobis_dist"], goi["gc3_deviation"],
                s=42, color="#d62728", edgecolor="black", linewidth=0.5,
                label=f"GOI (n={len(goi)})", zorder=4,
            )

        # 95th-percentile reference lines on the background distribution
        if len(bg_only) > 20:
            x95 = float(np.quantile(bg_only["mahalanobis_dist"].abs(), 0.95))
            y95 = float(np.quantile(bg_only["gc3_deviation"].abs(), 0.95))
            axL.axvline(x95, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
            axL.axhline(y95, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
            axL.axhline(-y95, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
            axL.text(x95, axL.get_ylim()[1] * 0.95,
                     " 95% genome", fontsize=7, color="black", alpha=0.7,
                     ha="left", va="top")

        axL.set_xlabel("Mahalanobis distance (genome centroid)\n← typical          unusual codon usage →")
        axL.set_ylabel("GC3 deviation (gene − genome mean)")
        axL.set_title("A. Anomaly scatter (codon-usage outliers + GC3 outliers)")
        axL.legend(loc="best", fontsize=7, frameon=False)
        for s in ("top", "right"):
            axL.spines[s].set_visible(False)

    # Right panel: per-flag count bars
    flag_cols = [c for c in
                 ("hgt_flag_combined", "hgt_flag_fdr", "gc3_outlier")
                 if c in base_df.columns and c in summary_df.columns]
    if not flag_cols:
        axR.text(0.5, 0.5, "Flag columns unavailable",
                 ha="center", va="center", transform=axR.transAxes)
    else:
        rows = []
        for f in flag_cols:
            # Coerce via boolean dtype to dodge pandas' object-downcast FutureWarning
            goi_n = int(summary_df[f].astype("boolean").fillna(False).astype(bool).sum())
            goi_total = int(summary_df[f].notna().sum())
            bg_subset = base_df[~base_df[COL_GENE].isin(set(summary_df[COL_GENE]))]
            bg_n = int(bg_subset[f].astype("boolean").fillna(False).astype(bool).sum())
            bg_total = int(bg_subset[f].notna().sum())
            rows.append({
                "flag": f,
                "goi_frac": goi_n / goi_total if goi_total else 0,
                "bg_frac": bg_n / bg_total if bg_total else 0,
                "n_goi": goi_n,
            })
        df = pd.DataFrame(rows)
        ypos = np.arange(len(df))
        axR.barh(ypos, df["goi_frac"], color="#d62728", height=0.55,
                 edgecolor="black", linewidth=0.5, label="GOI")
        # Background as small ticks behind each bar
        for y, bf in zip(ypos, df["bg_frac"]):
            axR.scatter([bf], [y], marker="|", s=200, color="black",
                        zorder=5, label="background" if y == 0 else None)
        for y, n, gf in zip(ypos, df["n_goi"], df["goi_frac"]):
            axR.text(gf + 0.01, y, f" n={n}", va="center", fontsize=7)
        axR.set_yticks(ypos)
        axR.set_yticklabels(df["flag"], fontsize=8)
        axR.set_xlabel("Fraction flagged")
        axR.set_xlim(0, max(0.3, df["goi_frac"].max() * 1.3 if not df.empty else 0.3))
        axR.set_title("B. Flag rates: GOI vs background")
        axR.legend(loc="best", fontsize=7, frameon=False)
        for s in ("top", "right"):
            axR.spines[s].set_visible(False)

    fig.suptitle(
        f"GOI anomaly map — {sample_id}",
        fontsize=11, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_goi_anomaly_map.png"
    svg = out_dir / f"{sample_id}_goi_anomaly_map.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


def render_internal_structure_figure(
    output_dir: Path,
    sample_id: str,
    cluster_result: dict,
    summary_df: pd.DataFrame | None,
) -> tuple[Path | None, Path | None]:
    """Three-panel publication figure for within-GOI structure.

    Left: dendrogram (Ward linkage on Aitchison distance).
    Centre: heatmap of CLR-Δ codon profile, rows ordered by dendrogram,
            columns ordered by their hierarchical clustering. Side annotation
            strip on the right shows cluster_id + partition + expression class.
    Right: per-cluster top-driver bar chart (top 6 codons + scalars by
           absolute Cliff's delta, BH-significant only).
    """
    plt = _ensure_matplotlib()
    if plt is None or not cluster_result:
        return (None, None)

    delta_mat = cluster_result["delta_mat"]
    gene_ids = cluster_result["gene_ids"]
    codon_cols = cluster_result["codon_cols"]
    link = cluster_result["linkage"]
    cluster_id = cluster_result["cluster_id"]
    n_clusters = cluster_result["n_clusters"]
    drivers_df = cluster_result.get("drivers_df", pd.DataFrame())

    n_goi = len(gene_ids)
    fig = plt.figure(
        figsize=(15, max(6, min(0.20 * n_goi + 4, 22))),
        constrained_layout=True,
    )
    # Layout: dendrogram (left, narrow) + heatmap (centre) + annotations (thin) + drivers (right)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[0.9, 4.0, 0.45, 2.5],
    )
    axDend = fig.add_subplot(gs[0, 0])
    axHeat = fig.add_subplot(gs[0, 1])
    axAnno = fig.add_subplot(gs[0, 2])
    axDrv = fig.add_subplot(gs[0, 3])

    # ── Dendrogram ────────────────────────────────────────────────────
    ddata = dendrogram(link, orientation="left", no_plot=True)
    leaves = ddata["leaves"][::-1]  # top-to-bottom in plot
    dendrogram(
        link, orientation="left", ax=axDend, no_labels=True,
        color_threshold=0.7 * max(link[:, 2]),
        link_color_func=lambda k: "#666666",
    )
    axDend.invert_yaxis()
    axDend.set_xticks([])
    axDend.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        axDend.spines[s].set_visible(False)
    axDend.set_title("Ward linkage", fontsize=9)

    # ── Heatmap of CLR-Δ ──────────────────────────────────────────────
    # Cluster columns too for nicer banding
    col_link = linkage(pdist(delta_mat.T, metric="euclidean"), method="ward")
    col_order = dendrogram(col_link, no_plot=True)["leaves"]
    mat_ordered = delta_mat[np.ix_(leaves, col_order)]
    codon_ordered = [codon_cols[i] for i in col_order]

    vmax = float(np.percentile(np.abs(mat_ordered), 98))
    im = axHeat.imshow(
        mat_ordered, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    axHeat.set_xticks(np.arange(len(codon_ordered)))
    axHeat.set_xticklabels(codon_ordered, rotation=90, fontsize=6)
    axHeat.set_yticks(np.arange(n_goi))
    label_every = max(1, n_goi // 30)
    yticks = np.arange(n_goi)[::label_every]
    axHeat.set_yticks(yticks)
    axHeat.set_yticklabels([gene_ids[leaves[i]] for i in yticks], fontsize=6)
    axHeat.set_title(
        f"CLR-Δ codon profile  (n_GOI = {n_goi}, "
        f"n_clusters = {n_clusters} via silhouette)",
        fontsize=10,
    )
    fig.colorbar(im, ax=axHeat, fraction=0.025, pad=0.01,
                 label="CLR-Δ (gene − genome mean)")

    # ── Annotation strip: cluster_id, partition, expression_class ─────
    annos = []
    legend_chunks = []
    cluster_palette = plt.get_cmap("tab10")
    cluster_colors = {
        cid: cluster_palette((cid - 1) % 10)
        for cid in sorted(set(cluster_id.values))
    }
    for i, leaf_idx in enumerate(leaves):
        gene = gene_ids[leaf_idx]
        cid = int(cluster_id.get(gene, 0))
        annos.append({
            "y": i,
            "cluster": cluster_colors.get(cid, "#cccccc"),
        })
    # Draw cluster strip
    for a in annos:
        axAnno.add_patch(plt.Rectangle(
            (0, a["y"] - 0.5), 1, 1, color=a["cluster"], linewidth=0,
        ))
    if summary_df is not None and "partition" in summary_df.columns:
        part_colors = {
            "mahal_cluster": "#2ca02c", "bulk": "#9467bd",
            "outlier": "#d62728", "unknown": "#cccccc",
        }
        part_map = dict(zip(summary_df[COL_GENE], summary_df["partition"]))
        for i, leaf_idx in enumerate(leaves):
            gene = gene_ids[leaf_idx]
            part = part_map.get(gene, "unknown")
            color = part_colors.get(part, "#cccccc")
            axAnno.add_patch(plt.Rectangle(
                (1, i - 0.5), 1, 1, color=color, linewidth=0,
            ))
        axAnno.set_xlim(0, 2)
    if summary_df is not None and "expression_class" in summary_df.columns:
        cls_colors = {"high": "#d62728", "medium": "#7f7f7f",
                      "low": "#1f77b4", "unknown": "#cccccc"}
        cls_map = dict(zip(summary_df[COL_GENE], summary_df["expression_class"]))
        for i, leaf_idx in enumerate(leaves):
            gene = gene_ids[leaf_idx]
            cls = cls_map.get(gene, "unknown")
            color = cls_colors.get(cls, "#cccccc")
            axAnno.add_patch(plt.Rectangle(
                (2, i - 0.5), 1, 1, color=color, linewidth=0,
            ))
        axAnno.set_xlim(0, 3)
    axAnno.set_ylim(-0.5, n_goi - 0.5)
    axAnno.invert_yaxis()
    axAnno.set_xticks([0.5, 1.5, 2.5][:int(axAnno.get_xlim()[1])])
    xticklabels = ["cluster"]
    if summary_df is not None and "partition" in summary_df.columns:
        xticklabels.append("partition")
    if summary_df is not None and "expression_class" in summary_df.columns:
        xticklabels.append("expr.class")
    axAnno.set_xticks(np.arange(len(xticklabels)) + 0.5)
    axAnno.set_xticklabels(xticklabels, rotation=90, fontsize=8)
    axAnno.set_yticks([])
    for s in ("top", "right", "bottom", "left"):
        axAnno.spines[s].set_visible(False)

    # ── Driver bar chart per cluster ──────────────────────────────────
    if drivers_df is not None and not drivers_df.empty:
        sig = drivers_df[drivers_df["significant"]]
        if sig.empty:
            sig = drivers_df  # fall back to all if nothing crossed BH 0.05

        n_drivers_per_cluster = 6
        rows_to_show = []
        for cid in sorted(sig["cluster_id"].unique()):
            sub = sig[sig["cluster_id"] == cid].head(n_drivers_per_cluster)
            for _, r in sub.iterrows():
                rows_to_show.append(r)
        if rows_to_show:
            d = pd.DataFrame(rows_to_show)
            ypos = np.arange(len(d))
            for i, (_, r) in enumerate(d.iterrows()):
                color = cluster_colors.get(int(r["cluster_id"]), "#888888")
                axDrv.barh(i, r["cliffs_delta"], color=color,
                           edgecolor="black", linewidth=0.4)
                label = f"c{int(r['cluster_id'])}: {r['feature']} ({r['feature_type']})"
                axDrv.text(
                    r["cliffs_delta"] + 0.02 * np.sign(r["cliffs_delta"] or 1),
                    i, label, fontsize=6,
                    va="center",
                    ha="left" if r["cliffs_delta"] >= 0 else "right",
                )
            axDrv.axvline(0, color="black", linewidth=0.5)
            axDrv.set_yticks([])
            axDrv.set_xlabel("Cliff's delta\n(in-cluster − rest)")
            axDrv.set_title("Top drivers per cluster\n(BH-significant)", fontsize=10)
            axDrv.set_xlim(-1.05, 1.05)
            for s in ("top", "right", "left"):
                axDrv.spines[s].set_visible(False)
        else:
            axDrv.text(0.5, 0.5, "No significant drivers",
                       ha="center", va="center", transform=axDrv.transAxes)
    else:
        axDrv.text(0.5, 0.5, "Driver analysis unavailable\n(need >= 2 clusters)",
                   ha="center", va="center", transform=axDrv.transAxes)

    fig.suptitle(
        f"Internal GOI structure — {sample_id}",
        fontsize=12, fontweight="bold",
    )
    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_goi_internal_structure.png"
    svg = out_dir / f"{sample_id}_goi_internal_structure.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)
