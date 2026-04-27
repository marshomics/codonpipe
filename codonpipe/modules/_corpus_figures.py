"""Cross-genome corpus visualizations.

Five publication-ready figures, each answering one question:

  render_multi_overlay_umap(...)          — global structure with confounder check
  render_cluster_signature_heatmap(...)   — what defines each cluster (codon-by-codon)
  render_cluster_drivers_forest(...)      — which features distinguish each cluster
  render_mantel_stratified(...)           — phylogenetic vs ecological signal
  render_focus_genome_locator(...)        — where does my genome sit?

Kept in its own module so the analytic core in cross_genome.py doesn't import
matplotlib at load time. All functions return (png_path, svg_path) and gracefully
return (None, None) if matplotlib isn't available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

logger = logging.getLogger("codonpipe")


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        logger.warning("matplotlib unavailable; skipping corpus figure (%s)", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 1. Multi-overlay UMAP/PCA grid
# ──────────────────────────────────────────────────────────────────────────────


def render_multi_overlay_umap(
    output_dir: Path,
    embedding: np.ndarray,
    sample_ids: list[str],
    cluster_labels: np.ndarray,
    overlay_df: pd.DataFrame,
    method_dim: str = "PCA",
    method_cluster: str = "HDBSCAN",
    overlay_columns: list[str] | None = None,
) -> tuple[Path | None, Path | None]:
    """Six-panel small-multiples scatter of the same 2D embedding.

    Each panel colours the points by a different feature/metadata column,
    so the reader can immediately see whether the embedding aligns with
    cluster assignment, taxonomy, GC content, growth rate, etc. Same
    confounder check, six different lenses.
    """
    plt = _plt()
    if plt is None:
        return (None, None)

    assert embedding.shape[0] == len(sample_ids) == len(cluster_labels)

    # Default overlays. Drop any not present.
    default_overlays = [
        ("cluster", "categorical"),
        ("phylum", "categorical"),
        ("median_gc3", "continuous"),
        ("grodon2_doubling_time_h", "continuous"),
        ("frac_in_optimized_set", "continuous"),
        ("hgt_candidate_frac", "continuous"),
    ]
    if overlay_columns is not None:
        # User-specified ordering; figure out continuous vs categorical from dtype.
        chosen = []
        for c in overlay_columns:
            if c == "cluster":
                chosen.append((c, "categorical"))
            elif c in overlay_df.columns:
                if pd.api.types.is_numeric_dtype(overlay_df[c]):
                    chosen.append((c, "continuous"))
                else:
                    chosen.append((c, "categorical"))
        overlays = chosen
    else:
        overlays = []
        for c, kind in default_overlays:
            if c == "cluster":
                overlays.append((c, kind))
                continue
            if c in overlay_df.columns:
                # For continuous, require some non-NaN values
                if kind == "continuous" and overlay_df[c].notna().sum() >= 3:
                    overlays.append((c, kind))
                elif kind == "categorical" and overlay_df[c].notna().sum() >= 3:
                    overlays.append((c, kind))

    # Cap at 6 panels for the small-multiples layout
    overlays = overlays[:6]
    n = len(overlays)
    if n == 0:
        logger.warning("No overlay columns available for multi-overlay UMAP")
        return (None, None)

    ncols = 3 if n > 3 else n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(5.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    cmap_cat = plt.get_cmap("tab20")
    cmap_cont = plt.get_cmap("viridis")

    for ax, (col, kind) in zip(axes_flat, overlays):
        if col == "cluster":
            values = cluster_labels
        else:
            # Align overlay_df values to sample_ids order
            lookup = dict(zip(overlay_df["sample_id"], overlay_df[col])) \
                if "sample_id" in overlay_df.columns else {}
            values = np.array([lookup.get(sid, np.nan) for sid in sample_ids])

        if kind == "categorical":
            uniq = sorted({v for v in values if v is not None and not (
                isinstance(v, float) and np.isnan(v)
            )})
            for i, lab in enumerate(uniq):
                mask = np.array([v == lab for v in values])
                color = "#cccccc" if (col == "cluster" and lab == -1) else cmap_cat(i % 20)
                ax.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    s=18, alpha=0.75, color=color, edgecolor="black", linewidth=0.15,
                    label=f"{lab} (n={int(mask.sum())})",
                )
            # Compact legend (max 12 entries)
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) <= 12:
                ax.legend(loc="best", fontsize=6, frameon=False)
            else:
                ax.text(
                    0.99, 0.01, f"{len(uniq)} categories",
                    ha="right", va="bottom", fontsize=7, transform=ax.transAxes,
                )
        else:
            # Continuous overlay
            vals_arr = np.array([v if v is not None else np.nan for v in values],
                                dtype=float)
            finite = np.isfinite(vals_arr)
            if not finite.any():
                ax.text(0.5, 0.5, f"{col}: no values",
                        ha="center", va="center", transform=ax.transAxes)
            else:
                # Plot NaN points as grey first so they're under the gradient
                if (~finite).any():
                    ax.scatter(
                        embedding[~finite, 0], embedding[~finite, 1],
                        s=10, alpha=0.3, color="#dddddd",
                    )
                vmin, vmax = np.nanpercentile(vals_arr[finite], [5, 95])
                if vmin == vmax:
                    vmax = vmin + 1e-9
                sc = ax.scatter(
                    embedding[finite, 0], embedding[finite, 1],
                    s=18, c=vals_arr[finite], cmap=cmap_cont,
                    vmin=vmin, vmax=vmax,
                    edgecolor="black", linewidth=0.15,
                )
                fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label=col)

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(f"{method_dim} dim 1", fontsize=8)
        ax.set_ylabel(f"{method_dim} dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide any unused axes
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Cross-genome embedding ({method_dim}, {method_cluster}) — n_genomes = {len(sample_ids)}",
        fontsize=12, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / "corpus_multi_overlay.png"
    svg = out_dir / "corpus_multi_overlay.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Cluster signature heatmap
# ──────────────────────────────────────────────────────────────────────────────


def render_cluster_signature_heatmap(
    output_dir: Path,
    corpus_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    *,
    skip_noise: bool = True,
) -> tuple[Path | None, Path | None]:
    """Two-panel codon-signature heatmap.

    Top: cluster mean CLR-Δ profiles (rows = clusters, cols = 38 codons),
         hierarchical clustering on both axes for interpretable banding.
    Bottom: full genome × codon CLR-Δ matrix, rows ordered by cluster
         (rasterized; supports thousands of genomes).
    """
    plt = _plt()
    if plt is None:
        return (None, None)

    # Two CLR-Δ feature blocks live in corpus_df: delta_clr_mahal_* and
    # delta_clr_rp_*. Use the Mahal block — it's the data-driven anchor.
    # Fall back to RP if Mahal is empty.
    cols_mahal = [c for c in corpus_df.columns if c.startswith("delta_clr_mahal_")]
    cols_rp = [c for c in corpus_df.columns if c.startswith("delta_clr_rp_")]
    if not cols_mahal and not cols_rp:
        logger.warning("No delta_clr columns in corpus; skipping cluster signature heatmap")
        return (None, None)
    cols = cols_mahal if cols_mahal else cols_rp
    block_label = "Mahal" if cols == cols_mahal else "RP"

    arr_labels = np.asarray(cluster_labels)
    valid_idx = arr_labels != -1 if skip_noise else np.ones(len(arr_labels), dtype=bool)
    if valid_idx.sum() < 4:
        logger.warning("Too few non-noise genomes for cluster signature heatmap")
        return (None, None)

    sub_df = corpus_df.iloc[valid_idx].copy()
    sub_labels = arr_labels[valid_idx]
    sub_df["__cluster"] = sub_labels

    # Cluster mean profiles
    cluster_means = sub_df.groupby("__cluster")[cols].mean()
    if len(cluster_means) < 2:
        logger.warning("Fewer than 2 clusters; skipping signature heatmap")
        return (None, None)

    # Hierarchical clustering on row (cluster) and column (codon) of the means
    row_link = linkage(pdist(cluster_means.values, metric="euclidean"), method="ward")
    col_link = linkage(pdist(cluster_means.values.T, metric="euclidean"), method="ward")
    row_order = leaves_list(row_link)
    col_order = leaves_list(col_link)

    cluster_means_ord = cluster_means.iloc[row_order, col_order]

    # Genome × codon ordered by (cluster_id_position_in_row_order, then within cluster)
    sub_df["__row_order"] = sub_df["__cluster"].apply(
        lambda c: list(cluster_means.index[row_order]).index(c)
    )
    sub_df = sub_df.sort_values(["__row_order", "__cluster"]).reset_index(drop=True)
    genome_mat = sub_df[cluster_means_ord.columns].values  # respects col_order

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 2.5],
        width_ratios=[1.0, 0.04],
    )
    axTop = fig.add_subplot(gs[0, 0])
    axBot = fig.add_subplot(gs[1, 0])
    axCb = fig.add_subplot(gs[:, 1])

    # Symmetric colour limits driven by 99th percentile of |values|
    vmax_top = float(np.percentile(np.abs(cluster_means_ord.values), 99))
    vmax_bot = float(np.percentile(np.abs(genome_mat), 99))
    vmax = max(vmax_top, vmax_bot, 0.1)

    im_top = axTop.imshow(
        cluster_means_ord.values, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    axTop.set_yticks(np.arange(len(cluster_means_ord)))
    axTop.set_yticklabels(
        [f"c{int(c)} (n={int((sub_labels == c).sum())})"
         for c in cluster_means_ord.index],
        fontsize=8,
    )
    axTop.set_xticks([])
    axTop.set_title(
        f"Cluster mean CLR-Δ profiles ({block_label} reference, "
        f"{len(cluster_means_ord)} clusters × {len(cols)} codons)",
        fontsize=11,
    )

    # Bottom: rasterized genome × codon
    im_bot = axBot.imshow(
        genome_mat, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest",
    )
    axBot.set_xticks(np.arange(len(cluster_means_ord.columns)))
    axBot.set_xticklabels(
        [c.replace(f"delta_clr_{block_label.lower()}_", "")
         for c in cluster_means_ord.columns],
        rotation=90, fontsize=6,
    )
    axBot.set_ylabel(f"{len(sub_df)} genomes (rows ordered by cluster, then within-cluster)")
    axBot.set_yticks([])
    axBot.set_title(
        f"Full genome × codon CLR-Δ matrix ({block_label} reference)",
        fontsize=10,
    )

    # Cluster boundary lines on bottom panel
    cluster_sizes = sub_df["__cluster"].value_counts().reindex(
        cluster_means_ord.index
    ).values
    cum_pos = 0
    for size in cluster_sizes[:-1]:
        cum_pos += size
        axBot.axhline(cum_pos - 0.5, color="black", linewidth=0.6, alpha=0.7)

    fig.colorbar(im_bot, cax=axCb, label="CLR-Δ (cluster − genome bulk)")

    fig.suptitle(
        "Cluster signature heatmap — what each cluster prefers / avoids relative to its host genome",
        fontsize=12, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / "corpus_cluster_signature.png"
    svg = out_dir / "corpus_cluster_signature.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Per-cluster driver forest plot
# ──────────────────────────────────────────────────────────────────────────────


def render_cluster_drivers_forest(
    output_dir: Path,
    drivers_df: pd.DataFrame,
    *,
    n_per_cluster: int = 8,
    significant_only: bool = True,
) -> tuple[Path | None, Path | None]:
    """Top-N drivers per cluster as a stacked forest plot.

    Each cluster gets its own colour. Within each cluster, features are
    ordered by absolute Cliff's delta and drawn as horizontal bars with
    cluster-coloured fills.
    """
    plt = _plt()
    if plt is None or drivers_df is None or drivers_df.empty:
        return (None, None)

    df = drivers_df
    if significant_only:
        sig = df[df["significant"]] if "significant" in df.columns else df
        if sig.empty:
            sig = df
            logger.info("No BH-significant cluster drivers; falling back to top by |effect|")
    else:
        sig = df

    cluster_ids = sorted(sig["cluster_id"].unique())
    if not cluster_ids:
        return (None, None)

    rows_per_cluster = []
    for cid in cluster_ids:
        sub = sig[sig["cluster_id"] == cid].head(n_per_cluster)
        if not sub.empty:
            rows_per_cluster.append(sub)
    if not rows_per_cluster:
        return (None, None)

    plot_df = pd.concat(rows_per_cluster, ignore_index=True)
    n = len(plot_df)
    fig_h = max(5, n * 0.28 + 2)

    fig, ax = plt.subplots(figsize=(11, fig_h), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    colors = {cid: cmap((i % 10)) for i, cid in enumerate(cluster_ids)}

    ypos = np.arange(n)[::-1]
    for i, (_, r) in enumerate(plot_df.iterrows()):
        c = colors[int(r["cluster_id"])]
        d = float(r["cliffs_delta"])
        ax.barh(ypos[i], d, color=c, edgecolor="black", linewidth=0.4, height=0.7)
        # Label inside or alongside the bar
        label = f"c{int(r['cluster_id'])}: {r['feature']}"
        ax.text(
            d + 0.02 * np.sign(d if d != 0 else 1),
            ypos[i], label,
            fontsize=7, va="center",
            ha="left" if d >= 0 else "right",
        )
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks([])
    ax.set_xlabel("Cliff's delta\n(in-cluster − rest of corpus)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(
        f"Top {n_per_cluster} drivers per cluster"
        + (" (BH-significant)" if significant_only else " (top by effect)"),
        fontsize=11, fontweight="bold",
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    out_dir = Path(output_dir)
    png = out_dir / "corpus_cluster_drivers.png"
    svg = out_dir / "corpus_cluster_drivers.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Mantel-stratified scatter
# ──────────────────────────────────────────────────────────────────────────────


def render_mantel_stratified(
    output_dir: Path,
    sample_ids: list[str],
    signature_dist: np.ndarray,
    phylo_dist: np.ndarray,
    metadata_df: pd.DataFrame | None = None,
    group_col: str = "phylum",
) -> tuple[Path | None, Path | None]:
    """Pairwise distance scatter: signature distance vs phylogenetic distance.

    Stratifies points by within-group vs between-group when *metadata_df*
    has *group_col* (typically phylum). When n is large, uses 2D hexbin
    density; for n<200 plots individual points.
    """
    plt = _plt()
    if plt is None:
        return (None, None)

    n = len(sample_ids)
    if n < 5:
        return (None, None)
    iu = np.triu_indices(n, k=1)
    sig_pairs = signature_dist[iu]
    phy_pairs = phylo_dist[iu]

    finite = np.isfinite(sig_pairs) & np.isfinite(phy_pairs)
    if finite.sum() < 10:
        return (None, None)
    sig_pairs = sig_pairs[finite]
    phy_pairs = phy_pairs[finite]

    # Within / between group classification
    has_meta = (
        metadata_df is not None and not metadata_df.empty
        and group_col in metadata_df.columns and "sample_id" in metadata_df.columns
    )
    within_mask = None
    if has_meta:
        sid_group = dict(zip(metadata_df["sample_id"], metadata_df[group_col]))
        groups = np.array([sid_group.get(s, None) for s in sample_ids])
        # For each upper-triangle pair, check whether the two genomes share group
        i_idx = iu[0][finite]
        j_idx = iu[1][finite]
        within_mask = np.array([
            (groups[a] is not None and groups[b] is not None and groups[a] == groups[b])
            for a, b in zip(i_idx, j_idx)
        ])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)

    n_pairs = len(sig_pairs)
    if n_pairs > 1500:
        # Density hexbin for large corpora
        hb = ax.hexbin(
            phy_pairs, sig_pairs, gridsize=50, cmap="Greys", mincnt=1,
        )
        fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02, label="pair count")
        if within_mask is not None and within_mask.any():
            ax.scatter(
                phy_pairs[within_mask], sig_pairs[within_mask],
                s=4, alpha=0.4, color="#1f77b4",
                label=f"within-{group_col}", zorder=3,
            )
    else:
        if within_mask is not None:
            ax.scatter(
                phy_pairs[~within_mask], sig_pairs[~within_mask],
                s=8, alpha=0.4, color="#d62728",
                label=f"between-{group_col} (n={int((~within_mask).sum())})",
            )
            ax.scatter(
                phy_pairs[within_mask], sig_pairs[within_mask],
                s=10, alpha=0.65, color="#1f77b4",
                label=f"within-{group_col} (n={int(within_mask.sum())})",
            )
        else:
            ax.scatter(
                phy_pairs, sig_pairs, s=8, alpha=0.4, color="#888888",
                label=f"all pairs (n={n_pairs})",
            )

    # Trend lines (overall + stratified)
    if len(phy_pairs) > 5 and np.std(phy_pairs) > 0:
        coef = np.polyfit(phy_pairs, sig_pairs, 1)
        x_line = np.linspace(phy_pairs.min(), phy_pairs.max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line),
                color="black", linewidth=1.0, label="overall trend")
        from scipy.stats import pearsonr
        r_overall, p_overall = pearsonr(phy_pairs, sig_pairs)
        ax.text(
            0.02, 0.98, f"overall r = {r_overall:.3f}, p = {p_overall:.3g}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        if within_mask is not None and within_mask.any() and (~within_mask).any():
            for sub_mask, color, label_prefix in (
                (within_mask, "#1f77b4", "within"),
                (~within_mask, "#d62728", "between"),
            ):
                if sub_mask.sum() < 5 or np.std(phy_pairs[sub_mask]) == 0:
                    continue
                c = np.polyfit(phy_pairs[sub_mask], sig_pairs[sub_mask], 1)
                ax.plot(x_line, np.polyval(c, x_line),
                        color=color, linewidth=0.8, linestyle="--")
                r_s, _ = pearsonr(phy_pairs[sub_mask], sig_pairs[sub_mask])
                ax.text(
                    0.02, 0.92 - 0.05 * (0 if label_prefix == "within" else 1),
                    f"  {label_prefix}-{group_col} r = {r_s:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=8, color=color,
                )

    ax.set_xlabel("Phylogenetic distance")
    ax.set_ylabel("Signature distance (Aitchison)")
    ax.set_title(
        "Mantel scatter — does signature distance recover phylogeny?",
        fontsize=11, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out_dir = Path(output_dir)
    png = out_dir / "corpus_mantel_scatter.png"
    svg = out_dir / "corpus_mantel_scatter.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


# ──────────────────────────────────────────────────────────────────────────────
# Gene-level corpus visualizations
# ──────────────────────────────────────────────────────────────────────────────


def render_gene_level_umap(
    output_dir: Path,
    gene_corpus_df: pd.DataFrame,
    gene_cluster_result: dict,
    host_metadata_df: pd.DataFrame | None = None,
) -> tuple[Path | None, Path | None]:
    """Four-panel small-multiples scatter of the gene-level embedding.

    Each panel shows the same 2-D embedding coloured by:
      A. gene cluster id
      B. host genome (categorical, capped at top-N)
      C. host phylum / habitat (if metadata available)
      D. host median GC3

    Uses density alpha and small markers so 100k+ gene scatters stay readable.
    """
    plt = _plt()
    if plt is None or gene_cluster_result is None:
        return (None, None)

    embed = gene_cluster_result["embedding"]
    labels = np.asarray(gene_cluster_result["cluster"])
    n = len(embed)

    # Lookup host metadata onto each gene
    host = host_metadata_df.copy() if host_metadata_df is not None else None
    join_cols = ["median_gc3", "grodon2_doubling_time_h"]
    cat_col = None
    if host is not None:
        for c in ("phylum", "class", "habitat"):
            if c in host.columns:
                cat_col = c
                break

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(13, 10), constrained_layout=True,
    )
    axes_flat = axes.flatten()

    # Adaptive scatter style: small markers + alpha for large corpora
    s = 2 if n > 50000 else (4 if n > 5000 else 8)
    alpha = 0.25 if n > 50000 else (0.5 if n > 5000 else 0.7)

    cmap_cat = plt.get_cmap("tab20")

    # Panel A: gene cluster
    ax = axes_flat[0]
    uniq = sorted(set(labels))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        color = "#cccccc" if lab == -1 else cmap_cat(i % 20)
        ax.scatter(
            embed[mask, 0], embed[mask, 1],
            s=s, alpha=alpha, color=color, edgecolor="none",
            label=f"c{lab} (n={int(mask.sum())})" if lab != -1 else f"noise (n={int(mask.sum())})",
        )
    handles, lab_strs = ax.get_legend_handles_labels()
    if len(handles) <= 12:
        ax.legend(loc="best", fontsize=6, frameon=False)
    else:
        ax.text(
            0.99, 0.01, f"{len(uniq)} clusters",
            ha="right", va="bottom", fontsize=7, transform=ax.transAxes,
        )
    ax.set_title("A. Gene cluster", fontsize=11)

    # Panel B: host genome (top-N)
    ax = axes_flat[1]
    if "sample_id" in gene_corpus_df.columns:
        sample_ids = gene_corpus_df["sample_id"].values
        top_hosts = pd.Series(sample_ids).value_counts().head(20).index.tolist()
        is_top = np.isin(sample_ids, top_hosts)
        ax.scatter(
            embed[~is_top, 0], embed[~is_top, 1],
            s=s, alpha=alpha * 0.5, color="#dddddd", edgecolor="none",
            label=f"other hosts (n={int((~is_top).sum())})",
        )
        for i, host_id in enumerate(top_hosts):
            mask = sample_ids == host_id
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                s=s, alpha=alpha, color=cmap_cat(i % 20), edgecolor="none",
                label=f"{host_id}" if i < 10 else None,
            )
        ax.text(
            0.99, 0.01, f"top 20 of {pd.Series(sample_ids).nunique()} hosts shown",
            ha="right", va="bottom", fontsize=7, transform=ax.transAxes,
        )
    ax.set_title("B. Host genome", fontsize=11)

    # Panel C: host categorical metadata (e.g. phylum)
    ax = axes_flat[2]
    if host is not None and cat_col is not None:
        host_lookup = dict(zip(host["sample_id"], host[cat_col]))
        cats = np.array([host_lookup.get(s, None) for s in gene_corpus_df["sample_id"]])
        uniq = sorted({c for c in cats if c is not None and not (
            isinstance(c, float) and np.isnan(c)
        )})
        for i, lab in enumerate(uniq):
            mask = cats == lab
            ax.scatter(
                embed[mask, 0], embed[mask, 1],
                s=s, alpha=alpha, color=cmap_cat(i % 20), edgecolor="none",
                label=f"{lab} (n={int(mask.sum())})",
            )
        if len(uniq) <= 12:
            ax.legend(loc="best", fontsize=6, frameon=False)
        ax.set_title(f"C. Host {cat_col}", fontsize=11)
    else:
        ax.text(0.5, 0.5, "no categorical metadata available",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("C. (host metadata unavailable)", fontsize=11)

    # Panel D: host median GC3 (continuous)
    ax = axes_flat[3]
    if host is not None and "median_gc3" in host.columns:
        host_lookup = dict(zip(host["sample_id"], host["median_gc3"]))
        gc_vals = np.array([host_lookup.get(s, np.nan) for s in gene_corpus_df["sample_id"]],
                           dtype=float)
        finite = np.isfinite(gc_vals)
        if finite.any():
            if (~finite).any():
                ax.scatter(
                    embed[~finite, 0], embed[~finite, 1],
                    s=s, alpha=alpha * 0.4, color="#dddddd", edgecolor="none",
                )
            vmin, vmax = np.nanpercentile(gc_vals[finite], [5, 95])
            sc = ax.scatter(
                embed[finite, 0], embed[finite, 1],
                s=s, alpha=alpha, c=gc_vals[finite], cmap="viridis",
                vmin=vmin, vmax=vmax, edgecolor="none",
            )
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="host median_gc3")
        ax.set_title("D. Host median GC3", fontsize=11)
    else:
        ax.text(0.5, 0.5, "host GC3 unavailable",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("D. (host GC3 unavailable)", fontsize=11)

    for ax in axes_flat:
        ax.set_xlabel(f"{gene_cluster_result['method_dim'].upper()} dim 1", fontsize=8)
        ax.set_ylabel(f"{gene_cluster_result['method_dim'].upper()} dim 2", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Gene-level cross-genome embedding "
        f"({gene_cluster_result['method_dim'].upper()} + "
        f"{gene_cluster_result['method_cluster'].upper()}) "
        f"— n_genes = {n:,}",
        fontsize=12, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / "corpus_gene_level_umap.png"
    svg = out_dir / "corpus_gene_level_umap.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


def render_category_diversity(
    output_dir: Path,
    by_cat_df: pd.DataFrame,
    *,
    top_n: int = 30,
) -> tuple[Path | None, Path | None]:
    """Per-category sub-cluster diversity bar chart.

    For each KO / COG category the user grouped by, shows how many
    sub-clusters its genes split into. Categories at the top of the bar
    chart are the most "diverse" — same functional class, multiple codon
    strategies. Annotates the absolute gene count next to each bar.
    """
    plt = _plt()
    if plt is None or by_cat_df is None or by_cat_df.empty:
        return (None, None)

    # Build summary if not already provided
    summary = (
        by_cat_df.groupby("category", as_index=False)
        .agg(
            n_genes=("n_in_category", "first"),
            n_subclusters=("sub_cluster", lambda v: int(len(set(c for c in v if c != -1)))),
            n_noise=("sub_cluster", lambda v: int((v == -1).sum())),
        )
    )
    summary["splits"] = summary["n_subclusters"] >= 2
    # Prefer to show categories that *split* first; fall back to high n_genes.
    summary = summary.sort_values(
        ["splits", "n_subclusters", "n_genes"], ascending=[False, False, False],
    ).head(top_n)

    if summary.empty:
        return (None, None)

    n = len(summary)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.32 + 1.5)),
                           constrained_layout=True)
    ypos = np.arange(n)[::-1]
    cmap = plt.get_cmap("plasma")
    norm_max = max(int(summary["n_subclusters"].max()), 2)
    for y, (_, r) in zip(ypos, summary.iterrows()):
        c = cmap(r["n_subclusters"] / norm_max)
        ax.barh(y, r["n_subclusters"], color=c,
                edgecolor="black", linewidth=0.4, height=0.7)
        ax.text(
            r["n_subclusters"] + 0.05, y,
            f"  n={int(r['n_genes'])}"
            + (f", {int(r['n_noise'])} noise" if r["n_noise"] else ""),
            va="center", fontsize=7,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(summary["category"], fontsize=7)
    ax.set_xlabel("Number of within-category sub-clusters")
    ax.set_title(
        "Functional-category codon-strategy diversity\n"
        "(more sub-clusters = same KO/COG, multiple codon strategies)",
        fontsize=11, fontweight="bold",
    )
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    out_dir = Path(output_dir)
    cat_col = by_cat_df["category_col"].iloc[0] if "category_col" in by_cat_df.columns else "category"
    png = out_dir / f"corpus_category_diversity_{cat_col}.png"
    svg = out_dir / f"corpus_category_diversity_{cat_col}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Single-genome locator
# ──────────────────────────────────────────────────────────────────────────────


def render_focus_genome_locator(
    output_dir: Path,
    focus_id: str,
    embedding: np.ndarray,
    sample_ids: list[str],
    cluster_labels: np.ndarray,
    corpus_df: pd.DataFrame,
    *,
    scalar_cols: tuple[str, ...] = (
        "median_cai", "median_melp", "median_fop", "median_enc",
        "median_gc3", "frac_in_optimized_set", "grodon2_doubling_time_h",
        "aitchison_genome_to_mahal", "aitchison_genome_to_rp",
    ),
    method_dim: str = "PCA",
) -> tuple[Path | None, Path | None]:
    """Two-panel locator: highlighted UMAP + per-scalar percentile dots on violin."""
    plt = _plt()
    if plt is None:
        return (None, None)
    if focus_id not in sample_ids:
        logger.warning("Focus genome '%s' not in corpus; skipping locator figure", focus_id)
        return (None, None)

    focus_idx = sample_ids.index(focus_id)
    focus_cluster = int(cluster_labels[focus_idx])

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # Left: UMAP with focus genome highlighted
    cmap = plt.get_cmap("tab20")
    uniq = sorted(set(cluster_labels))
    for i, lab in enumerate(uniq):
        mask = cluster_labels == lab
        color = "#cccccc" if lab == -1 else cmap((i % 20))
        axL.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=10, alpha=0.55, color=color, edgecolor="none",
            label=f"c{lab} (n={int(mask.sum())})" if lab != -1 else f"noise (n={int(mask.sum())})",
        )
    # Highlight focus genome
    axL.scatter(
        [embedding[focus_idx, 0]], [embedding[focus_idx, 1]],
        s=180, marker="*", color="red", edgecolor="black", linewidth=1.0,
        zorder=10, label=f"{focus_id} (c{focus_cluster})",
    )
    axL.set_xlabel(f"{method_dim} dim 1")
    axL.set_ylabel(f"{method_dim} dim 2")
    axL.set_title(f"A. {focus_id} on the corpus embedding", fontsize=11)
    axL.legend(loc="best", fontsize=6, frameon=False)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # Right: violin + focus genome dot for each scalar
    available = [c for c in scalar_cols if c in corpus_df.columns
                 and corpus_df[c].notna().sum() >= 10]
    if not available:
        axR.text(0.5, 0.5, "No scalar metrics available",
                 ha="center", va="center", transform=axR.transAxes)
    else:
        # Z-score per column so violins share an axis cleanly
        focus_row = corpus_df.iloc[focus_idx]
        ypos = np.arange(len(available))
        violin_data = []
        focus_z = []
        focus_pct = []
        for col in available:
            v = corpus_df[col].dropna().astype(float).values
            if v.std() == 0:
                violin_data.append(np.zeros_like(v))
                focus_z.append(0.0)
            else:
                z = (v - v.mean()) / v.std()
                violin_data.append(z)
                fv = focus_row.get(col, np.nan)
                focus_z.append(
                    float((fv - v.mean()) / v.std()) if pd.notna(fv) else np.nan
                )
            fv2 = focus_row.get(col, np.nan)
            if pd.notna(fv2):
                focus_pct.append(float((v < fv2).mean() * 100))
            else:
                focus_pct.append(np.nan)

        parts = axR.violinplot(violin_data, positions=ypos, orientation="horizontal",
                               showextrema=False, showmedians=False, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#cccccc")
            pc.set_edgecolor("black")
            pc.set_linewidth(0.4)
            pc.set_alpha(0.6)
        # Plot focus dot at its z-score
        valid = np.isfinite(focus_z)
        axR.scatter(
            np.array(focus_z)[valid], ypos[valid],
            s=110, marker="*", color="red", edgecolor="black", linewidth=0.8, zorder=10,
        )
        # Annotate percentile rank
        for y, pct in zip(ypos, focus_pct):
            if not np.isnan(pct):
                axR.text(
                    axR.get_xlim()[1] - 0.05, y,
                    f"  {pct:.0f}th pctile",
                    fontsize=7, va="center", ha="right",
                )
        axR.set_yticks(ypos)
        axR.set_yticklabels(available, fontsize=8)
        axR.axvline(0, color="black", linewidth=0.4, linestyle=":")
        axR.set_xlabel("z-score within corpus (★ = focus genome)")
        axR.set_title("B. Per-scalar position within corpus", fontsize=11)
        for s in ("top", "right"):
            axR.spines[s].set_visible(False)

    fig.suptitle(
        f"Genome locator — {focus_id}",
        fontsize=12, fontweight="bold",
    )

    out_dir = Path(output_dir)
    safe_name = focus_id.replace("/", "_").replace(" ", "_")
    png = out_dir / f"corpus_focus_{safe_name}.png"
    svg = out_dir / f"corpus_focus_{safe_name}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)
