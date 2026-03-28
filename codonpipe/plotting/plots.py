"""Publication-ready plotting module for codon usage analysis.

Generates figures at 300 DPI in PNG and optionally EPS format.
All plots use a consistent style suitable for journal submission.
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
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from codonpipe.utils.codon_tables import AMINO_ACID_FAMILIES, RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# Publication style defaults
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
}

DPI = 300
FORMATS = ["png", "eps"]


def _apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_PARAMS)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.3})


def _save_fig(fig: plt.Figure, path: Path, formats: list[str] | None = None):
    """Save figure in multiple formats."""
    formats = formats or FORMATS
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=DPI, bbox_inches="tight")
        logger.debug("Saved figure: %s", out)
    plt.close(fig)


# ─── Single-genome plots ────────────────────────────────────────────────────


def plot_codon_frequency_bar(freq_df: pd.DataFrame, output_path: Path, sample_id: str = ""):
    """Bar chart of codon usage frequency (per thousand codons).

    Args:
        freq_df: Output from compute_codon_frequency_table().
        output_path: Base path for saving (extension added automatically).
        sample_id: Sample name for title.
    """
    _apply_style()
    df = freq_df[freq_df["amino_acid"] != "*"].copy()

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = []
    aa_colors = sns.color_palette("husl", df["amino_acid"].nunique())
    aa_list = df["amino_acid"].unique()
    aa_cmap = dict(zip(aa_list, aa_colors))
    for _, row in df.iterrows():
        colors.append(aa_cmap[row["amino_acid"]])

    ax.bar(range(len(df)), df["per_thousand"], color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["codon"], rotation=90, fontsize=7)
    ax.set_ylabel("Frequency (per thousand codons)")
    ax.set_xlabel("Codon")
    if sample_id:
        ax.set_title(f"Codon Usage Frequency — {sample_id}")
    ax.axhline(y=1000 / 61, color="gray", linestyle="--", alpha=0.5, label="Expected (uniform)")

    # Add amino acid color legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in aa_cmap.values()]
    ax.legend(handles, aa_cmap.keys(), loc="upper right", ncol=4, fontsize=6, framealpha=0.8)

    _save_fig(fig, output_path)


def plot_rscu_bar(rscu_values: dict[str, float], output_path: Path, sample_id: str = "", title_suffix: str = ""):
    """Bar chart of RSCU values per codon.

    Args:
        rscu_values: Dict mapping RSCU column names to values.
        output_path: Base path for saving.
        sample_id: Sample name for title.
        title_suffix: Additional title text (e.g., "Ribosomal Proteins").
    """
    _apply_style()
    cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_values]
    vals = [rscu_values[c] for c in cols]

    fig, ax = plt.subplots(figsize=(16, 5))

    # Color by amino acid family
    aa_names = [c.split("-")[0].rstrip("0123456789") for c in cols]
    unique_aa = list(dict.fromkeys(aa_names))
    palette = sns.color_palette("husl", len(unique_aa))
    aa_cmap = dict(zip(unique_aa, palette))
    colors = [aa_cmap[aa] for aa in aa_names]

    ax.bar(range(len(cols)), vals, color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.split("-")[-1] for c in cols], rotation=90, fontsize=7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="RSCU = 1 (no bias)")
    ax.set_ylabel("RSCU")
    ax.set_xlabel("Codon")

    title = "Relative Synonymous Codon Usage"
    if sample_id:
        title += f" — {sample_id}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)

    _save_fig(fig, output_path)


def plot_enc_gc3(enc_df: pd.DataFrame, output_path: Path, sample_id: str = ""):
    """ENC vs GC3 plot (Nc plot) with the Wright (1990) expected curve.

    Shows how actual codon usage bias compares to the null expectation
    based solely on GC content at 3rd codon positions.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    # Expected ENC curve under mutational pressure alone
    gc3_range = np.linspace(0, 1, 200)
    enc_expected = 2 + gc3_range + (29 / (gc3_range**2 + (1 - gc3_range)**2))
    ax.plot(gc3_range, enc_expected, "k-", linewidth=1.5, label="Expected (Wright 1990)", zorder=1)

    ax.scatter(enc_df["GC3"], enc_df["ENC"], s=8, alpha=0.4, c="steelblue", edgecolors="none", zorder=2)
    ax.set_xlabel("GC3 (GC content at 3rd codon position)")
    ax.set_ylabel("Effective Number of Codons (ENC)")
    ax.set_xlim(0, 1)
    ax.set_ylim(15, 65)
    ax.legend(loc="upper right")
    if sample_id:
        ax.set_title(f"ENC–GC3 Plot — {sample_id}")

    _save_fig(fig, output_path)


def plot_encprime_gc3(
    encprime_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """ENCprime vs GC3 plot, overlaid with the Wright (1990) expected curve.

    ENCprime corrects for background GC composition, so points should cluster
    more tightly around the expected curve than raw ENC values.

    Args:
        encprime_df: DataFrame with ENCprime scores (from coRdon). Must have a score column.
        enc_df: DataFrame with GC3 column (from native ENC calculation).
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()

    # Identify the ENCprime score column (first column that isn't gene/width)
    score_col = [c for c in encprime_df.columns if c not in ("gene", "width")][0]

    # Merge ENCprime with GC3 from enc_df
    merged = encprime_df[["gene", score_col]].merge(
        enc_df[["gene", "GC3"]], on="gene", how="inner"
    )
    if len(merged) < 3:
        logger.info("Too few genes for ENCprime plot in %s, skipping", sample_id)
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Expected ENC curve under mutational pressure alone
    gc3_range = np.linspace(0, 1, 200)
    enc_expected = 2 + gc3_range + (29 / (gc3_range**2 + (1 - gc3_range)**2))
    ax.plot(gc3_range, enc_expected, "k-", linewidth=1.5, label="Expected (Wright 1990)", zorder=1)

    ax.scatter(merged["GC3"], merged[score_col], s=8, alpha=0.4,
               c="coral", edgecolors="none", zorder=2)
    ax.set_xlabel("GC3 (GC content at 3rd codon position)")
    ax.set_ylabel("ENC' (GC-corrected)")
    ax.set_xlim(0, 1)
    ax.set_ylim(15, 65)
    ax.legend(loc="upper right")
    if sample_id:
        ax.set_title(f"ENC' (Novembre 2002) vs GC3 — {sample_id}")

    _save_fig(fig, output_path)


def plot_milc_distribution(milc_df: pd.DataFrame, output_path: Path, sample_id: str = ""):
    """Histogram of per-gene MILC values.

    MILC (Measure Independent of Length and Composition) values near 0 indicate
    codon usage consistent with background composition; higher values indicate
    stronger bias toward specific codons.

    Args:
        milc_df: DataFrame with MILC scores from coRdon.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()

    score_col = [c for c in milc_df.columns if c not in ("gene", "width")][0]
    vals = milc_df[score_col].dropna()
    if vals.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=80, color="mediumseagreen", alpha=0.7, edgecolor="white", linewidth=0.3)

    median_val = np.median(vals)
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1,
               label=f"Median ({median_val:.3f})")
    ax.set_xlabel("MILC Score")
    ax.set_ylabel("Number of Genes")
    ax.legend(fontsize=9)
    if sample_id:
        ax.set_title(f"MILC Distribution — {sample_id}")

    _save_fig(fig, output_path)


def plot_expression_distribution(expr_df: pd.DataFrame, output_path: Path, sample_id: str = ""):
    """Distribution plot of expression scores (MELP, CAI, Fop) with percentile thresholds."""
    _apply_style()
    metrics = [m for m in ["MELP", "CAI", "Fop"] if m in expr_df.columns]
    if not metrics:
        return

    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {"MELP": "steelblue", "CAI": "steelblue", "Fop": "darkorange"}

    for ax, metric in zip(axes, metrics):
        vals = expr_df[metric].dropna()
        if vals.empty:
            ax.set_visible(False)
            continue

        ax.hist(vals, bins=80, color=colors.get(metric, "steelblue"),
                alpha=0.7, edgecolor="white", linewidth=0.3)
        p5, p95 = np.percentile(vals, [5, 95])
        ax.axvline(p5, color="red", linestyle="--", linewidth=1, label=f"5th pctl ({p5:.3f})")
        ax.axvline(p95, color="darkgreen", linestyle="--", linewidth=1, label=f"95th pctl ({p95:.3f})")
        ax.set_xlabel(f"{metric} Score")
        ax.set_ylabel("Number of Genes")
        ax.set_title(f"{metric} Distribution")
        ax.legend(fontsize=8)

    if sample_id:
        fig.suptitle(f"Expression Level Distributions — {sample_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rscu_heatmap_single(rscu_df: pd.DataFrame, output_path: Path, sample_id: str = ""):
    """Heatmap of per-gene RSCU values for a single genome.

    Rows = genes (clustered), Columns = codons.
    """
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_df.columns]
    if not rscu_cols or len(rscu_df) < 3:
        logger.info("Too few genes for heatmap in %s, skipping", sample_id)
        return

    data = rscu_df[rscu_cols].values
    # Cap at 500 genes for readability
    if len(data) > 500:
        idx = np.random.choice(len(data), 500, replace=False)
        data = data[idx]

    fig = plt.figure(figsize=(14, min(10, max(5, len(data) // 30))))
    g = sns.clustermap(
        pd.DataFrame(data, columns=[c.split("-")[-1] for c in rscu_cols]),
        cmap="RdYlBu_r",
        center=1.0,
        row_cluster=True,
        col_cluster=True,
        method="average",
        metric="euclidean",
        linewidths=0,
        xticklabels=True,
        yticklabels=False,
        figsize=(14, min(10, max(5, len(data) // 30))),
        cbar_kws={"label": "RSCU"},
    )
    if sample_id:
        g.fig.suptitle(f"Per-Gene RSCU Heatmap — {sample_id}", y=1.02)
    _save_fig(g.fig, output_path)


# ─── Batch-mode plots ───────────────────────────────────────────────────────


def plot_pca(
    df: pd.DataFrame,
    rscu_cols: list[str],
    color_col: str | None,
    output_path: Path,
    title: str = "PCA of RSCU Values",
):
    """PCA scatter plot of genome-level RSCU values, optionally colored by metadata."""
    _apply_style()
    data = df[rscu_cols].dropna()
    if len(data) < 3:
        logger.warning("Too few samples (%d) for PCA", len(data))
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    if color_col and color_col in df.columns:
        groups = df.loc[data.index, color_col]
        unique_groups = sorted(groups.dropna().unique())
        palette = sns.color_palette("husl", len(unique_groups))
        cmap = dict(zip(unique_groups, palette))
        for grp in unique_groups:
            mask = groups == grp
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=15, alpha=0.6, color=cmap[grp], label=grp, edgecolors="none",
            )
        ax.legend(fontsize=7, markerscale=1.5, loc="best", framealpha=0.8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.5, c="steelblue", edgecolors="none")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    _save_fig(fig, output_path)


def plot_umap(
    df: pd.DataFrame,
    rscu_cols: list[str],
    color_col: str | None,
    output_path: Path,
    title: str = "UMAP of RSCU Values",
    n_neighbors: int = 15,
):
    """UMAP scatter plot of genome-level RSCU values."""
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plot")
        return

    _apply_style()
    data = df[rscu_cols].dropna()
    if len(data) < 10:
        logger.warning("Too few samples (%d) for UMAP", len(data))
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    coords = reducer.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(8, 7))
    if color_col and color_col in df.columns:
        groups = df.loc[data.index, color_col]
        unique_groups = sorted(groups.dropna().unique())
        palette = sns.color_palette("husl", len(unique_groups))
        cmap = dict(zip(unique_groups, palette))
        for grp in unique_groups:
            mask = groups == grp
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                s=15, alpha=0.6, color=cmap[grp], label=grp, edgecolors="none",
            )
        if len(unique_groups) <= 20:
            ax.legend(fontsize=7, markerscale=1.5, loc="best", framealpha=0.8)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=15, alpha=0.5, c="steelblue", edgecolors="none")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    _save_fig(fig, output_path)


def plot_rscu_boxplots_by_group(
    df: pd.DataFrame,
    group_col: str,
    amino_acid: str,
    codon_cols: list[str],
    output_path: Path,
):
    """Boxplots of RSCU values per codon, faceted by group.

    One panel per codon in the amino acid family.
    """
    _apply_style()
    present_cols = [c for c in codon_cols if c in df.columns]
    if not present_cols:
        return

    n_codons = len(present_cols)
    ncols = min(3, n_codons)
    nrows = (n_codons + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, col in enumerate(present_cols):
        row, c = divmod(idx, ncols)
        ax = axes[row][c]
        plot_data = df[[group_col, col]].dropna()
        if plot_data.empty:
            ax.set_visible(False)
            continue

        order = sorted(plot_data[group_col].unique())
        sns.boxplot(
            data=plot_data, x=group_col, y=col, order=order,
            ax=ax, fliersize=1, linewidth=0.8,
            palette="Set2",
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("RSCU")
        codon_label = col.split("-")[-1]
        ax.set_title(f"{amino_acid} — {codon_label}")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes
    for idx in range(n_codons, nrows * ncols):
        row, c = divmod(idx, ncols)
        axes[row][c].set_visible(False)

    fig.suptitle(f"{amino_acid} Codon Usage by {group_col}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_heatmap_clustered(
    df: pd.DataFrame,
    rscu_cols: list[str],
    output_path: Path,
    color_col: str | None = None,
    title: str = "Clustered RSCU Heatmap",
):
    """Hierarchical clustering heatmap of genome-level RSCU values."""
    _apply_style()
    data = df[rscu_cols].dropna()
    if len(data) < 3:
        return

    # Subsample if too large
    if len(data) > 2000:
        data = data.sample(2000, random_state=42)

    col_labels = [c.split("-")[-1] for c in rscu_cols]

    row_colors = None
    if color_col and color_col in df.columns:
        groups = df.loc[data.index, color_col].fillna("Unknown")
        unique_groups = sorted(groups.unique())
        palette = sns.color_palette("husl", len(unique_groups))
        cmap = dict(zip(unique_groups, palette))
        row_colors = groups.map(cmap)

    g = sns.clustermap(
        data,
        cmap="RdYlGn_r",
        center=1.0,
        method="complete",
        metric="manhattan",
        row_cluster=True,
        col_cluster=True,
        xticklabels=col_labels,
        yticklabels=False,
        row_colors=row_colors,
        figsize=(14, min(12, max(6, len(data) // 100))),
        linewidths=0,
        cbar_kws={"label": "RSCU"},
    )
    g.fig.suptitle(title, y=1.02)
    _save_fig(g.fig, output_path)


def plot_significance_heatmap(
    wilcoxon_df: pd.DataFrame,
    output_path: Path,
    title: str = "Pairwise Significance",
):
    """Heatmap of corrected p-values from pairwise Wilcoxon tests."""
    _apply_style()
    if wilcoxon_df.empty:
        return

    # Pivot for heatmap: rows=comparisons, cols=codons
    wilcoxon_df["comparison"] = wilcoxon_df["group1"] + " vs " + wilcoxon_df["group2"]
    pivot = wilcoxon_df.pivot_table(
        index="comparison", columns="codon", values="corrected_p_value", aggfunc="first"
    )
    if pivot.empty:
        return

    # -log10 transform
    log_pivot = -np.log10(pivot.clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.6), max(4, len(pivot) * 0.4)))
    sns.heatmap(
        log_pivot, cmap="viridis", ax=ax,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "-log10(corrected p-value)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Codon")
    ax.set_ylabel("Comparison")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(fontsize=8)
    _save_fig(fig, output_path)


def plot_enrichment_bar(
    enrichment_df: pd.DataFrame,
    output_path: Path,
    title: str = "Pathway Enrichment",
    max_pathways: int = 25,
):
    """Horizontal bar chart of enriched pathways (-log10 FDR).

    Shows the top significant pathways ranked by FDR, with bar color indicating
    fold enrichment and bar length showing -log10(FDR).

    Args:
        enrichment_df: Output from hypergeometric_enrichment().
        output_path: Base path for saving.
        title: Plot title.
        max_pathways: Maximum number of pathways to display.
    """
    _apply_style()
    if enrichment_df.empty or "fdr" not in enrichment_df.columns:
        return

    # Filter to significant or top results
    sig = enrichment_df[enrichment_df["significant"] == True].copy()
    if sig.empty:
        # Show top results anyway, even if none pass FDR
        sig = enrichment_df.head(max_pathways).copy()
    else:
        sig = sig.head(max_pathways)

    if sig.empty:
        return

    sig["neg_log_fdr"] = -np.log10(sig["fdr"].clip(lower=1e-300))
    sig["label"] = sig.apply(
        lambda r: r["pathway_name"][:50] if r["pathway_name"] else r["pathway"],
        axis=1,
    )
    sig = sig.sort_values("neg_log_fdr")

    fig, ax = plt.subplots(figsize=(10, max(4, len(sig) * 0.35)))

    # Color by fold enrichment
    fe = sig["fold_enrichment"].values
    norm = plt.Normalize(vmin=max(fe.min(), 0), vmax=fe.max())
    colors = plt.cm.YlOrRd(norm(fe))

    bars = ax.barh(range(len(sig)), sig["neg_log_fdr"].values, color=colors, edgecolor="none")

    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["label"].values, fontsize=8)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title)

    # Add gene count labels on bars
    for i, (_, row) in enumerate(sig.iterrows()):
        ax.text(
            row["neg_log_fdr"] + 0.05, i, f"k={int(row['k'])}",
            va="center", fontsize=7, color="gray",
        )

    # FDR threshold line
    fdr_line = -np.log10(0.05)
    ax.axvline(fdr_line, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label="FDR = 0.05")
    ax.legend(fontsize=8)

    # Colorbar for fold enrichment
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Fold enrichment", fontsize=9)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_expression_tier_summary(
    expr_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Stacked bar chart showing gene counts per expression tier across metrics.

    One group of bars per metric (MELP, CAI, Fop), with three segments (high,
    medium, low) colored consistently.

    Args:
        expr_df: Expression table with *_class columns.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    class_cols = [c for c in expr_df.columns if c.endswith("_class") and c != "expression_class"]
    if not class_cols:
        return

    tier_colors = {"high": "#d62728", "medium": "#7f7f7f", "low": "#1f77b4", "unknown": "#bcbd22"}
    metrics = [c.replace("_class", "") for c in class_cols]

    counts = {}
    for col, metric in zip(class_cols, metrics):
        vc = expr_df[col].value_counts()
        counts[metric] = {tier: vc.get(tier, 0) for tier in ["high", "medium", "low"]}

    fig, ax = plt.subplots(figsize=(max(4, len(metrics) * 1.5), 5))
    x = np.arange(len(metrics))
    width = 0.6

    bottom = np.zeros(len(metrics))
    for tier in ["low", "medium", "high"]:
        vals = [counts[m].get(tier, 0) for m in metrics]
        ax.bar(x, vals, width, bottom=bottom, label=tier.capitalize(),
               color=tier_colors[tier], edgecolor="white", linewidth=0.3)
        # Add counts in the center of each segment
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(x[i], bottom[i] + v / 2, str(v), ha="center", va="center",
                        fontsize=8, color="white" if tier != "medium" else "black")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Number of Genes")
    ax.legend(loc="upper right")
    if sample_id:
        ax.set_title(f"Expression Classification by Metric — {sample_id}")

    fig.tight_layout()
    _save_fig(fig, output_path)


# ─── Advanced analysis plots ─────────────────────────────────────────────────


def plot_coa(
    coa_coords: pd.DataFrame,
    coa_inertia: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
    color_col: str | None = None,
):
    """COA scatter plot: genes on Axis 1 vs Axis 2, optionally colored by expression tier.

    Args:
        coa_coords: Gene coordinates from COA (Axis1, Axis2, optionally *_class cols).
        coa_inertia: Inertia summary (axis, pct_inertia).
        output_path: Base path for saving.
        sample_id: Sample name for title.
        color_col: Column to color by (e.g. 'CAI_class').
    """
    _apply_style()
    if len(coa_coords) < 3 or "Axis1" not in coa_coords.columns:
        return

    pct1 = coa_inertia.loc[coa_inertia["axis"] == 1, "pct_inertia"].values[0] if len(coa_inertia) > 0 else 0
    pct2 = coa_inertia.loc[coa_inertia["axis"] == 2, "pct_inertia"].values[0] if len(coa_inertia) > 1 else 0

    fig, ax = plt.subplots(figsize=(8, 7))

    if color_col and color_col in coa_coords.columns:
        tier_colors = {"high": "#d62728", "medium": "#aaaaaa", "low": "#1f77b4", "unknown": "#bcbd22"}
        for tier in ["medium", "low", "high"]:
            mask = coa_coords[color_col] == tier
            if mask.any():
                ax.scatter(
                    coa_coords.loc[mask, "Axis1"], coa_coords.loc[mask, "Axis2"],
                    s=6, alpha=0.5, c=tier_colors.get(tier, "gray"),
                    label=tier.capitalize(), edgecolors="none", zorder=2 if tier == "medium" else 3,
                )
        ax.legend(fontsize=8, markerscale=2, loc="best", framealpha=0.8)
    else:
        ax.scatter(coa_coords["Axis1"], coa_coords["Axis2"],
                   s=6, alpha=0.4, c="steelblue", edgecolors="none")

    ax.set_xlabel(f"COA Axis 1 ({pct1:.1f}% inertia)")
    ax.set_ylabel(f"COA Axis 2 ({pct2:.1f}% inertia)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    if sample_id:
        ax.set_title(f"Correspondence Analysis on RSCU — {sample_id}")

    _save_fig(fig, output_path)


def plot_coa_codons(
    coa_codon_coords: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """COA biplot showing codon loadings on Axes 1 and 2.

    Args:
        coa_codon_coords: Codon coordinates (codon, Axis1, Axis2).
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    if len(coa_codon_coords) < 3:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by amino acid family
    aa_names = [c.split("-")[0].rstrip("0123456789") for c in coa_codon_coords["codon"]]
    unique_aa = list(dict.fromkeys(aa_names))
    palette = sns.color_palette("husl", len(unique_aa))
    aa_cmap = dict(zip(unique_aa, palette))

    for _, row in coa_codon_coords.iterrows():
        aa = row["codon"].split("-")[0].rstrip("0123456789")
        codon_label = row["codon"].split("-")[-1]
        color = aa_cmap.get(aa, "gray")
        ax.annotate(
            codon_label, (row["Axis1"], row["Axis2"]),
            fontsize=7, color=color, ha="center", va="center", fontweight="bold",
        )

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("COA Axis 1")
    ax.set_ylabel("COA Axis 2")
    if sample_id:
        ax.set_title(f"COA Codon Loadings — {sample_id}")

    # AA legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=6)
               for c in aa_cmap.values()]
    ax.legend(handles, aa_cmap.keys(), loc="upper right", ncol=3, fontsize=6, framealpha=0.8)

    _save_fig(fig, output_path)


def plot_s_value_scatter(
    s_val_df: pd.DataFrame,
    expr_df: pd.DataFrame | None,
    output_path: Path,
    sample_id: str = "",
    metric: str = "CAI",
):
    """Scatter plot of S-value vs expression metric (e.g. CAI).

    Args:
        s_val_df: DataFrame with gene, S_value.
        expr_df: Expression table with gene and score columns.
        output_path: Base path for saving.
        sample_id: Sample name for title.
        metric: Expression score to plot on x-axis.
    """
    _apply_style()
    if s_val_df.empty or expr_df is None or metric not in expr_df.columns:
        return

    merged = s_val_df.merge(expr_df[["gene", metric]], on="gene", how="inner")
    if len(merged) < 10:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged[metric], merged["S_value"], s=6, alpha=0.35, c="steelblue", edgecolors="none")

    # Regression line
    valid = merged.dropna(subset=[metric, "S_value"])
    if len(valid) > 10:
        slope, intercept, r, p_val, _ = stats.linregress(valid[metric], valid["S_value"])
        x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1,
                label=f"r = {r:.3f}, p = {p_val:.2e}")
        ax.legend(fontsize=8)

    ax.set_xlabel(f"{metric} Score")
    ax.set_ylabel("S-value (RSCU distance to ribosomal proteins)")
    if sample_id:
        ax.set_title(f"S-value vs {metric} — {sample_id}")

    _save_fig(fig, output_path)


def plot_enc_diff(
    enc_diff_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
    expr_df: pd.DataFrame | None = None,
):
    """Scatter plot of (ENC - ENC') vs GC3, colored by expression tier.

    Genes where ENC << ENC' are under selection beyond what GC predicts.

    Args:
        enc_diff_df: DataFrame with gene, ENC, ENCprime, ENC_diff, GC3.
        output_path: Base path for saving.
        sample_id: Sample name for title.
        expr_df: Optional expression table for color overlay.
    """
    _apply_style()
    if enc_diff_df.empty or len(enc_diff_df) < 10:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: histogram of ENC - ENC'
    ax = axes[0]
    ax.hist(enc_diff_df["ENC_diff"].dropna(), bins=60, color="mediumpurple",
            alpha=0.7, edgecolor="white", linewidth=0.3)
    median_val = enc_diff_df["ENC_diff"].median()
    ax.axvline(median_val, color="black", linestyle="--", linewidth=1,
               label=f"Median ({median_val:.2f})")
    ax.axvline(0, color="red", linestyle="-", linewidth=0.8, alpha=0.7, label="ENC = ENC'")
    ax.set_xlabel("ENC - ENC'")
    ax.set_ylabel("Number of Genes")
    ax.set_title("Distribution of ENC - ENC'")
    ax.legend(fontsize=8)

    # Panel B: scatter ENC_diff vs GC3
    ax = axes[1]
    plot_df = enc_diff_df.copy()
    if expr_df is not None and "gene" in expr_df.columns and "CAI_class" in expr_df.columns:
        plot_df = plot_df.merge(expr_df[["gene", "CAI_class"]], on="gene", how="left")
        tier_colors = {"high": "#d62728", "medium": "#aaaaaa", "low": "#1f77b4"}
        for tier in ["medium", "low", "high"]:
            mask = plot_df["CAI_class"] == tier
            if mask.any():
                ax.scatter(plot_df.loc[mask, "GC3"], plot_df.loc[mask, "ENC_diff"],
                           s=6, alpha=0.4, c=tier_colors.get(tier, "gray"),
                           label=tier.capitalize(), edgecolors="none",
                           zorder=2 if tier == "medium" else 3)
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(plot_df["GC3"], plot_df["ENC_diff"],
                   s=6, alpha=0.4, c="mediumpurple", edgecolors="none")

    ax.axhline(0, color="red", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("GC3")
    ax.set_ylabel("ENC - ENC'")
    ax.set_title("ENC - ENC' vs GC3")

    if sample_id:
        fig.suptitle(f"ENC vs ENC' Selection Signal — {sample_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_neutrality(
    gc12_gc3_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Neutrality plot: GC12 vs GC3 (Sueoka 1988).

    Slope near 1 = mutational drift dominates.
    Slope near 0 = selection dominates codon usage.

    Args:
        gc12_gc3_df: DataFrame with gene, GC12, GC3.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    if gc12_gc3_df.empty or len(gc12_gc3_df) < 10:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gc12_gc3_df["GC3"], gc12_gc3_df["GC12"],
               s=6, alpha=0.35, c="steelblue", edgecolors="none")

    # Diagonal (perfect neutrality: GC12 = GC3)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5, label="GC12 = GC3 (neutrality)")

    # Regression line
    valid = gc12_gc3_df[["GC3", "GC12"]].dropna()
    if len(valid) > 10:
        slope, intercept, r, p_val, _ = stats.linregress(valid["GC3"], valid["GC12"])
        x_line = np.linspace(valid["GC3"].min(), valid["GC3"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1.2,
                label=f"Regression: slope = {slope:.3f}, r = {r:.3f}")

    ax.set_xlabel("GC3 (3rd codon position)")
    ax.set_ylabel("GC12 (1st + 2nd codon positions)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    if sample_id:
        ax.set_title(f"Neutrality Plot (Sueoka 1988) — {sample_id}")

    _save_fig(fig, output_path)


def plot_pr2(
    pr2_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
    expr_df: pd.DataFrame | None = None,
):
    """PR2 (Parity Rule 2) plot: A3/(A3+T3) vs G3/(G3+C3).

    Center at (0.5, 0.5) = no bias. Deviations reveal strand-specific
    mutational asymmetry and translational selection.

    Args:
        pr2_df: DataFrame with gene, A3_ratio, G3_ratio.
        output_path: Base path for saving.
        sample_id: Sample name for title.
        expr_df: Optional expression table for color overlay.
    """
    _apply_style()
    if pr2_df.empty or len(pr2_df) < 10:
        return

    fig, ax = plt.subplots(figsize=(7, 7))

    plot_df = pr2_df.copy()
    if expr_df is not None and "gene" in expr_df.columns and "CAI_class" in expr_df.columns:
        plot_df = plot_df.merge(expr_df[["gene", "CAI_class"]], on="gene", how="left")
        tier_colors = {"high": "#d62728", "medium": "#aaaaaa", "low": "#1f77b4"}
        for tier in ["medium", "low", "high"]:
            mask = plot_df["CAI_class"] == tier
            if mask.any():
                ax.scatter(plot_df.loc[mask, "A3_ratio"], plot_df.loc[mask, "G3_ratio"],
                           s=6, alpha=0.4, c=tier_colors.get(tier, "gray"),
                           label=tier.capitalize(), edgecolors="none",
                           zorder=2 if tier == "medium" else 3)
        ax.legend(fontsize=8, markerscale=2)
    else:
        ax.scatter(plot_df["A3_ratio"], plot_df["G3_ratio"],
                   s=6, alpha=0.35, c="steelblue", edgecolors="none")

    # Reference crosshairs at (0.5, 0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Mean point
    mean_a3 = plot_df["A3_ratio"].mean()
    mean_g3 = plot_df["G3_ratio"].mean()
    ax.plot(mean_a3, mean_g3, "k+", markersize=12, markeredgewidth=2, zorder=10)

    ax.set_xlabel("A3 / (A3 + T3)")
    ax.set_ylabel("G3 / (G3 + C3)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if sample_id:
        ax.set_title(f"PR2 Bias Plot — {sample_id}")

    _save_fig(fig, output_path)


def plot_delta_rscu_heatmap(
    delta_rscu_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
    metric: str = "CAI",
):
    """Heatmap of delta RSCU (high-expression minus genome average) per codon.

    Positive = favored in highly expressed genes (translationally selected).
    Negative = avoided in highly expressed genes.

    Args:
        delta_rscu_df: DataFrame with codon, amino_acid, delta_rscu.
        output_path: Base path for saving.
        sample_id: Sample name for title.
        metric: Which expression metric was used.
    """
    _apply_style()
    if delta_rscu_df.empty:
        return

    # Sort by amino acid then by delta
    df = delta_rscu_df.sort_values(["amino_acid", "delta_rscu"], ascending=[True, False]).copy()

    fig, ax = plt.subplots(figsize=(16, 4))

    colors = np.where(df["delta_rscu"] > 0, "#d62728", "#1f77b4")
    ax.bar(range(len(df)), df["delta_rscu"].values, color=colors, edgecolor="none", width=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["codon"].values, rotation=90, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"ΔRSCU ({metric} high − genome avg)")
    ax.set_xlabel("Codon")

    # Add amino acid group separators
    aa_changes = df["amino_acid"].ne(df["amino_acid"].shift()).cumsum()
    boundaries = []
    for _, grp in df.groupby(aa_changes):
        boundaries.append(grp.index[0])
    # Draw light separators
    for b in boundaries[1:]:
        idx = df.index.get_loc(b)
        ax.axvline(idx - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    if sample_id:
        ax.set_title(f"ΔRSCU (High-Expression vs Genome Average, {metric}) — {sample_id}")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_trna_codon_correlation(
    trna_corr_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Scatter plot of tRNA gene copy number vs codon RSCU in high-expression genes.

    Strong positive correlation = tRNA pool co-adapted with highly expressed gene codons.

    Args:
        trna_corr_df: DataFrame with codon, tRNA_copy_number, rscu columns.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    if trna_corr_df.empty or "tRNA_copy_number" not in trna_corr_df.columns:
        return

    rscu_col = "rscu_high_expr" if "rscu_high_expr" in trna_corr_df.columns else "rscu_all_genes"
    if rscu_col not in trna_corr_df.columns:
        return

    fig, axes = plt.subplots(1, 2 if "rscu_high_expr" in trna_corr_df.columns else 1,
                             figsize=(12 if "rscu_high_expr" in trna_corr_df.columns else 7, 6))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Plot 1: tRNA copies vs RSCU in high-expression genes
    ax = axes[0]
    df = trna_corr_df.dropna(subset=["tRNA_copy_number", rscu_col])
    ax.scatter(df["tRNA_copy_number"], df[rscu_col],
               s=30, alpha=0.6, c="steelblue", edgecolors="white", linewidth=0.3)

    # Label each point with codon
    for _, row in df.iterrows():
        ax.annotate(row["codon"], (row["tRNA_copy_number"], row[rscu_col]),
                    fontsize=5, alpha=0.7, ha="center", va="bottom")

    # Correlation
    valid = df[df["tRNA_copy_number"] > 0]
    if len(valid) > 5:
        r, p_val = stats.pearsonr(valid["tRNA_copy_number"], valid[rscu_col])
        ax.set_xlabel(f"tRNA Gene Copy Number")
        label_text = "high-expression genes" if rscu_col == "rscu_high_expr" else "all genes"
        ax.set_ylabel(f"RSCU ({label_text})")
        ax.set_title(f"r = {r:.3f}, p = {p_val:.2e}")

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(valid["tRNA_copy_number"], valid[rscu_col])
        x_line = np.linspace(valid["tRNA_copy_number"].min(), valid["tRNA_copy_number"].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1, alpha=0.7)
    else:
        ax.set_xlabel("tRNA Gene Copy Number")
        ax.set_ylabel(f"RSCU")

    # Plot 2: compare high vs low expression (if available)
    if len(axes) > 1 and "rscu_low_expr" in trna_corr_df.columns:
        ax = axes[1]
        df2 = trna_corr_df.dropna(subset=["tRNA_copy_number", "rscu_high_expr", "rscu_low_expr"])
        valid2 = df2[df2["tRNA_copy_number"] > 0]
        if len(valid2) > 5:
            r_hi, p_hi = stats.pearsonr(valid2["tRNA_copy_number"], valid2["rscu_high_expr"])
            r_lo, p_lo = stats.pearsonr(valid2["tRNA_copy_number"], valid2["rscu_low_expr"])
            ax.scatter(valid2["tRNA_copy_number"], valid2["rscu_high_expr"],
                       s=25, alpha=0.6, c="#d62728", label=f"High (r={r_hi:.3f})", edgecolors="none")
            ax.scatter(valid2["tRNA_copy_number"], valid2["rscu_low_expr"],
                       s=25, alpha=0.6, c="#1f77b4", label=f"Low (r={r_lo:.3f})", edgecolors="none")
            ax.legend(fontsize=8)
            ax.set_xlabel("tRNA Gene Copy Number")
            ax.set_ylabel("RSCU")
            ax.set_title("High vs Low Expression Genes")

    if sample_id:
        fig.suptitle(f"tRNA–Codon Co-adaptation — {sample_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_cog_enrichment(
    cog_enrich_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Grouped bar chart of COG category enrichment in expression tiers.

    Args:
        cog_enrich_df: DataFrame from compute_cog_enrichment.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    if cog_enrich_df.empty:
        return

    # Show categories with at least marginal signal (p < 0.1 for either tier)
    show = cog_enrich_df[cog_enrich_df["p_value"] < 0.1].copy()
    if show.empty:
        # Show top 10 by p-value
        show = cog_enrich_df.head(10).copy()
    if show.empty:
        return

    # Make labels
    show["label"] = show.apply(
        lambda r: f"{r['COG_category']} — {r['description'][:40]}" if r["description"] else r["COG_category"],
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(show) * 0.35)))

    # Color by tier
    tier_colors = {"high": "#d62728", "low": "#1f77b4"}
    show["neg_log_p"] = -np.log10(show["p_value"].clip(lower=1e-300))
    show["color"] = show["tier"].map(tier_colors).fillna("gray")

    # Sort by -log10(p)
    show = show.sort_values("neg_log_p")

    bars = ax.barh(range(len(show)), show["neg_log_p"].values,
                   color=show["color"].values, edgecolor="none")
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels(show["label"].values, fontsize=8)
    ax.set_xlabel("-log10(p-value)")

    # Significance thresholds
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="p = 0.05")

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#d62728", label="High-expression"),
                       Patch(facecolor="#1f77b4", label="Low-expression")]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    if sample_id:
        ax.set_title(f"COG Category Enrichment by Expression Tier — {sample_id}")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_gene_length_vs_bias(
    length_bias_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Scatter plots of gene length vs codon bias metrics (ENC, CAI).

    Args:
        length_bias_df: DataFrame with gene, length, ENC, and optionally CAI/MELP/Fop.
        output_path: Base path for saving.
        sample_id: Sample name for title.
    """
    _apply_style()
    if length_bias_df.empty:
        return

    metrics = ["ENC"] + [m for m in ["CAI", "MELP", "Fop"] if m in length_bias_df.columns]
    n_panels = len(metrics)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = {"ENC": "steelblue", "CAI": "coral", "MELP": "mediumseagreen", "Fop": "darkorange"}

    for ax, metric in zip(axes, metrics):
        valid = length_bias_df.dropna(subset=["length", metric])
        if len(valid) < 10:
            ax.set_visible(False)
            continue

        # Convert length to kb
        length_kb = valid["length"] / 1000
        ax.scatter(length_kb, valid[metric], s=5, alpha=0.3,
                   c=colors.get(metric, "gray"), edgecolors="none")

        # LOESS-like smoothing via binning
        n_bins = min(30, len(valid) // 20)
        if n_bins > 3:
            bins = pd.qcut(length_kb, n_bins, duplicates="drop")
            grouped = valid.assign(bin=bins.values).groupby("bin", observed=True)
            bin_centers = grouped["length"].median() / 1000
            bin_medians = grouped[metric].median()
            ax.plot(bin_centers, bin_medians, "k-", linewidth=1.5, alpha=0.8, label="Binned median")

        # Correlation
        r, p_val = stats.pearsonr(length_kb, valid[metric])
        ax.set_title(f"{metric} (r = {r:.3f})")
        ax.set_xlabel("Gene Length (kb)")
        ax.set_ylabel(metric)

    if sample_id:
        fig.suptitle(f"Gene Length vs Codon Bias — {sample_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


# ─── Master plot runner ──────────────────────────────────────────────────────


def generate_single_genome_plots(
    sample_id: str,
    output_dir: Path,
    freq_df: pd.DataFrame | None = None,
    rscu_all: dict[str, float] | None = None,
    rscu_rp: dict[str, float] | None = None,
    rscu_gene_df: pd.DataFrame | None = None,
    enc_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    milc_df: pd.DataFrame | None = None,
    enrichment_results: dict[str, pd.DataFrame] | None = None,
    advanced_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Generate all single-genome plots.

    Returns dict of plot paths.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if freq_df is not None and not freq_df.empty:
        p = plot_dir / f"{sample_id}_codon_frequency"
        plot_codon_frequency_bar(freq_df, p, sample_id)
        outputs["codon_frequency_bar"] = p.with_suffix(".png")

    if rscu_all is not None:
        p = plot_dir / f"{sample_id}_rscu_all"
        plot_rscu_bar(rscu_all, p, sample_id, "All CDS")
        outputs["rscu_bar_all"] = p.with_suffix(".png")

    if rscu_rp is not None:
        p = plot_dir / f"{sample_id}_rscu_ribosomal"
        plot_rscu_bar(rscu_rp, p, sample_id, "Ribosomal Proteins")
        outputs["rscu_bar_rp"] = p.with_suffix(".png")

    if rscu_gene_df is not None and not rscu_gene_df.empty:
        p = plot_dir / f"{sample_id}_rscu_heatmap"
        plot_rscu_heatmap_single(rscu_gene_df, p, sample_id)
        outputs["rscu_heatmap"] = p.with_suffix(".png")

    if enc_df is not None and not enc_df.empty:
        p = plot_dir / f"{sample_id}_enc_gc3"
        plot_enc_gc3(enc_df, p, sample_id)
        outputs["enc_gc3"] = p.with_suffix(".png")

    if encprime_df is not None and not encprime_df.empty and enc_df is not None and not enc_df.empty:
        p = plot_dir / f"{sample_id}_encprime_gc3"
        plot_encprime_gc3(encprime_df, enc_df, p, sample_id)
        outputs["encprime_gc3"] = p.with_suffix(".png")

    if milc_df is not None and not milc_df.empty:
        p = plot_dir / f"{sample_id}_milc_dist"
        plot_milc_distribution(milc_df, p, sample_id)
        outputs["milc_dist"] = p.with_suffix(".png")

    if expr_df is not None and not expr_df.empty:
        p = plot_dir / f"{sample_id}_expression_dist"
        plot_expression_distribution(expr_df, p, sample_id)
        outputs["expression_dist"] = p.with_suffix(".png")

        # Expression tier summary (stacked bars per metric)
        class_cols = [c for c in expr_df.columns if c.endswith("_class") and c != "expression_class"]
        if class_cols:
            p = plot_dir / f"{sample_id}_expression_tiers"
            plot_expression_tier_summary(expr_df, p, sample_id)
            outputs["expression_tiers"] = p.with_suffix(".png")

    # Enrichment bar plots
    if enrichment_results:
        for key, edf in enrichment_results.items():
            if edf.empty:
                continue
            # key is like "enrichment_CAI_high"
            parts = key.replace("enrichment_", "").split("_", 1)
            metric = parts[0] if parts else ""
            tier = parts[1] if len(parts) > 1 else ""
            p = plot_dir / f"{sample_id}_{key}"
            plot_enrichment_bar(edf, p, title=f"{metric} {tier}-expression pathway enrichment — {sample_id}")
            outputs[f"plot_{key}"] = p.with_suffix(".png")

    # ── Advanced analysis plots ──────────────────────────────────────────
    if advanced_results:
        _generate_advanced_plots(
            advanced_results, plot_dir, sample_id, outputs,
            expr_df=expr_df,
        )

    return outputs


def _generate_advanced_plots(
    adv: dict,
    plot_dir: Path,
    sample_id: str,
    outputs: dict[str, Path],
    expr_df: pd.DataFrame | None = None,
):
    """Generate all advanced analysis plots from pre-computed data."""

    # COA on RSCU
    if "coa_coords" in adv and "coa_inertia" in adv:
        coa_coords = adv["coa_coords"]
        coa_inertia = adv["coa_inertia"]

        # Colored by CAI_class if available
        color_col = "CAI_class" if "CAI_class" in coa_coords.columns else None
        p = plot_dir / f"{sample_id}_coa"
        plot_coa(coa_coords, coa_inertia, p, sample_id, color_col=color_col)
        outputs["coa"] = p.with_suffix(".png")

    if "coa_codon_coords" in adv:
        p = plot_dir / f"{sample_id}_coa_codons"
        plot_coa_codons(adv["coa_codon_coords"], p, sample_id)
        outputs["coa_codons"] = p.with_suffix(".png")

    # S-value scatter
    if "s_value" in adv and expr_df is not None:
        for metric in ["CAI", "MELP", "Fop"]:
            if metric in expr_df.columns:
                p = plot_dir / f"{sample_id}_s_value_vs_{metric.lower()}"
                plot_s_value_scatter(adv["s_value"], expr_df, p, sample_id, metric)
                outputs[f"s_value_vs_{metric.lower()}"] = p.with_suffix(".png")
                break  # One S-value plot is enough

    # ENC - ENC' difference
    if "enc_diff" in adv:
        p = plot_dir / f"{sample_id}_enc_diff"
        plot_enc_diff(adv["enc_diff"], p, sample_id, expr_df=expr_df)
        outputs["enc_diff"] = p.with_suffix(".png")

    # Neutrality plot
    if "gc12_gc3" in adv:
        p = plot_dir / f"{sample_id}_neutrality"
        plot_neutrality(adv["gc12_gc3"], p, sample_id)
        outputs["neutrality"] = p.with_suffix(".png")

    # PR2 plot
    if "pr2" in adv:
        p = plot_dir / f"{sample_id}_pr2"
        plot_pr2(adv["pr2"], p, sample_id, expr_df=expr_df)
        outputs["pr2"] = p.with_suffix(".png")

    # Delta RSCU heatmaps
    for metric in ["CAI", "MELP", "Fop"]:
        key = f"delta_rscu_{metric}"
        if key in adv:
            p = plot_dir / f"{sample_id}_delta_rscu_{metric.lower()}"
            plot_delta_rscu_heatmap(adv[key], p, sample_id, metric)
            outputs[f"delta_rscu_{metric.lower()}"] = p.with_suffix(".png")

    # tRNA-codon correlation
    if "trna_codon_correlation" in adv:
        p = plot_dir / f"{sample_id}_trna_codon"
        plot_trna_codon_correlation(adv["trna_codon_correlation"], p, sample_id)
        outputs["trna_codon"] = p.with_suffix(".png")

    # COG enrichment
    if "cog_enrichment" in adv:
        p = plot_dir / f"{sample_id}_cog_enrichment"
        plot_cog_enrichment(adv["cog_enrichment"], p, sample_id)
        outputs["cog_enrichment"] = p.with_suffix(".png")

    # Gene length vs bias
    if "gene_length_bias" in adv:
        p = plot_dir / f"{sample_id}_gene_length_bias"
        plot_gene_length_vs_bias(adv["gene_length_bias"], p, sample_id)
        outputs["gene_length_bias"] = p.with_suffix(".png")


def generate_batch_plots(
    combined_df: pd.DataFrame,
    output_dir: Path,
    metadata_cols: list[str] | None = None,
    wilcoxon_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Generate batch-mode comparative plots.

    Returns dict of plot paths.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in combined_df.columns]

    if not rscu_cols:
        logger.warning("No RSCU columns found in combined data")
        return outputs

    # PCA plots
    p = plot_dir / "pca_all"
    plot_pca(combined_df, rscu_cols, None, p, "PCA of Genome-Level RSCU")
    outputs["pca_all"] = p.with_suffix(".png")

    for col in (metadata_cols or []):
        if col in combined_df.columns and combined_df[col].nunique() <= 30:
            p = plot_dir / f"pca_{col}"
            plot_pca(combined_df, rscu_cols, col, p, f"PCA by {col}")
            outputs[f"pca_{col}"] = p.with_suffix(".png")

    # UMAP plots
    if len(combined_df) >= 10:
        p = plot_dir / "umap_all"
        plot_umap(combined_df, rscu_cols, None, p, "UMAP of Genome-Level RSCU")
        outputs["umap_all"] = p.with_suffix(".png")

        for col in (metadata_cols or []):
            if col in combined_df.columns and combined_df[col].nunique() <= 30:
                p = plot_dir / f"umap_{col}"
                plot_umap(combined_df, rscu_cols, col, p, f"UMAP by {col}")
                outputs[f"umap_{col}"] = p.with_suffix(".png")

    # Clustered heatmap
    for col in (metadata_cols or [None]):
        suffix = f"_{col}" if col else ""
        p = plot_dir / f"heatmap_clustered{suffix}"
        plot_heatmap_clustered(
            combined_df, rscu_cols, p, color_col=col,
            title=f"Clustered RSCU Heatmap" + (f" (colored by {col})" if col else ""),
        )
        outputs[f"heatmap_clustered{suffix}"] = p.with_suffix(".png")

    # Per-amino-acid boxplots by group
    for col in (metadata_cols or []):
        if col not in combined_df.columns or combined_df[col].nunique() > 20:
            continue
        for aa, codons in AMINO_ACID_FAMILIES.items():
            p = plot_dir / f"boxplot_{col}_{aa}"
            plot_rscu_boxplots_by_group(combined_df, col, aa, codons, p)
            outputs[f"boxplot_{col}_{aa}"] = p.with_suffix(".png")

    # Significance heatmaps
    if wilcoxon_results:
        for key, wdf in wilcoxon_results.items():
            p = plot_dir / f"significance_{key}"
            plot_significance_heatmap(wdf, p, title=f"Significance: {key}")
            outputs[f"significance_{key}"] = p.with_suffix(".png")

    return outputs
