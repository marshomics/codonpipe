"""Plotting functions for pre-ranked GSEA results.

Single-genome plots
-------------------
* Horizontal bar plot of significant gene sets (directed enrichment score)
* Running-sum enrichment plot for top hits

Batch / comparative plots
-------------------------
* NES heatmap across samples
* Between-condition NES comparison (grouped bar / forest plot)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from codonpipe.plotting.utils import (
    DPI, FORMATS, apply_style as _apply_style, save_fig as _save_fig,
)

logger = logging.getLogger("codonpipe")

# Consistent condition palette (shared with comparative_plots)
_CONDITION_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

_ENRICHED_COLOR = "#c0392b"    # red — enriched at top of ranked list (divergent CU)
_DEPLETED_COLOR = "#2980b9"    # blue — enriched at bottom (optimized CU)
_NEUTRAL_COLOR = "#95a5a6"


def _condition_colors(conditions: list[str]) -> dict[str, str]:
    unique = sorted(set(conditions))
    return {c: _CONDITION_PALETTE[i % len(_CONDITION_PALETTE)] for i, c in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════════
# Single-genome GSEA bar plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gsea_bar(
    gsea_df: pd.DataFrame,
    output_path: Path,
    title: str = "GSEA Results",
    max_sets: int = 30,
    fdr_threshold: float = 0.25,
    metric: str = "nes",
) -> Path | None:
    """Horizontal bar plot of GSEA results, colored by direction.

    Bars show NES (or -log10(p) * sign(NES) if metric="signed_log_p").
    Only gene sets with FDR < fdr_threshold are shown.

    Args:
        gsea_df: GSEA results from run_preranked_gsea().
        output_path: Base path for output (extension replaced per format).
        title: Plot title.
        max_sets: Maximum number of gene sets to display.
        fdr_threshold: FDR cutoff for inclusion.
        metric: "nes" for normalized enrichment score or "signed_log_p"
            for -log10(p_value) * sign(NES).

    Returns:
        output_path on success, None if no significant results.
    """
    _apply_style()

    if gsea_df.empty:
        return None

    sig = gsea_df[gsea_df["fdr"] < fdr_threshold].copy()
    if sig.empty:
        logger.info("GSEA bar plot: no gene sets with FDR < %.2f", fdr_threshold)
        return None

    # Compute display metric
    if metric == "signed_log_p":
        sig["display_val"] = -np.log10(sig["p_value"].clip(lower=1e-300)) * np.sign(sig["nes"])
        xlabel = r"$-\log_{10}(p) \times \mathrm{sign}(\mathrm{NES})$"
    else:
        sig["display_val"] = sig["nes"]
        xlabel = "Normalized Enrichment Score (NES)"

    # Sort by absolute value, take top N
    sig = sig.reindex(sig["display_val"].abs().sort_values(ascending=False).index)
    sig = sig.head(max_sets)

    # Reverse for horizontal bar (top of plot = highest)
    sig = sig.iloc[::-1]

    colors = [_ENRICHED_COLOR if v > 0 else _DEPLETED_COLOR for v in sig["display_val"]]

    # Dynamic figure height
    n_bars = len(sig)
    fig_height = max(4, n_bars * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(n_bars)
    ax.barh(y_pos, sig["display_val"].values, color=colors, edgecolor="none", height=0.7)

    # Labels: gene set name with FDR annotation
    labels = []
    for _, row in sig.iterrows():
        name = str(row["gene_set"])
        # Truncate long names
        if len(name) > 55:
            name = name[:52] + "..."
        labels.append(name)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")

    # Add FDR annotations to the right margin
    for i, (_, row) in enumerate(sig.iterrows()):
        fdr_val = row["fdr"]
        marker = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*" if fdr_val < 0.05 else ""
        if marker:
            x_pos = row["display_val"]
            offset = 0.05 * abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(
                x_pos + (offset if x_pos >= 0 else -offset),
                i, marker, va="center",
                ha="left" if x_pos >= 0 else "right",
                fontsize=8, color="black",
            )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_ENRICHED_COLOR, label="Enriched among divergent CU genes"),
        Patch(facecolor=_DEPLETED_COLOR, label="Enriched among optimized CU genes"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("GSEA bar plot saved: %d gene sets", n_bars)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Batch: NES heatmap across samples
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gsea_nes_heatmap(
    nes_matrix: pd.DataFrame,
    output_path: Path,
    title: str = "GSEA NES Across Samples",
    max_sets: int = 40,
    condition_map: dict[str, str] | None = None,
) -> Path | None:
    """Heatmap of NES values: rows = gene sets, columns = samples.

    Args:
        nes_matrix: DataFrame from compare_gsea_between_samples().
        output_path: Output path.
        title: Plot title.
        max_sets: Max gene sets to show (ranked by variance across samples).
        condition_map: sample_id → condition label for column color coding.

    Returns:
        output_path on success, None if insufficient data.
    """
    _apply_style()

    if nes_matrix.empty or nes_matrix.shape[1] < 2:
        return None

    # Rank by variance and take top N
    var = nes_matrix.var(axis=1).sort_values(ascending=False)
    top_sets = var.head(max_sets).index
    mat = nes_matrix.loc[top_sets].copy()

    # Truncate long names
    new_idx = []
    for name in mat.index:
        s = str(name)
        new_idx.append(s[:50] + "..." if len(s) > 50 else s)
    mat.index = new_idx

    # Column colors by condition
    col_colors = None
    if condition_map:
        cond_palette = _condition_colors(list(condition_map.values()))
        col_colors = pd.Series(
            {sid: cond_palette.get(condition_map.get(sid, ""), _NEUTRAL_COLOR)
             for sid in mat.columns}
        )

    fig_w = max(8, mat.shape[1] * 0.8 + 4)
    fig_h = max(6, mat.shape[0] * 0.35 + 2)

    g = sns.clustermap(
        mat.fillna(0),
        cmap="RdBu_r",
        center=0,
        figsize=(fig_w, fig_h),
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=True,
        yticklabels=True,
        xticklabels=True,
        linewidths=0.5,
        cbar_kws={"label": "NES", "shrink": 0.6},
    )
    g.fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=45, ha="right")

    # Add condition legend if available
    if condition_map:
        from matplotlib.patches import Patch
        cond_palette = _condition_colors(list(condition_map.values()))
        legend_patches = [
            Patch(facecolor=cond_palette[c], label=c)
            for c in sorted(set(condition_map.values()))
        ]
        g.fig.legend(
            handles=legend_patches, loc="upper left", fontsize=8,
            title="Condition", title_fontsize=9, framealpha=0.9,
            bbox_to_anchor=(0.02, 0.98),
        )

    _save_fig(g.fig, output_path)
    logger.info("GSEA NES heatmap saved: %d sets × %d samples", mat.shape[0], mat.shape[1])
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Batch: Between-condition NES comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_gsea_condition_comparison(
    between_df: pd.DataFrame,
    output_path: Path,
    title: str = "GSEA Between-Condition Comparison",
    max_sets: int = 25,
    fdr_threshold: float = 0.25,
) -> Path | None:
    """Bar plot comparing mean NES between conditions for significant gene sets.

    Shows mean NES per condition as grouped horizontal bars with significance
    markers from the between-condition statistical test.

    Args:
        between_df: Output of compare_gsea_between_conditions().
        output_path: Output path.
        title: Plot title.
        max_sets: Max gene sets to show.
        fdr_threshold: FDR threshold for display.

    Returns:
        output_path on success, None if no data.
    """
    _apply_style()

    if between_df.empty:
        return None

    sig = between_df[between_df["fdr"] < fdr_threshold].copy()
    if sig.empty:
        logger.info("GSEA condition comparison: no significant gene sets")
        return None

    sig = sig.head(max_sets)

    # Identify condition columns (mean_nes_*)
    cond_cols = [c for c in sig.columns if c.startswith("mean_nes_")]
    conditions = [c.replace("mean_nes_", "") for c in cond_cols]

    if len(conditions) < 2:
        return None

    cond_palette = _condition_colors(conditions)

    n_sets = len(sig)
    n_cond = len(conditions)
    fig_height = max(5, n_sets * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    bar_height = 0.8 / n_cond
    y_base = np.arange(n_sets)

    # Reverse order for top-down reading
    sig = sig.iloc[::-1].reset_index(drop=True)

    for j, (cond, col) in enumerate(zip(conditions, cond_cols)):
        y_pos = y_base + j * bar_height - 0.4 + bar_height / 2
        vals = sig[col].fillna(0).values
        ax.barh(
            y_pos, vals, height=bar_height * 0.9,
            color=cond_palette[cond], edgecolor="none", label=cond,
        )

    # Labels
    labels = []
    for _, row in sig.iterrows():
        name = str(row["gene_set"])
        labels.append(name[:50] + "..." if len(name) > 50 else name)

    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean NES")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.5)

    # Significance markers
    for i, (_, row) in enumerate(sig.iterrows()):
        fdr_val = row["fdr"]
        marker = "***" if fdr_val < 0.001 else "**" if fdr_val < 0.01 else "*" if fdr_val < 0.05 else ""
        if marker:
            max_val = max(abs(row[c]) for c in cond_cols if pd.notna(row[c]))
            ax.text(
                max_val + 0.1, i, marker,
                va="center", ha="left", fontsize=8, fontweight="bold",
            )

    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, output_path)
    logger.info("GSEA condition comparison plot saved: %d gene sets", n_sets)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrators
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gsea_plots(
    gsea_results: dict[str, pd.DataFrame | Path],
    output_dir: Path,
    sample_id: str,
) -> dict[str, Path]:
    """Generate all GSEA plots for a single genome.

    Args:
        gsea_results: Dict from run_gsea_analysis().
        output_dir: Base output directory.
        sample_id: Sample identifier.

    Returns:
        Dict of output paths.
    """
    outputs: dict[str, Path] = {}
    plot_dir = output_dir / "gsea" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for source in ["modules", "pathways", "cog"]:
        key = f"gsea_{source}"
        if key not in gsea_results or not isinstance(gsea_results[key], pd.DataFrame):
            continue
        df = gsea_results[key]
        if df.empty:
            continue

        source_label = {
            "modules": "KEGG Modules",
            "pathways": "KEGG Pathways",
            "cog": "COG Categories",
        }[source]

        # Bar plot (NES)
        bar_path = plot_dir / f"{sample_id}_gsea_{source}_bar"
        result = plot_gsea_bar(
            df, bar_path,
            title=f"GSEA: {source_label} — {sample_id}",
            metric="nes",
        )
        if result:
            outputs[f"gsea_{source}_bar_plot"] = bar_path.with_suffix(".png")

        # Bar plot (signed -log10 p)
        bar_p_path = plot_dir / f"{sample_id}_gsea_{source}_signed_logp"
        result = plot_gsea_bar(
            df, bar_p_path,
            title=f"GSEA: {source_label} — {sample_id}",
            metric="signed_log_p",
        )
        if result:
            outputs[f"gsea_{source}_signed_logp_plot"] = bar_p_path.with_suffix(".png")

    return outputs


def generate_batch_gsea_plots(
    sample_gsea: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    condition_map: dict[str, str] | None = None,
    between_results: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Generate batch-level GSEA comparative plots.

    Args:
        sample_gsea: sample_id → {source → GSEA DataFrame}.
        output_dir: Output directory.
        condition_map: sample_id → condition label.
        between_results: source → between-condition comparison DataFrame.

    Returns:
        Dict of output paths.
    """
    from codonpipe.modules.gsea import compare_gsea_between_samples

    outputs: dict[str, Path] = {}
    plot_dir = output_dir / "batch_gsea" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for source in ["modules", "pathways", "cog"]:
        # Collect per-sample results for this source
        source_data = {}
        for sid, gsea_dict in sample_gsea.items():
            if source in gsea_dict and not gsea_dict[source].empty:
                source_data[sid] = gsea_dict[source]

        if len(source_data) < 2:
            continue

        source_label = {
            "modules": "KEGG Modules",
            "pathways": "KEGG Pathways",
            "cog": "COG Categories",
        }[source]

        # NES heatmap
        nes_matrix = compare_gsea_between_samples(source_data, source)
        if not nes_matrix.empty:
            hm_path = plot_dir / f"gsea_{source}_nes_heatmap"
            result = plot_gsea_nes_heatmap(
                nes_matrix, hm_path,
                title=f"GSEA NES: {source_label}",
                condition_map=condition_map,
            )
            if result:
                outputs[f"gsea_{source}_nes_heatmap"] = hm_path.with_suffix(".png")

        # Between-condition comparison
        if between_results and source in between_results:
            bw_df = between_results[source]
            if not bw_df.empty:
                cmp_path = plot_dir / f"gsea_{source}_condition_comparison"
                result = plot_gsea_condition_comparison(
                    bw_df, cmp_path,
                    title=f"GSEA Between-Condition: {source_label}",
                )
                if result:
                    outputs[f"gsea_{source}_condition_comparison"] = cmp_path.with_suffix(".png")

    return outputs
