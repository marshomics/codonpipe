"""Condition-aware comparative visualizations for batch mode.

Within-condition plots show variability and conserved patterns per condition.
Between-condition plots show statistical comparisons with effect sizes.
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
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from codonpipe.plotting.utils import DPI, FORMATS, apply_style as _apply_style, save_fig as _save_fig
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")

# Consistent condition palette (up to 12 conditions)
_CONDITION_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]


def _condition_colors(conditions: list[str]) -> dict[str, str]:
    unique = sorted(set(conditions))
    return {c: _CONDITION_PALETTE[i % len(_CONDITION_PALETTE)] for i, c in enumerate(unique)}


# ═══════════════════════════════════════════════════════════════════════════
# WITHIN-CONDITION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_within_metric_violins(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Violin + strip plots of key genome-level metrics per condition.

    Shows distribution shape and individual sample points within each
    condition for: median CAI, MELP, Fop, ENC, GC3, S-value, HGT fraction,
    doubling time.
    """
    _apply_style()
    plot_metrics = [
        ("median_CAI", "Median CAI"),
        ("median_MELP", "Median MELP"),
        ("median_Fop", "Median Fop"),
        ("median_ENC", "Median ENC"),
        ("mean_GC3", "Mean GC3"),
        ("mean_S_value", "Mean S-value"),
        ("hgt_fraction", "HGT fraction"),
        ("doubling_time_hours", "Doubling time (h)"),
    ]
    available = [(col, label) for col, label in plot_metrics if col in metrics_df.columns]
    if not available or condition_col not in metrics_df.columns:
        return

    n = len(available)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = _condition_colors(metrics_df[condition_col].dropna().tolist())
    palette = [colors[c] for c in sorted(colors)]
    order = sorted(metrics_df[condition_col].dropna().unique())

    n_plotted = 0
    for i, (col, label) in enumerate(available):
        ax = axes[i]
        data = metrics_df.dropna(subset=[col, condition_col])
        if data.empty:
            ax.set_visible(False)
            continue
        sns.violinplot(
            data=data, x=condition_col, y=col, hue=condition_col,
            order=order, hue_order=order,
            palette=colors, inner=None, alpha=0.3, ax=ax, cut=0,
            legend=False,
        )
        sns.stripplot(
            data=data, x=condition_col, y=col, hue=condition_col,
            order=order, hue_order=order,
            palette=colors, size=4, alpha=0.7, jitter=True, ax=ax,
            legend=False,
        )
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2, axis="y")
        n_plotted = i + 1

    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Within-Condition Metric Distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_within_rscu_heatmap(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Per-condition mean RSCU heatmap (conditions × codons).

    Rows = conditions, columns = codons grouped by amino acid.
    Shows how each condition's average codon usage profile differs.
    """
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    if not rscu_cols or condition_col not in metrics_df.columns:
        return

    # Compute per-condition mean RSCU
    grouped = metrics_df.groupby(condition_col)[rscu_cols].mean()
    if grouped.empty or len(grouped) < 2:
        return

    fig, ax = plt.subplots(figsize=(max(16, len(rscu_cols) * 0.25), max(3, len(grouped) * 0.5 + 1)))
    sns.heatmap(
        grouped, ax=ax, cmap="RdYlBu_r", center=1.0,
        linewidths=0.1, linecolor="gray",
        xticklabels=True, yticklabels=True,
        cbar_kws={"label": "Mean RSCU"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_title("Mean RSCU by Condition", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_within_rscu_cv(
    rscu_disp_df: pd.DataFrame,
    output_path: Path,
):
    """Bar chart of mean CV per condition across all codons.

    A higher bar means more variable codon usage within that condition.
    """
    _apply_style()
    if rscu_disp_df.empty or "condition" not in rscu_disp_df.columns:
        return

    summary = rscu_disp_df.groupby("condition")["cv"].agg(["mean", "std"]).reset_index()
    summary.columns = ["condition", "mean_cv", "std_cv"]
    summary = summary.sort_values("mean_cv")

    colors = _condition_colors(summary["condition"].tolist())

    fig, ax = plt.subplots(figsize=(max(4, len(summary) * 1.2), 5))
    bars = ax.bar(
        summary["condition"], summary["mean_cv"],
        yerr=summary["std_cv"], capsize=4,
        color=[colors[c] for c in summary["condition"]],
        edgecolor="black", linewidth=0.5, alpha=0.8,
    )
    ax.set_ylabel("Mean CV of RSCU", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Within-Condition RSCU Variability", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_within_pca(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """PCA of genome-level RSCU profiles with 95% confidence ellipses per condition."""
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    if not rscu_cols or condition_col not in metrics_df.columns:
        return

    df = metrics_df.dropna(subset=[condition_col]).copy()
    df[rscu_cols] = df[rscu_cols].fillna(0.0)
    if len(df) < 4:
        return

    X = StandardScaler().fit_transform(df[rscu_cols].values)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    df = df.copy()
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    colors = _condition_colors(df[condition_col].tolist())
    conditions = sorted(df[condition_col].unique())

    fig, ax = plt.subplots(figsize=(9, 7))
    for cond in conditions:
        mask = df[condition_col] == cond
        sub = df[mask]
        ax.scatter(sub["PC1"], sub["PC2"], c=colors[cond], label=cond,
                   alpha=0.7, s=50, edgecolors="black", linewidth=0.3)

        # 95% confidence ellipse
        if len(sub) >= 5:
            _draw_confidence_ellipse(sub["PC1"].values, sub["PC2"].values,
                                     ax, colors[cond], alpha=0.15)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title("PCA of RSCU Profiles by Condition", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, output_path)


def _draw_confidence_ellipse(x, y, ax, color, n_std=1.96, alpha=0.2):
    """Draw a 95% confidence ellipse."""
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        (mean_x, mean_y), width=width, height=height, angle=angle,
        facecolor=color, edgecolor=color, alpha=alpha, linewidth=1.5,
    )
    ax.add_patch(ellipse)


# ═══════════════════════════════════════════════════════════════════════════
# BETWEEN-CONDITION PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_between_metric_comparison(
    metrics_df: pd.DataFrame,
    between_tests_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Box + significance plots for each metric that differs between conditions.

    Shows box plots with Mann-Whitney U p-values and Cliff's delta annotations.
    Only metrics with at least one significant comparison are shown.
    """
    _apply_style()
    if between_tests_df.empty or condition_col not in metrics_df.columns:
        return

    mw = between_tests_df[between_tests_df["test"] == "mann_whitney_u"]
    sig_metrics = mw.loc[mw["significant"], "metric"].unique()
    if len(sig_metrics) == 0:
        # Show top 8 by smallest p-value
        sig_metrics = mw.nsmallest(8, "p_value")["metric"].unique()

    if len(sig_metrics) == 0:
        return

    n = len(sig_metrics)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = _condition_colors(metrics_df[condition_col].dropna().tolist())
    order = sorted(metrics_df[condition_col].dropna().unique())

    n_plotted = 0
    for i, metric in enumerate(sig_metrics):
        ax = axes[i]
        data = metrics_df.dropna(subset=[metric, condition_col])
        if data.empty:
            ax.set_visible(False)
            continue
        sns.boxplot(
            data=data, x=condition_col, y=metric, hue=condition_col,
            order=order, hue_order=order,
            palette=colors, ax=ax, fliersize=3, linewidth=0.8,
            legend=False,
        )
        sns.stripplot(
            data=data, x=condition_col, y=metric, order=order,
            color="black", size=3, alpha=0.4, jitter=True, ax=ax,
        )

        # Annotate significant comparisons
        metric_tests = mw[mw["metric"] == metric]
        y_max = data[metric].max()
        y_range = data[metric].max() - data[metric].min()
        offset = 0
        for _, row in metric_tests.iterrows():
            if row.get("corrected_p", 1) < 0.05:
                g1_idx = order.index(row["group1"]) if row["group1"] in order else None
                g2_idx = order.index(row["group2"]) if row["group2"] in order else None
                if g1_idx is not None and g2_idx is not None:
                    y_bar = y_max + y_range * (0.08 + 0.08 * offset)
                    ax.plot([g1_idx, g1_idx, g2_idx, g2_idx],
                            [y_bar - y_range * 0.02, y_bar, y_bar, y_bar - y_range * 0.02],
                            color="black", linewidth=0.8)
                    p_str = f"p={row['corrected_p']:.1e}"
                    d_str = f"|d|={abs(row.get('effect_size', 0)):.2f}"
                    ax.text((g1_idx + g2_idx) / 2, y_bar, f"{p_str}\n{d_str}",
                            ha="center", va="bottom", fontsize=6)
                    offset += 1

        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2, axis="y")
        n_plotted = i + 1

    for j in range(n_plotted, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Between-Condition Metric Comparisons", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rscu_volcano(
    rscu_tests_df: pd.DataFrame,
    output_path: Path,
):
    """Volcano plot: log2 fold-change vs -log10(corrected p) per codon.

    One panel per condition pair.  Codons with |log2FC| > 0.3 and FDR < 0.05
    are labeled.
    """
    _apply_style()
    if rscu_tests_df.empty:
        return

    pairs = rscu_tests_df.groupby(["group1", "group2"])
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6), squeeze=False)

    for idx, ((g1, g2), pair_df) in enumerate(pairs):
        ax = axes[0, idx]
        pair_df = pair_df.copy()
        pair_df["neg_log_p"] = -np.log10(pair_df["corrected_p"].clip(lower=1e-300))

        sig = pair_df["significant"] & (pair_df["log2_fold_change"].abs() > 0.3)
        ns = ~sig

        ax.scatter(
            pair_df.loc[ns, "log2_fold_change"],
            pair_df.loc[ns, "neg_log_p"],
            c="gray", alpha=0.4, s=30, edgecolors="none",
        )
        ax.scatter(
            pair_df.loc[sig, "log2_fold_change"],
            pair_df.loc[sig, "neg_log_p"],
            c="red", alpha=0.7, s=45, edgecolors="black", linewidth=0.3,
        )

        # Label significant codons
        for _, row in pair_df[sig].iterrows():
            codon = row["codon"].split("-")[-1] if "-" in str(row["codon"]) else str(row["codon"])
            ax.annotate(
                codon, (row["log2_fold_change"], row["neg_log_p"]),
                fontsize=6, alpha=0.8,
                xytext=(3, 3), textcoords="offset points",
            )

        # Threshold lines for significance region
        fdr_threshold = 0.05
        fc_threshold = 0.3
        ax.axhline(-np.log10(fdr_threshold), color="blue", linestyle="--", alpha=0.5, linewidth=0.8,
                   label=f"FDR = {fdr_threshold}")
        ax.axvline(fc_threshold, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(-fc_threshold, color="red", linestyle="--", alpha=0.5, linewidth=0.8,
                   label=f"|log₂FC| = {fc_threshold}")

        # Light background shading for the non-significant region
        # Only shade the region below the FDR threshold OR within the FC threshold
        y_max = ax.get_ylim()[1]
        # Shade left and right non-significant regions (low fold-change)
        ax.axvspan(-fc_threshold, fc_threshold, alpha=0.05, color="gray", zorder=0)
        ax.set_xlabel("log₂ Fold Change (RSCU)", fontsize=11)
        ax.set_ylabel("-log₁₀(FDR)", fontsize=11)
        ax.set_title(f"{g1} vs {g2}", fontsize=11)
        ax.grid(True, alpha=0.2)

    fig.suptitle("RSCU Differential Between Conditions", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rscu_condition_heatmap(
    rscu_tests_df: pd.DataFrame,
    output_path: Path,
):
    """Heatmap of log2 fold-changes (conditions × codons).

    Significant codons are marked with asterisks.
    """
    _apply_style()
    if rscu_tests_df.empty:
        return

    pairs = rscu_tests_df.groupby(["group1", "group2"])
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(max(16, 0.25 * 59), 2 + n_pairs * 1.5),
                             squeeze=False)

    for idx, ((g1, g2), pair_df) in enumerate(pairs):
        ax = axes[idx, 0]
        pivot = pair_df.set_index("codon")["log2_fold_change"]
        sig_mask = pair_df.set_index("codon")["significant"]

        # Build matrix row
        matrix = pivot.values.reshape(1, -1)
        labels = pivot.index.tolist()

        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=4, rotation=90)
        ax.set_yticks([0])
        ax.set_yticklabels([f"{g1} vs {g2}"], fontsize=9)

        # Mark significant
        for j, codon in enumerate(labels):
            if sig_mask.get(codon, False):
                ax.text(j, 0, "*", ha="center", va="center", fontsize=8,
                        color="black", weight="bold")

    fig.colorbar(im, ax=axes, label="log₂ FC (RSCU)", shrink=0.6)
    fig.suptitle("RSCU Fold-Change Between Conditions", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_enc_gc3_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """ENC vs GC3 scatter by condition with Wright's expected curve.

    Shows whether conditions differ in the balance of mutation vs selection.
    """
    _apply_style()
    if "median_ENC" not in metrics_df.columns or "mean_GC3" not in metrics_df.columns:
        return
    if condition_col not in metrics_df.columns:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = _condition_colors(metrics_df[condition_col].dropna().tolist())

    for cond in sorted(metrics_df[condition_col].dropna().unique()):
        sub = metrics_df[metrics_df[condition_col] == cond]
        ax.scatter(sub["mean_GC3"], sub["median_ENC"],
                   c=colors[cond], label=cond, s=50, alpha=0.7,
                   edgecolors="black", linewidth=0.3)

    # Wright's expected ENC curve
    gc3 = np.linspace(0.01, 0.99, 200)
    enc_exp = 2 + gc3 + 29 / (gc3**2 + (1 - gc3)**2)
    ax.plot(gc3, enc_exp, "k--", alpha=0.4, linewidth=1, label="Wright expected")

    ax.set_xlabel("Mean GC3", fontsize=11)
    ax.set_ylabel("Median ENC", fontsize=11)
    ax.set_title("ENC vs GC3 by Condition", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_growth_rate_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Strip + box plot of predicted doubling times per condition."""
    _apply_style()
    if "doubling_time_hours" not in metrics_df.columns or condition_col not in metrics_df.columns:
        return

    data = metrics_df.dropna(subset=["doubling_time_hours", condition_col])
    if data.empty:
        return

    colors = _condition_colors(data[condition_col].tolist())
    order = sorted(data[condition_col].unique())

    fig, ax = plt.subplots(figsize=(max(4, len(order) * 1.5), 5))

    sns.boxplot(data=data, x=condition_col, y="doubling_time_hours",
                hue=condition_col, order=order, hue_order=order,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False)
    sns.stripplot(data=data, x=condition_col, y="doubling_time_hours", order=order,
                  color="black", size=5, alpha=0.5, jitter=True, ax=ax)

    # Growth class zones
    ax.axhspan(0, 2, color="green", alpha=0.05)
    ax.axhspan(2, 8, color="yellow", alpha=0.05)
    ax.axhspan(8, ax.get_ylim()[1], color="red", alpha=0.05)

    ax.set_ylabel("Predicted Doubling Time (hours)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Predicted Growth Rate by Condition", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_hgt_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """HGT fraction bar chart by condition."""
    _apply_style()
    if "hgt_fraction" not in metrics_df.columns or condition_col not in metrics_df.columns:
        return

    data = metrics_df.dropna(subset=["hgt_fraction", condition_col])
    if data.empty:
        return

    colors = _condition_colors(data[condition_col].tolist())
    order = sorted(data[condition_col].unique())

    fig, ax = plt.subplots(figsize=(max(4, len(order) * 1.5), 5))
    sns.boxplot(data=data, x=condition_col, y="hgt_fraction",
                hue=condition_col, order=order, hue_order=order,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False)
    sns.stripplot(data=data, x=condition_col, y="hgt_fraction", order=order,
                  color="black", size=5, alpha=0.5, jitter=True, ax=ax)

    ax.set_ylabel("HGT Candidate Fraction", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Horizontal Gene Transfer Prevalence by Condition", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_neutrality_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Neutrality plot regression slopes by condition.

    Bar chart of GC12 vs GC3 regression slopes: slope ≈ 1 indicates
    mutational bias dominates; slope ≈ 0 indicates strong selection.
    """
    _apply_style()
    if "neutrality_slope" not in metrics_df.columns or condition_col not in metrics_df.columns:
        return

    data = metrics_df.dropna(subset=["neutrality_slope", condition_col])
    if data.empty:
        return

    colors = _condition_colors(data[condition_col].tolist())
    order = sorted(data[condition_col].unique())

    fig, ax = plt.subplots(figsize=(max(4, len(order) * 1.5), 5))
    sns.boxplot(data=data, x=condition_col, y="neutrality_slope",
                hue=condition_col, order=order, hue_order=order,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False)
    sns.stripplot(data=data, x=condition_col, y="neutrality_slope", order=order,
                  color="black", size=5, alpha=0.5, jitter=True, ax=ax)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.4, linewidth=1, label="Pure mutation")
    ax.axhline(0.0, color="blue", linestyle="--", alpha=0.4, linewidth=1, label="Pure selection")
    ax.set_ylabel("Neutrality Slope (GC12 vs GC3)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title("Mutation-Selection Balance by Condition", fontsize=12)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_radar_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Radar/spider chart comparing mean-scaled metrics across conditions.

    Normalizes each metric to [0, 1] range across all conditions, then
    overlays one polygon per condition.
    """
    _apply_style()
    radar_metrics = [
        ("median_CAI", "CAI"), ("median_MELP", "MELP"), ("median_Fop", "Fop"),
        ("median_ENC", "ENC"), ("mean_GC3", "GC3"),
        ("mean_S_value", "S-value"), ("hgt_fraction", "HGT"),
    ]
    available = [(col, label) for col, label in radar_metrics
                 if col in metrics_df.columns and metrics_df[col].notna().sum() > 0]
    if len(available) < 3 or condition_col not in metrics_df.columns:
        return

    cols = [c for c, _ in available]
    labels = [l for _, l in available]
    conditions = sorted(metrics_df[condition_col].dropna().unique())
    if len(conditions) < 2:
        return

    # Compute per-condition means
    means = metrics_df.groupby(condition_col)[cols].mean()

    # Min-max normalize each metric across conditions
    normed = (means - means.min()) / (means.max() - means.min() + 1e-10)

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    colors = _condition_colors(conditions)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for cond in conditions:
        values = normed.loc[cond].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=cond,
                color=colors[cond], alpha=0.8, markersize=4)
        ax.fill(angles, values, alpha=0.08, color=colors[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Codon Usage Profile by Condition", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    # Annotation noting independent normalization
    ax.annotate("Note: Each metric is independently min-max normalized",
                xy=(0.5, -0.05), xycoords="axes fraction", ha="center",
                fontsize=7, fontstyle="italic", color="gray")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_permanova_summary(
    perm_result: dict,
    output_path: Path,
):
    """Simple text/annotation plot for PERMANOVA result."""
    _apply_style()
    if not perm_result:
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    text = (
        f"PERMANOVA on RSCU Profiles\n\n"
        f"F-statistic: {perm_result.get('F_statistic', 'N/A')}\n"
        f"p-value: {perm_result.get('p_value', 'N/A')}\n"
        f"R²: {perm_result.get('R2', 'N/A')}\n"
        f"Permutations: {perm_result.get('n_perm', 'N/A')}\n"
        f"Samples: {perm_result.get('n_samples', 'N/A')} | "
        f"Groups: {perm_result.get('n_groups', 'N/A')}"
    )
    sig = perm_result.get("p_value", 1) < 0.05
    color = "darkgreen" if sig else "darkred"
    verdict = "Significant difference" if sig else "No significant difference"

    ax.text(0.5, 0.6, text, transform=ax.transAxes, fontsize=11,
            va="center", ha="center", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.text(0.5, 0.08, verdict, transform=ax.transAxes, fontsize=13,
            va="center", ha="center", weight="bold", color=color)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED COMPARATIVE PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def plot_effect_size_forest(
    between_tests_df: pd.DataFrame,
    output_path: Path,
):
    """Forest plot of Cliff's delta effect sizes with 95% CIs for every metric.

    Metrics are sorted by absolute effect size. Significant results (FDR < 0.05)
    are colored; non-significant results are gray.  Vertical dashed lines mark
    the negligible/small/medium/large effect-size thresholds.
    """
    _apply_style()
    if between_tests_df.empty:
        return

    mw = between_tests_df[between_tests_df["test"] == "mann_whitney_u"].copy()
    if mw.empty:
        return

    # Sort by absolute effect size descending
    mw["abs_effect"] = mw["effect_size"].abs()
    mw = mw.sort_values("abs_effect", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(mw) * 0.35)))
    y_pos = np.arange(len(mw))

    colors = []
    for _, row in mw.iterrows():
        if row.get("significant", False):
            colors.append("#d62728" if row["effect_size"] > 0 else "#1f77b4")
        else:
            colors.append("#999999")

    ax.barh(y_pos, mw["effect_size"], height=0.6, color=colors,
            edgecolor="black", linewidth=0.3, alpha=0.8)

    # Effect size threshold lines
    for threshold, label in [(0.147, "small"), (0.33, "medium"), (0.474, "large")]:
        ax.axvline(threshold, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
        ax.axvline(-threshold, color="gray", linestyle=":", alpha=0.4, linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.8)

    # Labels
    labels = []
    for _, row in mw.iterrows():
        metric_name = row["metric"].replace("_", " ").title()
        p_str = f"q={row.get('corrected_p', row['p_value']):.1e}"
        labels.append(f"{metric_name}  ({p_str})")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cliff's delta (effect size)", fontsize=11)
    ax.set_title("Between-Condition Effect Sizes", fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Significant (higher in g2)"),
        Patch(facecolor="#1f77b4", label="Significant (higher in g1)"),
        Patch(facecolor="#999999", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.15, axis="x")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rscu_paired_dot(
    rscu_tests_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 25,
):
    """Cleveland dot plot showing mean RSCU per codon in each condition.

    Dots are connected by lines; color indicates which condition has higher
    RSCU. Only the top_n most different codons (by |log2FC|) are shown.
    """
    _apply_style()
    if rscu_tests_df.empty:
        return

    pairs = list(rscu_tests_df.groupby(["group1", "group2"]))
    if not pairs:
        return

    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, max(6, top_n * 0.3)),
                              squeeze=False)

    for idx, ((g1, g2), pair_df) in enumerate(pairs):
        ax = axes[0, idx]
        pair_df = pair_df.copy()
        pair_df["abs_lfc"] = pair_df["log2_fold_change"].abs()
        top = pair_df.nlargest(top_n, "abs_lfc").sort_values("log2_fold_change")

        y_pos = np.arange(len(top))

        for i, (_, row) in enumerate(top.iterrows()):
            color = "#d62728" if row["mean_g2"] > row["mean_g1"] else "#1f77b4"
            line_alpha = 0.8 if row.get("significant", False) else 0.25
            marker_size = 7 if row.get("significant", False) else 4

            ax.plot([row["mean_g1"], row["mean_g2"]], [i, i],
                    color=color, linewidth=1.5, alpha=line_alpha)
            ax.scatter(row["mean_g1"], i, color="#1f77b4", s=marker_size**2,
                       zorder=5, edgecolors="black", linewidth=0.3)
            ax.scatter(row["mean_g2"], i, color="#d62728", s=marker_size**2,
                       zorder=5, edgecolors="black", linewidth=0.3)

            # Star for significant
            if row.get("significant", False):
                x_label = max(row["mean_g1"], row["mean_g2"]) + 0.05
                ax.text(x_label, i, "*", fontsize=10, va="center", color="black",
                        weight="bold")

        codon_labels = []
        for _, row in top.iterrows():
            codon = row["codon"]
            aa = row.get("amino_acid", codon.split("-")[0])
            triplet = codon.split("-")[-1] if "-" in str(codon) else str(codon)
            codon_labels.append(f"{triplet} ({aa})")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(codon_labels, fontsize=7)
        ax.set_xlabel("Mean RSCU", fontsize=11)
        ax.set_title(f"{g1} vs {g2}", fontsize=11)
        ax.grid(True, alpha=0.15, axis="x")

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                   markersize=7, label=g1),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=7, label=g2),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    fig.suptitle("RSCU Differences: Top Differentially Used Codons", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_condition_summary_dashboard(
    metrics_df: pd.DataFrame,
    condition_col: str,
    between_tests_df: pd.DataFrame | None,
    perm_result: dict | None,
    output_path: Path,
):
    """Multi-panel summary dashboard: PCA, key metrics, PERMANOVA, top effects.

    A single figure that provides a publication-ready overview of the main
    differences between conditions.
    """
    _apply_style()
    if condition_col not in metrics_df.columns:
        return

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    if len(conditions) < 2:
        return

    colors = _condition_colors(conditions)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # ── Panel A: PCA with ellipses ──────────────────────────────────
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in metrics_df.columns]
    ax_pca = fig.add_subplot(gs[0, 0])
    if rscu_cols:
        df = metrics_df.dropna(subset=[condition_col]).copy()
        df[rscu_cols] = df[rscu_cols].fillna(0.0)
        if len(df) >= 4:
            X = StandardScaler().fit_transform(df[rscu_cols].values)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            for cond in conditions:
                mask = df[condition_col].values == cond
                ax_pca.scatter(coords[mask, 0], coords[mask, 1], c=colors[cond],
                               label=cond, s=40, alpha=0.7, edgecolors="black",
                               linewidth=0.3)
                if mask.sum() >= 5:
                    _draw_confidence_ellipse(coords[mask, 0], coords[mask, 1],
                                             ax_pca, colors[cond], alpha=0.12)
            ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=9)
            ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=9)
            ax_pca.legend(fontsize=7, framealpha=0.8)
    ax_pca.set_title("A. RSCU Profile Separation", fontsize=10, weight="bold")
    ax_pca.grid(True, alpha=0.15)

    # ── Panel B: Key expression metrics ─────────────────────────────
    ax_expr = fig.add_subplot(gs[0, 1])
    expr_metrics = [("median_CAI", "CAI"), ("median_MELP", "MELP"), ("median_ENC", "ENC")]
    available_expr = [(c, l) for c, l in expr_metrics if c in metrics_df.columns]
    if available_expr:
        plot_data = []
        for col, label in available_expr:
            for cond in conditions:
                vals = metrics_df.loc[metrics_df[condition_col] == cond, col].dropna()
                for v in vals:
                    plot_data.append({"metric": label, "condition": cond, "value": v})
        if plot_data:
            pdf = pd.DataFrame(plot_data)
            for i, (col, label) in enumerate(available_expr):
                for j, cond in enumerate(conditions):
                    sub = pdf[(pdf["metric"] == label) & (pdf["condition"] == cond)]
                    x_pos = i + (j - 0.5 * (len(conditions) - 1)) * 0.25
                    bp = ax_expr.boxplot(
                        sub["value"].values, positions=[x_pos], widths=0.2,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=colors[cond], alpha=0.6),
                        medianprops=dict(color="black"),
                    )
            ax_expr.set_xticks(range(len(available_expr)))
            ax_expr.set_xticklabels([l for _, l in available_expr], fontsize=9)
    ax_expr.set_title("B. Expression & Bias Metrics", fontsize=10, weight="bold")
    ax_expr.grid(True, alpha=0.15, axis="y")

    # ── Panel C: Biological metrics ─────────────────────────────────
    ax_bio = fig.add_subplot(gs[0, 2])
    bio_metrics = [
        ("hgt_fraction", "HGT\nfraction"),
        ("strand_asym_fraction", "Strand\nasymmetry"),
        ("neutrality_slope", "Neutrality\nslope"),
    ]
    available_bio = [(c, l) for c, l in bio_metrics if c in metrics_df.columns]
    if available_bio:
        for i, (col, label) in enumerate(available_bio):
            for j, cond in enumerate(conditions):
                vals = metrics_df.loc[metrics_df[condition_col] == cond, col].dropna()
                x_pos = i + (j - 0.5 * (len(conditions) - 1)) * 0.25
                bp = ax_bio.boxplot(
                    vals.values, positions=[x_pos], widths=0.2,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=colors[cond], alpha=0.6),
                    medianprops=dict(color="black"),
                )
        ax_bio.set_xticks(range(len(available_bio)))
        ax_bio.set_xticklabels([l for _, l in available_bio], fontsize=8)
    ax_bio.set_title("C. Biological Features", fontsize=10, weight="bold")
    ax_bio.grid(True, alpha=0.15, axis="y")

    # ── Panel D: PERMANOVA result ───────────────────────────────────
    ax_perm = fig.add_subplot(gs[1, 0])
    ax_perm.axis("off")
    if perm_result:
        sig = perm_result.get("p_value", 1) < 0.05
        verdict_color = "#2ca02c" if sig else "#d62728"
        verdict = "RSCU profiles differ significantly" if sig else "No significant RSCU difference"
        text = (
            f"PERMANOVA\n"
            f"F = {perm_result.get('F_statistic', 'N/A'):.2f}\n"
            f"p = {perm_result.get('p_value', 'N/A'):.4f}\n"
            f"R² = {perm_result.get('R2', 'N/A'):.3f}\n"
            f"({perm_result.get('n_perm', 999)} permutations)"
        )
        ax_perm.text(0.5, 0.6, text, transform=ax_perm.transAxes, fontsize=11,
                     va="center", ha="center", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                               edgecolor="gray", alpha=0.9))
        ax_perm.text(0.5, 0.12, verdict, transform=ax_perm.transAxes, fontsize=11,
                     va="center", ha="center", weight="bold", color=verdict_color)
    ax_perm.set_title("D. PERMANOVA", fontsize=10, weight="bold")

    # ── Panel E: Top significant effect sizes ───────────────────────
    ax_eff = fig.add_subplot(gs[1, 1:])
    if between_tests_df is not None and not between_tests_df.empty:
        mw = between_tests_df[between_tests_df["test"] == "mann_whitney_u"].copy()
        mw["abs_effect"] = mw["effect_size"].abs()
        top = mw.nlargest(10, "abs_effect")
        if not top.empty:
            top = top.sort_values("abs_effect", ascending=True).reset_index(drop=True)
            y_pos = np.arange(len(top))
            bar_colors = []
            for _, row in top.iterrows():
                if row.get("significant", False):
                    bar_colors.append("#d62728" if row["effect_size"] > 0 else "#1f77b4")
                else:
                    bar_colors.append("#cccccc")

            ax_eff.barh(y_pos, top["effect_size"], height=0.55, color=bar_colors,
                        edgecolor="black", linewidth=0.3, alpha=0.85)
            labels = [f"{r['metric'].replace('_', ' ').title()} (q={r.get('corrected_p', r['p_value']):.1e})"
                      for _, r in top.iterrows()]
            ax_eff.set_yticks(y_pos)
            ax_eff.set_yticklabels(labels, fontsize=7)
            ax_eff.axvline(0, color="black", linewidth=0.8)
            for threshold in (0.147, 0.33, 0.474):
                ax_eff.axvline(threshold, color="gray", linestyle=":", alpha=0.3)
                ax_eff.axvline(-threshold, color="gray", linestyle=":", alpha=0.3)
    ax_eff.set_xlabel("Cliff's delta", fontsize=9)
    ax_eff.set_title("E. Top Effect Sizes (metrics)", fontsize=10, weight="bold")
    ax_eff.grid(True, alpha=0.15, axis="x")

    # Condition legend at top
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[c], edgecolor="black", linewidth=0.5, label=c)
                       for c in conditions]
    fig.legend(handles=legend_elements, loc="upper center", ncol=len(conditions),
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Condition Comparison Summary", fontsize=14, weight="bold", y=1.05)
    _save_fig(fig, output_path)


def plot_translational_selection_comparison(
    metrics_df: pd.DataFrame,
    condition_col: str,
    between_tests_df: pd.DataFrame | None,
    output_path: Path,
):
    """Two-panel figure: Fop gradient slopes and position effects by condition.

    Left: box+strip of fop_gradient_slope per condition.
    Right: grouped bar chart of mean 5'/middle/3' Fop per condition.
    """
    _apply_style()
    if condition_col not in metrics_df.columns:
        return

    has_gradient = "fop_gradient_slope" in metrics_df.columns
    has_position = all(c in metrics_df.columns for c in
                       ("mean_fop_5prime", "mean_fop_middle", "mean_fop_3prime"))

    if not has_gradient and not has_position:
        return

    n_panels = int(has_gradient) + int(has_position)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    colors = _condition_colors(conditions)
    panel_idx = 0

    if has_gradient:
        ax = axes[panel_idx]
        data = metrics_df.dropna(subset=["fop_gradient_slope", condition_col])
        if not data.empty:
            sns.boxplot(
                data=data, x=condition_col, y="fop_gradient_slope",
                hue=condition_col, order=conditions, hue_order=conditions,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False,
            )
            sns.stripplot(
                data=data, x=condition_col, y="fop_gradient_slope",
                order=conditions, color="black", size=5, alpha=0.5,
                jitter=True, ax=ax,
            )
            # Annotate significance
            if between_tests_df is not None and not between_tests_df.empty:
                row = between_tests_df[
                    (between_tests_df["metric"] == "fop_gradient_slope")
                    & (between_tests_df["test"] == "mann_whitney_u")
                ]
                if not row.empty:
                    r = row.iloc[0]
                    p_val = r.get("corrected_p", r["p_value"])
                    star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                    y_max = data["fop_gradient_slope"].max()
                    y_range = data["fop_gradient_slope"].max() - data["fop_gradient_slope"].min()
                    ax.text(0.5, y_max + y_range * 0.1, f"{star} (q={p_val:.1e})",
                            ha="center", fontsize=9, transform=ax.get_xaxis_transform())

        ax.set_ylabel("Fop Gradient Slope", fontsize=11)
        ax.set_xlabel("")
        ax.set_title("Translational Selection Gradient", fontsize=11)
        ax.grid(True, alpha=0.15, axis="y")
        panel_idx += 1

    if has_position:
        ax = axes[panel_idx]
        positions = ["mean_fop_5prime", "mean_fop_middle", "mean_fop_3prime"]
        pos_labels = ["5' end", "Middle", "3' end"]
        x_pos = np.arange(len(positions))
        bar_width = 0.35

        for j, cond in enumerate(conditions):
            sub = metrics_df[metrics_df[condition_col] == cond]
            means = [sub[col].mean() for col in positions]
            sems = [sub[col].sem() for col in positions]
            offset = (j - 0.5 * (len(conditions) - 1)) * bar_width
            ax.bar(x_pos + offset, means, bar_width * 0.9, yerr=sems,
                   color=colors[cond], label=cond, edgecolor="black",
                   linewidth=0.3, capsize=3, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(pos_labels, fontsize=10)
        ax.set_ylabel("Mean Fop", fontsize=11)
        ax.set_title("Positional Fop by Condition", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15, axis="y")

    fig.suptitle("Translational Selection Comparison", fontsize=12, weight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_strand_asymmetry_comparison(
    metrics_df: pd.DataFrame,
    condition_col: str,
    between_tests_df: pd.DataFrame | None,
    output_path: Path,
):
    """Box+strip of strand asymmetry fraction by condition with significance."""
    _apply_style()
    if "strand_asym_fraction" not in metrics_df.columns or condition_col not in metrics_df.columns:
        return

    data = metrics_df.dropna(subset=["strand_asym_fraction", condition_col])
    if data.empty:
        return

    conditions = sorted(data[condition_col].unique())
    colors = _condition_colors(conditions)

    fig, ax = plt.subplots(figsize=(max(4, len(conditions) * 1.5), 5))
    sns.boxplot(data=data, x=condition_col, y="strand_asym_fraction",
                hue=condition_col, order=conditions, hue_order=conditions,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False)
    sns.stripplot(data=data, x=condition_col, y="strand_asym_fraction",
                  order=conditions, color="black", size=5, alpha=0.5,
                  jitter=True, ax=ax)

    # Annotate
    if between_tests_df is not None and not between_tests_df.empty:
        row = between_tests_df[
            (between_tests_df["metric"] == "strand_asym_fraction")
            & (between_tests_df["test"] == "mann_whitney_u")
        ]
        if not row.empty:
            r = row.iloc[0]
            p_val = r.get("corrected_p", r["p_value"])
            star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            y_max = data["strand_asym_fraction"].max()
            y_range = data["strand_asym_fraction"].max() - data["strand_asym_fraction"].min()
            ax.text(0.5, y_max + max(y_range * 0.1, 0.01), f"{star} (q={p_val:.1e})",
                    ha="center", fontsize=9, transform=ax.get_xaxis_transform())

    ax.set_ylabel("Fraction of Codons with Strand Asymmetry", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Strand-Biased Codon Usage by Condition", fontsize=12)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_operon_coadaptation_comparison(
    metrics_df: pd.DataFrame,
    condition_col: str,
    between_tests_df: pd.DataFrame | None,
    output_path: Path,
):
    """Box+strip of operon RSCU distance by condition."""
    _apply_style()
    if "operon_mean_rscu_dist" not in metrics_df.columns or condition_col not in metrics_df.columns:
        return

    data = metrics_df.dropna(subset=["operon_mean_rscu_dist", condition_col])
    if data.empty:
        return

    conditions = sorted(data[condition_col].unique())
    colors = _condition_colors(conditions)

    fig, ax = plt.subplots(figsize=(max(4, len(conditions) * 1.5), 5))
    sns.boxplot(data=data, x=condition_col, y="operon_mean_rscu_dist",
                hue=condition_col, order=conditions, hue_order=conditions,
                palette=colors, ax=ax, linewidth=0.8, fliersize=0, legend=False)
    sns.stripplot(data=data, x=condition_col, y="operon_mean_rscu_dist",
                  order=conditions, color="black", size=5, alpha=0.5,
                  jitter=True, ax=ax)

    ax.set_ylabel("Mean RSCU Distance (Operon Pairs)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Operon Codon Coadaptation by Condition", fontsize=12)
    ax.grid(True, alpha=0.15, axis="y")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# BIO/ECOLOGY BETWEEN-CONDITION PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def plot_hgt_burden_comparison(
    hgt_burden: dict,
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
) -> None:
    """Two-panel comparison of HGT burden between conditions.

    Left: box+strip of median Mahalanobis distance per sample.
    Right: box+strip of HGT candidate fraction per sample.
    Both annotated with Mann-Whitney U p-value and Cliff's delta.
    """
    _apply_style()
    conditions = metrics_df[condition_col].dropna().unique()
    colors = _condition_colors(list(conditions))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    panels = [
        ("mahalanobis", "median_mahalanobis_dist", "Median Mahalanobis Distance",
         "Genomic RSCU Heterogeneity"),
        ("hgt_fraction", "hgt_fraction", "HGT Candidate Fraction",
         "Proportion of Genes Flagged as HGT"),
    ]

    n_conds = len(conditions)
    for ax, (burden_key, metric_col, ylabel, title) in zip(axes, panels):
        if metric_col in metrics_df.columns:
            plot_data = metrics_df[[condition_col, metric_col]].dropna()
            if not plot_data.empty:
                order = sorted(conditions)
                sns.boxplot(
                    data=plot_data, x=condition_col, y=metric_col,
                    order=order, palette=colors, ax=ax, width=0.5,
                    hue=condition_col, hue_order=order, legend=False,
                    boxprops=dict(alpha=0.6), showfliers=False,
                )
                sns.stripplot(
                    data=plot_data, x=condition_col, y=metric_col,
                    order=order, palette=colors, ax=ax,
                    hue=condition_col, hue_order=order, legend=False,
                    size=6, alpha=0.7, jitter=0.15,
                )
                ax.set_ylabel(ylabel, fontsize=10)
                ax.set_title(title, fontsize=11)

                # Annotate with test results
                if n_conds == 2:
                    # 2-condition: use backward-compat "mahalanobis" key
                    if burden_key in hgt_burden:
                        info = hgt_burden[burden_key]
                        p_val = info.get("p_value", 1)
                        delta = info.get("cliffs_delta", 0)
                        label = info.get("effect_size", "")
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        ax.text(0.5, 0.98, f"p = {p_val:.3e}  |  δ = {delta:.3f} ({label}) {sig}",
                                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
                else:
                    # 3+ conditions: use omnibus key if available
                    omnibus_key = f"{burden_key}_omnibus"
                    if omnibus_key in hgt_burden:
                        info = hgt_burden[omnibus_key]
                        p_val = info.get("p_value", 1)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        ax.text(0.5, 0.98, f"Omnibus: p = {p_val:.3e} {sig}",
                                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title, fontsize=11)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_phage_mobile_comparison(
    metrics_df: pd.DataFrame,
    condition_col: str,
    between_tests_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Bar+strip comparison of phage/mobile element counts between conditions."""
    _apply_style()
    # Check for phage-related columns
    phage_cols = [c for c in ("n_phage_mobile", "n_mobilome", "n_phage") if c in metrics_df.columns]
    if not phage_cols:
        return

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    colors = _condition_colors(list(conditions))
    n_panels = len(phage_cols)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    labels = {
        "n_phage_mobile": "Phage/Mobile Element Genes",
        "n_mobilome": "Mobilome Genes (COG X)",
        "n_phage": "Phage-Related Genes",
    }

    for ax, col in zip(axes, phage_cols):
        plot_data = metrics_df[[condition_col, col]].dropna()
        if plot_data.empty:
            continue
        sns.boxplot(
            data=plot_data, x=condition_col, y=col,
            order=conditions, palette=colors, ax=ax, width=0.5,
            hue=condition_col, hue_order=conditions, legend=False,
            boxprops=dict(alpha=0.6), showfliers=False,
        )
        sns.stripplot(
            data=plot_data, x=condition_col, y=col,
            order=conditions, palette=colors, ax=ax,
            hue=condition_col, hue_order=conditions, legend=False,
            size=6, alpha=0.7, jitter=0.15,
        )
        ax.set_ylabel(labels.get(col, col), fontsize=10)
        ax.set_title(labels.get(col, col), fontsize=11)

        # Add significance from between_tests if available
        if between_tests_df is not None and not between_tests_df.empty:
            match = between_tests_df[between_tests_df["metric"] == col]
            if not match.empty:
                p_val = match.iloc[0].get("p_adjusted", match.iloc[0].get("p_value", 1))
                delta = match.iloc[0].get("cliffs_delta", 0)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.text(0.5, 0.98, f"p = {p_val:.3e}  |  δ = {delta:.3f} {sig}",
                        transform=ax.transAxes, ha="center", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    fig.suptitle("Phage & Mobile Element Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_optimal_codon_divergence(
    optimal_codons_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Dot plot showing which codons are optimal in each condition.

    For 2 conditions: X-axis: delta-RSCU in condition A, Y-axis: delta-RSCU in condition B.
    For 3+ conditions: Heatmap of mean_delta_rscu across all conditions with optimal indicator.
    Points are colored by agreement (green = same, red = divergent).
    Quadrant lines at 0 divide optimal/non-optimal per condition.
    """
    _apply_style()
    if optimal_codons_df.empty:
        return

    # Identify the condition columns
    delta_cols = [c for c in optimal_codons_df.columns if c.startswith("mean_delta_rscu_")]
    if len(delta_cols) < 2:
        return

    if len(delta_cols) == 2:
        # 2-condition scatter plot
        cond_a = delta_cols[0].replace("mean_delta_rscu_", "")
        cond_b = delta_cols[1].replace("mean_delta_rscu_", "")

        fig, ax = plt.subplots(figsize=(8, 8))

        agree = optimal_codons_df.get("unanimous", optimal_codons_df.get("agreement", pd.Series(True, index=optimal_codons_df.index)))
        ax.scatter(
            optimal_codons_df.loc[agree, delta_cols[0]],
            optimal_codons_df.loc[agree, delta_cols[1]],
            c="#2ca02c", alpha=0.6, s=40, label="Agree", edgecolors="none",
        )
        ax.scatter(
            optimal_codons_df.loc[~agree, delta_cols[0]],
            optimal_codons_df.loc[~agree, delta_cols[1]],
            c="#d62728", alpha=0.8, s=60, label="Diverge", edgecolors="black", linewidths=0.5,
        )

        # Label the most divergent codons
        if "delta_difference" in optimal_codons_df.columns:
            top_div = optimal_codons_df.loc[~agree].nlargest(10, "delta_difference", keep="first")
            bottom_div = optimal_codons_df.loc[~agree].nsmallest(10, "delta_difference", keep="first")
            for _, row in pd.concat([top_div, bottom_div]).iterrows():
                ax.annotate(
                    row["codon"], (row[delta_cols[0]], row[delta_cols[1]]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points",
                )

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel(f"ΔRSCU (high-expr enrichment) — {cond_a}", fontsize=11)
        ax.set_ylabel(f"ΔRSCU (high-expr enrichment) — {cond_b}", fontsize=11)
        ax.set_title("Optimal Codon Divergence Between Conditions", fontsize=12)
        ax.legend(loc="upper left", fontsize=9)

        # Quadrant labels
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.text(lim * 0.6, lim * 0.9, f"Optimal in both", fontsize=8, color="gray", ha="center")
        ax.text(-lim * 0.6, -lim * 0.9, f"Non-optimal in both", fontsize=8, color="gray", ha="center")
        ax.text(lim * 0.6, -lim * 0.9, f"Optimal only\nin {cond_a}", fontsize=8, color="gray", ha="center")
        ax.text(-lim * 0.6, lim * 0.9, f"Optimal only\nin {cond_b}", fontsize=8, color="gray", ha="center")

        n_agree = agree.sum()
        n_disagree = (~agree).sum()
        ax.text(0.02, 0.02, f"Agreement: {n_agree}/{len(agree)} codons  ({100*n_agree/len(agree):.0f}%)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    else:
        # 3+ conditions: heatmap of delta-RSCU values
        conditions = [c.replace("mean_delta_rscu_", "") for c in delta_cols]
        df = optimal_codons_df.copy()
        df = df.sort_values(delta_cols[0], key=abs, ascending=False)

        # Build heatmap matrix
        heat_data = df[delta_cols].copy()
        heat_data.columns = conditions
        heat_data.index = df["codon"]

        # Build heatmap with n_conditions_optimal as side column
        fig, (ax_heat, ax_side) = plt.subplots(
            1, 2, figsize=(10 + len(conditions) * 0.8, max(8, len(df) * 0.25)),
            gridspec_kw={"width_ratios": [1, 0.15]}
        )

        # Heatmap
        vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()), 0.1)
        im = ax_heat.imshow(heat_data.values, aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax)
        ax_heat.set_xticks(range(len(conditions)))
        ax_heat.set_xticklabels(conditions, fontsize=10)
        ax_heat.set_yticks(range(len(heat_data)))
        ax_heat.set_yticklabels(heat_data.index, fontsize=7)
        ax_heat.set_title("ΔRSCU (high-expr enrichment) Across Conditions", fontsize=11)
        plt.colorbar(im, ax=ax_heat, shrink=0.8, label="ΔRSCU")

        # Side column: n_conditions_optimal
        if "n_conditions_optimal" in df.columns:
            n_opt = df["n_conditions_optimal"].values
            n_opt_normalized = n_opt / len(conditions)
            ax_side.imshow(n_opt_normalized.reshape(-1, 1), aspect="auto", cmap="YlGn",
                           vmin=0, vmax=1)
            ax_side.set_xticks([0])
            ax_side.set_xticklabels(["n_opt"], fontsize=9)
            ax_side.set_yticks(range(len(heat_data)))
            ax_side.set_yticklabels([""] * len(heat_data), fontsize=7)
            # Add text annotations for n_conditions_optimal
            for i, val in enumerate(n_opt):
                ax_side.text(0, i, str(int(val)), ha="center", va="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_strand_asymmetry_pattern_comparison(
    strand_asym_patterns_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap + lollipop of per-codon strand asymmetry differences between conditions.

    Top: heatmap of mean asymmetry (plus-minus RSCU) per condition per codon.
    Bottom: lollipop plot of difference in asymmetry, colored by significance.
    For 3+ conditions: multi-panel lollipops (one per pair) with shared heatmap.
    """
    _apply_style()
    if strand_asym_patterns_df.empty:
        return

    df = strand_asym_patterns_df.copy()

    # Check for pairwise format (test, group1, group2 columns)
    has_pairs = "test" in df.columns and "group1" in df.columns and "group2" in df.columns

    if has_pairs:
        # Filter to pairwise mann_whitney_u results
        df = df[df["test"] == "mann_whitney_u"].copy()
        if df.empty:
            return

        # Get unique pairs
        pair_list = []
        for (g1, g2), group_data in df.groupby(["group1", "group2"], sort=False):
            pair_list.append((g1, g2, group_data))

        n_pairs = len(pair_list)
        fig = plt.figure(figsize=(max(12, 8 * n_pairs), 10))
        gs = gridspec.GridSpec(2, n_pairs, height_ratios=[1, 2], figure=fig)

        # Collect all conditions for heatmap
        all_conditions = set()
        for g1, g2, _ in pair_list:
            all_conditions.add(g1)
            all_conditions.add(g2)
        all_conditions = sorted(all_conditions)

        # Build heatmap from mean_group1/mean_group2 values
        # For simplicity, reconstruct per-condition means from first pair's data
        first_pair_data = pair_list[0][2]
        all_codons = sorted(first_pair_data["codon"].unique()) if "codon" in first_pair_data.columns else []

        # Create heatmap for all conditions
        ax_heat = fig.add_subplot(gs[0, :])
        heat_data = pd.DataFrame(index=all_codons)

        for pair_idx, (cond_a, cond_b, pair_df) in enumerate(pair_list):
            pair_df = pair_df.sort_values("diff", key=abs, ascending=False)
            # Use mean_group1 and mean_group2 for this pair
            if pair_idx == 0:
                # First pair: use its values for heatmap columns
                heat_data[cond_a] = pair_df.set_index("codon").reindex(all_codons)["mean_group1"]
                heat_data[cond_b] = pair_df.set_index("codon").reindex(all_codons)["mean_group2"]

        if not heat_data.empty and heat_data.notna().any().any():
            vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()))
            im = ax_heat.imshow(heat_data.T.values, aspect="auto", cmap="RdBu_r",
                                vmin=-vmax, vmax=vmax)
            ax_heat.set_xticks(range(len(all_codons)))
            ax_heat.set_xticklabels(all_codons, rotation=90, fontsize=7)
            ax_heat.set_yticks(range(len(heat_data.columns)))
            ax_heat.set_yticklabels(heat_data.columns, fontsize=9)
            ax_heat.set_title("Per-Codon Strand Asymmetry (+ minus − strand RSCU)", fontsize=11)
            plt.colorbar(im, ax=ax_heat, shrink=0.8, label="ΔRSCU (plus − minus)")

        # Create lollipop plots for each pair
        for pair_idx, (cond_a, cond_b, pair_df) in enumerate(pair_list):
            ax_lollipop = fig.add_subplot(gs[1, pair_idx])
            pair_df = pair_df.sort_values("diff", key=abs, ascending=False)

            x = range(len(pair_df))
            sig_mask = pair_df["significant"].values
            colors_lollipop = ["#d62728" if s else "#cccccc" for s in sig_mask]
            ax_lollipop.vlines(x, 0, pair_df["diff"].values, colors=colors_lollipop, linewidth=1.5)
            ax_lollipop.scatter(x, pair_df["diff"].values, c=colors_lollipop, s=30, zorder=3)
            ax_lollipop.axhline(0, color="black", linewidth=0.5)
            ax_lollipop.set_xticks(list(x))
            ax_lollipop.set_xticklabels(pair_df["codon"].values, rotation=90, fontsize=7)
            ax_lollipop.set_ylabel(f"Asymmetry Difference", fontsize=9)
            ax_lollipop.set_title(
                f"{cond_a} vs {cond_b}  "
                f"(red = FDR < 0.05, {sig_mask.sum()}/{len(sig_mask)} significant)",
                fontsize=10,
            )

    else:
        # Legacy 2-condition format
        # Identify condition columns
        asym_cols = [c for c in df.columns if c.startswith("mean_asymmetry_")]
        if len(asym_cols) < 2:
            return

        cond_a = asym_cols[0].replace("mean_asymmetry_", "")
        cond_b = asym_cols[1].replace("mean_asymmetry_", "")

        df = df.sort_values("diff", key=abs, ascending=False)

        fig, (ax_heat, ax_lollipop) = plt.subplots(
            2, 1, figsize=(max(12, len(df) * 0.35), 10),
            gridspec_kw={"height_ratios": [1, 2]},
        )

        # -- Heatmap --
        heat_data = df[[asym_cols[0], asym_cols[1]]].T
        heat_data.columns = df["codon"].values
        heat_data.index = [cond_a, cond_b]
        vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()))
        im = ax_heat.imshow(heat_data.values, aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax)
        ax_heat.set_xticks(range(len(heat_data.columns)))
        ax_heat.set_xticklabels(heat_data.columns, rotation=90, fontsize=7)
        ax_heat.set_yticks(range(2))
        ax_heat.set_yticklabels([cond_a, cond_b], fontsize=10)
        ax_heat.set_title("Per-Codon Strand Asymmetry (+ minus − strand RSCU)", fontsize=11)
        plt.colorbar(im, ax=ax_heat, shrink=0.6, label="ΔRSCU (plus − minus)")

        # -- Lollipop --
        x = range(len(df))
        sig_mask = df["significant"].values
        colors_lollipop = ["#d62728" if s else "#cccccc" for s in sig_mask]
        ax_lollipop.vlines(x, 0, df["diff"].values, colors=colors_lollipop, linewidth=1.5)
        ax_lollipop.scatter(x, df["diff"].values, c=colors_lollipop, s=30, zorder=3)
        ax_lollipop.axhline(0, color="black", linewidth=0.5)
        ax_lollipop.set_xticks(list(x))
        ax_lollipop.set_xticklabels(df["codon"].values, rotation=90, fontsize=7)
        ax_lollipop.set_ylabel(f"Asymmetry Difference ({cond_a} − {cond_b})", fontsize=10)
        ax_lollipop.set_title(
            f"Condition Difference in Strand Asymmetry  "
            f"(red = FDR < 0.05, {sig_mask.sum()}/{len(sig_mask)} significant)",
            fontsize=11,
        )

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_neutrality_scatter_by_condition(
    gc3_gc12_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
) -> None:
    """GC3 vs GC12 scatter colored by condition with separate regression lines.

    Shows mutational bias (neutrality) vs selection pressure across conditions.
    Points are per-gene values pooled from all samples within each condition.
    """
    _apply_style()
    if gc3_gc12_df.empty or "condition" not in gc3_gc12_df.columns:
        return
    if "GC3" not in gc3_gc12_df.columns or "GC12" not in gc3_gc12_df.columns:
        return

    conditions = sorted(gc3_gc12_df["condition"].unique())
    colors = _condition_colors(conditions)

    fig, ax = plt.subplots(figsize=(8, 7))

    for cond in conditions:
        sub = gc3_gc12_df[gc3_gc12_df["condition"] == cond]
        ax.scatter(sub["GC3"], sub["GC12"], c=colors[cond], alpha=0.15, s=8,
                   label=f"{cond} (n={len(sub):,})", edgecolors="none")

        # Regression line
        if len(sub) > 10 and sub["GC3"].nunique() > 1:
            slope, intercept, r, p_val, _ = sp_stats.linregress(sub["GC3"], sub["GC12"])
            x_line = np.linspace(sub["GC3"].min(), sub["GC3"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=colors[cond],
                    linewidth=2, label=f"  slope={slope:.3f}, r={r:.3f}")

    # Reference: neutrality line (slope=1)
    gc_range = np.linspace(gc3_gc12_df["GC3"].min(), gc3_gc12_df["GC3"].max(), 100)
    mean_gc12 = gc3_gc12_df["GC12"].mean()
    ax.plot(gc_range, gc_range * 1.0 + (mean_gc12 - gc3_gc12_df["GC3"].mean()),
            "k--", linewidth=1, alpha=0.4, label="Neutral expectation (slope=1)")

    ax.set_xlabel("GC3 (third codon position GC content)", fontsize=11)
    ax.set_ylabel("GC12 (first + second position GC content)", fontsize=11)
    ax.set_title("Neutrality Plot by Condition", fontsize=12)
    ax.legend(fontsize=8, loc="best", ncol=1)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# EXPRESSION-CLASS RSCU & ENRICHMENT COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════


def plot_expression_class_rscu_volcano(
    rscu_tests_df: pd.DataFrame,
    gene_set_label: str,
    output_path: Path,
) -> None:
    """Volcano plot for expression-class-specific RSCU differences between conditions.

    X-axis: log2 fold-change, Y-axis: -log10(adjusted p-value).
    Significant codons (FDR < 0.05, |log2FC| > 0.3) are labeled.
    For 3+ conditions, creates multi-panel figure (one subplot per pair).
    """
    _apply_style()
    if rscu_tests_df.empty or "p_adjusted" not in rscu_tests_df.columns:
        return

    df = rscu_tests_df.copy()
    df["-log10_padj"] = -np.log10(df["p_adjusted"].clip(lower=1e-50))

    # Check for new format (group1/group2 columns for pairwise results)
    if "group1" in df.columns and "group2" in df.columns:
        # Multi-pair format: group by pair
        pair_list = []
        for (g1, g2), group_data in df.groupby(["group1", "group2"], sort=False):
            pair_list.append((g1, g2, group_data))

        n_pairs = len(pair_list)
        fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 6))
        if n_pairs == 1:
            axes = [axes]

        for ax, (cond_a, cond_b, pair_df) in zip(axes, pair_list):
            sig_mask = pair_df["significant"] & (pair_df["log2_fc"].abs() > 0.3)
            nonsig = pair_df[~sig_mask]
            sig = pair_df[sig_mask]

            ax.scatter(nonsig["log2_fc"], nonsig["-log10_padj"], c="#cccccc", s=30,
                       alpha=0.5, edgecolors="none", label="Not significant")

            # Color by direction
            sig_up = sig[sig["log2_fc"] > 0]
            sig_down = sig[sig["log2_fc"] < 0]

            ax.scatter(sig_up["log2_fc"], sig_up["-log10_padj"], c="#d62728", s=50,
                       alpha=0.8, edgecolors="black", linewidths=0.5, label=f"Higher in {cond_a}")
            ax.scatter(sig_down["log2_fc"], sig_down["-log10_padj"], c="#1f77b4", s=50,
                       alpha=0.8, edgecolors="black", linewidths=0.5, label=f"Higher in {cond_b}")

            # Label significant codons
            for _, row in sig.iterrows():
                codon_label = f"{row['amino_acid']}-{row['codon']}" if row.get("amino_acid") else row["codon"]
                ax.annotate(codon_label, (row["log2_fc"], row["-log10_padj"]),
                            fontsize=7, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points")

            ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(-0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("log₂ Fold-Change", fontsize=11)
            ax.set_ylabel("-log₁₀(adjusted p-value)", fontsize=11)
            title_label = "Ribosomal Protein" if gene_set_label == "ribosomal" else "High-Expression"
            ax.set_title(f"{title_label} ({cond_a} vs {cond_b})", fontsize=11)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.15)
    else:
        # Legacy 2-condition format
        sig_mask = df["significant"] & (df["log2_fc"].abs() > 0.3)
        nonsig = df[~sig_mask]
        sig = df[sig_mask]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(nonsig["log2_fc"], nonsig["-log10_padj"], c="#cccccc", s=30,
                   alpha=0.5, edgecolors="none", label="Not significant")

        # Color by direction
        sig_up = sig[sig["log2_fc"] > 0]
        sig_down = sig[sig["log2_fc"] < 0]
        cond_cols = [c for c in df.columns if c.startswith("mean_")]
        cond_names = [c.replace("mean_", "") for c in cond_cols]
        label_up = f"Higher in {cond_names[0]}" if len(cond_names) >= 1 else "Higher in cond1"
        label_down = f"Higher in {cond_names[1]}" if len(cond_names) >= 2 else "Higher in cond2"

        ax.scatter(sig_up["log2_fc"], sig_up["-log10_padj"], c="#d62728", s=50,
                   alpha=0.8, edgecolors="black", linewidths=0.5, label=label_up)
        ax.scatter(sig_down["log2_fc"], sig_down["-log10_padj"], c="#1f77b4", s=50,
                   alpha=0.8, edgecolors="black", linewidths=0.5, label=label_down)

        # Label significant codons
        for _, row in sig.iterrows():
            codon_label = f"{row['amino_acid']}-{row['codon']}" if row.get("amino_acid") else row["codon"]
            ax.annotate(codon_label, (row["log2_fc"], row["-log10_padj"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(-0.3, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("log₂ Fold-Change", fontsize=11)
        ax.set_ylabel("-log₁₀(adjusted p-value)", fontsize=11)
        title_label = "Ribosomal Protein" if gene_set_label == "ribosomal" else "High-Expression"
        ax.set_title(f"{title_label} RSCU: Condition Differences", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.15)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_expression_class_rscu_heatmap(
    rp_tests_df: pd.DataFrame | None,
    he_tests_df: pd.DataFrame | None,
    output_path: Path,
) -> None:
    """Side-by-side heatmap of log2FC for ribosomal and high-expression RSCU.

    Codons on y-axis, columns per gene set per pair (for 3+ conditions).
    Significant codons marked with asterisks.
    """
    _apply_style()
    dfs = []
    labels = []
    if rp_tests_df is not None and not rp_tests_df.empty and "log2_fc" in rp_tests_df.columns:
        dfs.append(rp_tests_df)
        labels.append("Ribosomal")
    if he_tests_df is not None and not he_tests_df.empty and "log2_fc" in he_tests_df.columns:
        dfs.append(he_tests_df)
        labels.append("High-Expression")
    if not dfs:
        return

    # Get union of codons
    all_codons = set()
    for df in dfs:
        codon_col = "codon_col" if "codon_col" in df.columns else "codon"
        all_codons.update(df[codon_col].values)
    all_codons = sorted(all_codons)

    # Check if data has pairwise structure (group1/group2 columns)
    has_pairs = any("group1" in df.columns and "group2" in df.columns for df in dfs)

    if has_pairs:
        # Multi-pair format: build column per gene set per pair
        heat_data = pd.DataFrame(index=all_codons)
        sig_data = pd.DataFrame(index=all_codons)

        for df, gene_set_label in zip(dfs, labels):
            codon_col = "codon_col" if "codon_col" in df.columns else "codon"
            # Get unique pairs from this dataframe
            pairs = df[["group1", "group2"]].drop_duplicates()
            for _, row in pairs.iterrows():
                g1, g2 = row["group1"], row["group2"]
                pair_data = df[(df["group1"] == g1) & (df["group2"] == g2)]
                col_name = f"{gene_set_label} ({g1} vs {g2})"
                fc_map = dict(zip(pair_data[codon_col], pair_data["log2_fc"]))
                sig_map = dict(zip(pair_data[codon_col], pair_data.get("significant", pd.Series(dtype=bool))))
                heat_data[col_name] = [fc_map.get(c, 0) for c in all_codons]
                sig_data[col_name] = [sig_map.get(c, False) for c in all_codons]

        fig, ax = plt.subplots(figsize=(4 + heat_data.shape[1] * 1.5, max(8, len(all_codons) * 0.22)))
        vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()), 0.1)
        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)

        # Asterisks for significant
        for i in range(len(all_codons)):
            for j in range(heat_data.shape[1]):
                if sig_data.iloc[i, j]:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(heat_data.shape[1]))
        ax.set_xticklabels(heat_data.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_codons)))
        ax.set_yticklabels(all_codons, fontsize=7)
        ax.set_title("Expression-Class RSCU Differences Between Conditions\n(log₂FC, * = FDR < 0.05)",
                     fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.6, label="log₂ Fold-Change")
    else:
        # Legacy 2-condition format
        heat_data = pd.DataFrame(index=all_codons)
        sig_data = pd.DataFrame(index=all_codons)
        codon_col = "codon_col" if "codon_col" in dfs[0].columns else "codon"
        for df, label in zip(dfs, labels):
            fc_map = dict(zip(df[codon_col], df["log2_fc"]))
            sig_map = dict(zip(df[codon_col], df.get("significant", pd.Series(dtype=bool))))
            heat_data[label] = [fc_map.get(c, 0) for c in all_codons]
            sig_data[label] = [sig_map.get(c, False) for c in all_codons]

        fig, ax = plt.subplots(figsize=(4 + len(labels) * 1.5, max(8, len(all_codons) * 0.22)))
        vmax = max(abs(heat_data.values.min()), abs(heat_data.values.max()), 0.1)
        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)

        # Asterisks for significant
        for i in range(len(all_codons)):
            for j in range(len(labels)):
                if sig_data.iloc[i, j]:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticks(range(len(all_codons)))
        ax.set_yticklabels(all_codons, fontsize=7)
        ax.set_title("Expression-Class RSCU Differences Between Conditions\n(log₂FC, * = FDR < 0.05)",
                     fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.6, label="log₂ Fold-Change")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_enrichment_comparison(
    enrichment_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Dot plot comparing pathway enrichment frequency between conditions.

    Each pathway is a row. Dot size = fraction of samples enriched.
    Color = -log10(adjusted p-value). Only pathways enriched in ≥1 sample shown.
    For 3+ conditions, creates one subplot per pair.
    """
    _apply_style()
    if enrichment_df.empty:
        return

    df = enrichment_df.copy()

    # Check for pairwise format (test, group1, group2 columns)
    has_pairs = "test" in df.columns and "group1" in df.columns and "group2" in df.columns

    if has_pairs:
        # Filter to pairwise fisher_exact results
        df = df[df["test"] == "fisher_exact"].copy()
        if df.empty:
            return

        # Get unique pairs
        pair_list = []
        for (g1, g2), group_data in df.groupby(["group1", "group2"], sort=False):
            pair_list.append((g1, g2, group_data))

        n_pairs = len(pair_list)
        fig, axes = plt.subplots(1, n_pairs, figsize=(10 * n_pairs, max(5, 8)))
        if n_pairs == 1:
            axes = [axes]

        for ax, (cond_a, cond_b, pair_df) in zip(axes, pair_list):
            frac_cols = [c for c in pair_df.columns if c.startswith("frac_enriched_")]
            if len(frac_cols) < 2:
                continue

            cond_a_col = next((c for c in frac_cols if cond_a in c), frac_cols[0])
            cond_b_col = next((c for c in frac_cols if cond_b in c), frac_cols[1])

            # Filter to pathways with at least some enrichment
            plot_df = pair_df[(pair_df[cond_a_col] > 0) | (pair_df[cond_b_col] > 0)].copy()
            if plot_df.empty:
                continue

            # Sort by significance
            plot_df = plot_df.sort_values("p_adjusted").head(30)
            plot_df = plot_df.sort_values("p_adjusted", ascending=False)

            y_labels = plot_df["pathway_name"].fillna(plot_df["pathway"]).values
            y_pos = np.arange(len(plot_df))

            # Plot two columns of dots
            offset = 0.15
            colors_palette = _condition_colors([cond_a, cond_b])
            for i, (frac_col, cond, color) in enumerate([
                (cond_a_col, cond_a, colors_palette[cond_a]),
                (cond_b_col, cond_b, colors_palette[cond_b]),
            ]):
                sizes = plot_df[frac_col].values * 300 + 20
                ax.scatter(
                    plot_df[frac_col].values, y_pos + (i - 0.5) * offset,
                    s=sizes, c=color, marker="o" if i == 0 else "s", alpha=0.7,
                    edgecolors="black", linewidths=0.5, label=cond,
                )

            # Mark significant with bold y-label
            sig_mask = plot_df["significant"].values
            for idx, (label, is_sig) in enumerate(zip(y_labels, sig_mask)):
                weight = "bold" if is_sig else "normal"
                ax.text(-0.02, idx, label, ha="right", va="center", fontsize=8,
                        fontweight=weight, transform=ax.get_yaxis_transform())

            ax.set_yticks(y_pos)
            ax.set_yticklabels([""] * len(y_pos))
            ax.set_xlabel("Fraction Enriched (FDR < 0.05)", fontsize=9)
            ax.set_title(f"Pathway Enrichment ({cond_a} vs {cond_b})", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")
            ax.set_xlim(-0.05, 1.05)
            ax.grid(True, axis="x", alpha=0.2)
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    else:
        # Legacy 2-condition format
        frac_cols = [c for c in df.columns if c.startswith("frac_enriched_")]
        if len(frac_cols) < 2:
            return

        cond_a = frac_cols[0].replace("frac_enriched_", "")
        cond_b = frac_cols[1].replace("frac_enriched_", "")

        # Filter to pathways with at least some enrichment
        df = df[(df[frac_cols[0]] > 0) | (df[frac_cols[1]] > 0)]
        if df.empty:
            return

        # Sort by significance
        df = df.sort_values("p_adjusted").head(30)
        df = df.sort_values("p_adjusted", ascending=False)

        y_labels = df["pathway_name"].fillna(df["pathway"]).values
        y_pos = np.arange(len(df))

        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.35)))

        # Plot two columns of dots
        offset = 0.15
        for i, (frac_col, cond, color, marker) in enumerate([
            (frac_cols[0], cond_a, "#1f77b4", "o"),
            (frac_cols[1], cond_b, "#d62728", "s"),
        ]):
            sizes = df[frac_col].values * 300 + 20
            ax.scatter(
                df[frac_col].values, y_pos + (i - 0.5) * offset,
                s=sizes, c=color, marker=marker, alpha=0.7,
                edgecolors="black", linewidths=0.5, label=cond,
            )

        # Mark significant with bold y-label
        sig_mask = df["significant"].values
        for idx, (label, is_sig) in enumerate(zip(y_labels, sig_mask)):
            weight = "bold" if is_sig else "normal"
            ax.text(-0.02, idx, label, ha="right", va="center", fontsize=8,
                    fontweight=weight, transform=ax.get_yaxis_transform())

        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * len(y_pos))
        ax.set_xlabel("Fraction of Samples with Pathway Enriched (FDR < 0.05)", fontsize=10)
        ax.set_title("Pathway Enrichment Comparison Between Conditions\n(bold = significantly different)",
                     fontsize=11)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, axis="x", alpha=0.2)
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    _save_fig(fig, output_path)


# ─── Ribosomal vs high-expression & MGE comparative plots ─────────────────


def plot_rp_he_divergence_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Show RP-vs-HE RSCU divergence per condition.

    For each sample that has both ribosomal (rp_*) and high-expression (he_*)
    RSCU columns, computes the Euclidean distance between the two profiles.
    Violin + strip plot shows whether conditions differ in how much their
    ribosomal and highly expressed genes diverge in codon preference.

    Args:
        metrics_df: Sample-level metrics table with rp_* and he_* RSCU columns.
        condition_col: Column designating conditions.
        output_path: Base path for saving.
    """
    _apply_style()
    rp_cols = sorted([c for c in metrics_df.columns if c.startswith("rp_")])
    he_cols = sorted([c for c in metrics_df.columns if c.startswith("he_")])

    if len(rp_cols) < 10 or len(he_cols) < 10:
        return

    # Match columns by codon name (rp_Phe-UUU ↔ he_Phe-UUU)
    rp_suffix = {c[3:]: c for c in rp_cols}
    he_suffix = {c[3:]: c for c in he_cols}
    shared = sorted(set(rp_suffix) & set(he_suffix))
    if len(shared) < 10:
        return

    # Compute per-sample Euclidean distance
    dists = []
    for _, row in metrics_df.iterrows():
        rp_vec = np.array([row.get(rp_suffix[s], np.nan) for s in shared])
        he_vec = np.array([row.get(he_suffix[s], np.nan) for s in shared])
        mask = np.isfinite(rp_vec) & np.isfinite(he_vec)
        if mask.sum() >= 10:
            d = np.sqrt(np.sum((rp_vec[mask] - he_vec[mask]) ** 2))
            dists.append(d)
        else:
            dists.append(np.nan)

    plot_df = metrics_df[[condition_col, "sample_id"]].copy()
    plot_df["rp_he_euclidean"] = dists
    plot_df = plot_df.dropna(subset=["rp_he_euclidean"])

    if plot_df.empty or plot_df[condition_col].nunique() < 2:
        return

    conditions = sorted(plot_df[condition_col].unique())
    palette = _condition_colors(conditions)

    fig, ax = plt.subplots(figsize=(max(5, len(conditions) * 1.5), 5))
    sns.violinplot(
        data=plot_df, x=condition_col, y="rp_he_euclidean",
        hue=condition_col, hue_order=conditions,
        order=conditions, palette=palette, inner=None, cut=0,
        alpha=0.3, ax=ax, legend=False,
    )
    sns.stripplot(
        data=plot_df, x=condition_col, y="rp_he_euclidean",
        hue=condition_col, hue_order=conditions,
        order=conditions, palette=palette, size=5, alpha=0.7,
        jitter=0.15, ax=ax, legend=False,
    )

    # Kruskal-Wallis across conditions
    groups = [g["rp_he_euclidean"].dropna().values for _, g in plot_df.groupby(condition_col)]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        stat, pval = sp_stats.kruskal(*groups)
        ax.set_title(f"Kruskal-Wallis p = {pval:.3e}", fontsize=9, fontstyle="italic")

    ax.set_ylabel("RP–HE RSCU Euclidean distance")
    ax.set_xlabel("")
    fig.suptitle("Ribosomal vs high-expression codon usage divergence", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_grodon2_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Compare gRodon2 doubling times across conditions.

    Two-panel figure:
      Left: violin+strip of gRodon2 doubling time by condition
      Right: scatter of CUBHE vs ConsistencyHE colored by condition

    Args:
        metrics_df: Combined metrics table with grodon2_* columns.
        condition_col: Condition column name.
        output_path: Base path for saving.
    """
    _apply_style()

    has_d = "grodon2_doubling_time_hours" in metrics_df.columns
    has_cubhe = "grodon2_CUBHE" in metrics_df.columns and "grodon2_ConsistencyHE" in metrics_df.columns

    if not has_d:
        return

    df = metrics_df.dropna(subset=["grodon2_doubling_time_hours"])
    if df.empty:
        return

    n_panels = 2 if has_cubhe else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    conditions = df[condition_col].unique()
    palette = sns.color_palette("Set2", len(conditions))

    # Panel 1: doubling time violin + strip
    ax = axes[0]
    sns.violinplot(data=df, x=condition_col, y="grodon2_doubling_time_hours",
                   hue=condition_col, palette=palette, inner=None, alpha=0.4,
                   ax=ax, legend=False)
    sns.stripplot(data=df, x=condition_col, y="grodon2_doubling_time_hours",
                  hue=condition_col, palette=palette, size=6, edgecolor="black",
                  linewidth=0.5, ax=ax, legend=False)

    ax.axhline(2, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(5, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)

    # Kruskal-Wallis if ≥2 conditions with ≥2 samples each
    groups = [g["grodon2_doubling_time_hours"].dropna().values
              for _, g in df.groupby(condition_col)
              if len(g["grodon2_doubling_time_hours"].dropna()) >= 2]
    if len(groups) >= 2:
        from scipy.stats import kruskal
        try:
            stat, pval = kruskal(*groups)
            ax.set_title(f"Kruskal-Wallis p = {pval:.3g}", fontsize=10)
        except Exception as e:
            logger.debug("Kruskal-Wallis test failed: %s", e)

    ax.set_ylabel("gRodon2 Doubling Time (h)", fontsize=10)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: CUBHE vs ConsistencyHE
    if has_cubhe and n_panels == 2:
        ax2 = axes[1]
        for i, cond in enumerate(conditions):
            cond_df = df[df[condition_col] == cond]
            ax2.scatter(cond_df["grodon2_CUBHE"], cond_df["grodon2_ConsistencyHE"],
                        c=[palette[i]], label=cond, s=60, edgecolors="black", linewidth=0.5, zorder=3)
        ax2.set_xlabel("CUBHE", fontsize=10)
        ax2.set_ylabel("ConsistencyHE", fontsize=10)
        ax2.set_title("CUB Landscape by Condition", fontsize=10)
        ax2.legend(fontsize=8, frameon=True)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("gRodon2 Growth Rate by Condition", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_path)


def plot_mge_cu_deviation_by_condition(
    metrics_df: pd.DataFrame,
    condition_col: str,
    output_path: Path,
):
    """Compare codon usage deviation of MGE/HGT genes across conditions.

    Three-panel figure:
    - Left: mean Mahalanobis distance per sample (genome heterogeneity)
    - Center: fraction of HGT-flagged genes
    - Right: count of mobilome/phage genes

    Args:
        metrics_df: Sample-level metrics with mean_mahalanobis_dist,
            hgt_fraction, n_mobilome, n_phage columns.
        condition_col: Column designating conditions.
        output_path: Base path for saving.
    """
    _apply_style()
    panels = []
    if "mean_mahalanobis_dist" in metrics_df.columns:
        panels.append(("mean_mahalanobis_dist", "Mean Mahalanobis\ndistance"))
    if "hgt_fraction" in metrics_df.columns:
        panels.append(("hgt_fraction", "HGT gene fraction"))
    if "n_mobilome" in metrics_df.columns:
        panels.append(("n_mobilome", "Mobilome genes"))
    if "n_phage" in metrics_df.columns:
        panels.append(("n_phage", "Phage-related genes"))

    if not panels or condition_col not in metrics_df.columns:
        return

    conditions = sorted(metrics_df[condition_col].dropna().unique())
    if len(conditions) < 2:
        return

    palette = _condition_colors(conditions)
    n_panels = min(len(panels), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, panels[:n_panels]):
        plot_data = metrics_df.dropna(subset=[col, condition_col])
        if plot_data.empty:
            ax.set_visible(False)
            continue

        sns.boxplot(
            data=plot_data, x=condition_col, y=col, order=conditions,
            hue=condition_col, hue_order=conditions, palette=palette,
            ax=ax, linewidth=0.8, fliersize=3, legend=False,
        )
        sns.stripplot(
            data=plot_data, x=condition_col, y=col, order=conditions,
            color="black", size=4, alpha=0.5, jitter=0.15, ax=ax,
        )
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

        # Kruskal-Wallis
        groups = [g[col].dropna().values for _, g in plot_data.groupby(condition_col)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            _, pval = sp_stats.kruskal(*groups)
            ax.set_title(f"p = {pval:.3e}", fontsize=8, fontstyle="italic")

    fig.suptitle("MGE / HGT codon usage deviation by condition", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_comparative_plots(
    metrics_df: pd.DataFrame,
    output_dir: Path,
    condition_col: str = "condition",
    between_tests_df: pd.DataFrame | None = None,
    rscu_tests_df: pd.DataFrame | None = None,
    rscu_disp_df: pd.DataFrame | None = None,
    perm_result: dict | None = None,
    hgt_burden: dict | None = None,
    strand_asym_patterns_df: pd.DataFrame | None = None,
    optimal_codons_df: pd.DataFrame | None = None,
    gc3_gc12_df: pd.DataFrame | None = None,
    rp_rscu_tests_df: pd.DataFrame | None = None,
    he_rscu_tests_df: pd.DataFrame | None = None,
    enrichment_comp_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate all within- and between-condition comparative plots.

    Returns dict of plot paths.
    """
    plot_dir = output_dir / "batch_condition" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    if condition_col not in metrics_df.columns:
        logger.info("SKIPPED: comparative plots (no condition column)")
        return outputs

    conditions = metrics_df[condition_col].dropna().unique()
    logger.info("Generating comparative plots for %d conditions", len(conditions))

    # ── Within-condition ──────────────────────────────────────────────
    try:
        p = plot_dir / "within_metric_violins"
        plot_within_metric_violins(metrics_df, condition_col, p)
        outputs["within_metric_violins"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Failed: within_metric_violins — %s", e)

    try:
        p = plot_dir / "within_rscu_heatmap"
        plot_within_rscu_heatmap(metrics_df, condition_col, p)
        outputs["within_rscu_heatmap"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Failed: within_rscu_heatmap — %s", e)

    if rscu_disp_df is not None and not rscu_disp_df.empty:
        try:
            p = plot_dir / "within_rscu_cv"
            plot_within_rscu_cv(rscu_disp_df, p)
            outputs["within_rscu_cv"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: within_rscu_cv — %s", e)

    try:
        p = plot_dir / "within_pca_ellipse"
        plot_within_pca(metrics_df, condition_col, p)
        outputs["within_pca_ellipse"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Failed: within_pca_ellipse — %s", e)

    try:
        p = plot_dir / "radar_by_condition"
        plot_radar_by_condition(metrics_df, condition_col, p)
        outputs["radar_by_condition"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Failed: radar_by_condition — %s", e)

    # ── Between-condition ─────────────────────────────────────────────
    if len(conditions) >= 2:
        if between_tests_df is not None and not between_tests_df.empty:
            try:
                p = plot_dir / "between_metric_comparison"
                plot_between_metric_comparison(metrics_df, between_tests_df, condition_col, p)
                outputs["between_metric_comparison"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: between_metric_comparison — %s", e)

        if rscu_tests_df is not None and not rscu_tests_df.empty:
            try:
                p = plot_dir / "rscu_volcano"
                plot_rscu_volcano(rscu_tests_df, p)
                outputs["rscu_volcano"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: rscu_volcano — %s", e)

            try:
                p = plot_dir / "rscu_condition_heatmap"
                plot_rscu_condition_heatmap(rscu_tests_df, p)
                outputs["rscu_condition_heatmap"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: rscu_condition_heatmap — %s", e)

        try:
            p = plot_dir / "enc_gc3_by_condition"
            plot_enc_gc3_by_condition(metrics_df, condition_col, p)
            outputs["enc_gc3_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: enc_gc3_by_condition — %s", e)

        try:
            p = plot_dir / "growth_rate_by_condition"
            plot_growth_rate_by_condition(metrics_df, condition_col, p)
            outputs["growth_rate_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: growth_rate_by_condition — %s", e)

        try:
            p = plot_dir / "hgt_by_condition"
            plot_hgt_by_condition(metrics_df, condition_col, p)
            outputs["hgt_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: hgt_by_condition — %s", e)

        try:
            p = plot_dir / "neutrality_by_condition"
            plot_neutrality_by_condition(metrics_df, condition_col, p)
            outputs["neutrality_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: neutrality_by_condition — %s", e)

        if perm_result:
            try:
                p = plot_dir / "permanova_summary"
                plot_permanova_summary(perm_result, p)
                outputs["permanova_summary"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: permanova_summary — %s", e)

        # ── Enhanced comparative plots ───────────────────────────────
        if between_tests_df is not None and not between_tests_df.empty:
            try:
                p = plot_dir / "effect_size_forest"
                plot_effect_size_forest(between_tests_df, p)
                outputs["effect_size_forest"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: effect_size_forest — %s", e)

        if rscu_tests_df is not None and not rscu_tests_df.empty:
            try:
                p = plot_dir / "rscu_paired_dot"
                plot_rscu_paired_dot(rscu_tests_df, p)
                outputs["rscu_paired_dot"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: rscu_paired_dot — %s", e)

        try:
            p = plot_dir / "condition_summary_dashboard"
            plot_condition_summary_dashboard(
                metrics_df, condition_col, between_tests_df, perm_result, p,
            )
            outputs["condition_summary_dashboard"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: condition_summary_dashboard — %s", e)

        try:
            p = plot_dir / "translational_selection_comparison"
            plot_translational_selection_comparison(
                metrics_df, condition_col, between_tests_df, p,
            )
            outputs["translational_selection_comparison"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: translational_selection_comparison — %s", e)

        try:
            p = plot_dir / "strand_asymmetry_comparison"
            plot_strand_asymmetry_comparison(
                metrics_df, condition_col, between_tests_df, p,
            )
            outputs["strand_asymmetry_comparison"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: strand_asymmetry_comparison — %s", e)

        try:
            p = plot_dir / "operon_coadaptation_comparison"
            plot_operon_coadaptation_comparison(
                metrics_df, condition_col, between_tests_df, p,
            )
            outputs["operon_coadaptation_comparison"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: operon_coadaptation_comparison — %s", e)

        # ── Bio/ecology between-condition plots ───────────────────────
        if hgt_burden:
            try:
                p = plot_dir / "hgt_burden_comparison"
                plot_hgt_burden_comparison(hgt_burden, metrics_df, condition_col, p)
                outputs["hgt_burden_comparison"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: hgt_burden_comparison — %s", e)

        try:
            p = plot_dir / "phage_mobile_comparison"
            plot_phage_mobile_comparison(metrics_df, condition_col, between_tests_df, p)
            outputs["phage_mobile_comparison"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: phage_mobile_comparison — %s", e)

        if optimal_codons_df is not None and not optimal_codons_df.empty:
            try:
                p = plot_dir / "optimal_codon_divergence"
                plot_optimal_codon_divergence(optimal_codons_df, p)
                outputs["optimal_codon_divergence"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: optimal_codon_divergence — %s", e)

        if strand_asym_patterns_df is not None and not strand_asym_patterns_df.empty:
            try:
                p = plot_dir / "strand_asymmetry_pattern_comparison"
                plot_strand_asymmetry_pattern_comparison(strand_asym_patterns_df, p)
                outputs["strand_asymmetry_pattern_comparison"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: strand_asymmetry_pattern_comparison — %s", e)

        if gc3_gc12_df is not None and not gc3_gc12_df.empty:
            try:
                p = plot_dir / "neutrality_scatter_by_condition"
                plot_neutrality_scatter_by_condition(gc3_gc12_df, metrics_df, condition_col, p)
                outputs["neutrality_scatter_by_condition"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: neutrality_scatter_by_condition — %s", e)

        # ── Expression-class RSCU & enrichment plots ─────────────────
        if rp_rscu_tests_df is not None and not rp_rscu_tests_df.empty:
            try:
                p = plot_dir / "ribosomal_rscu_volcano"
                plot_expression_class_rscu_volcano(rp_rscu_tests_df, "ribosomal", p)
                outputs["ribosomal_rscu_volcano"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: ribosomal_rscu_volcano — %s", e)

        if he_rscu_tests_df is not None and not he_rscu_tests_df.empty:
            try:
                p = plot_dir / "high_expression_rscu_volcano"
                plot_expression_class_rscu_volcano(he_rscu_tests_df, "high_expression", p)
                outputs["high_expression_rscu_volcano"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: high_expression_rscu_volcano — %s", e)

        if rp_rscu_tests_df is not None or he_rscu_tests_df is not None:
            try:
                p = plot_dir / "expression_class_rscu_heatmap"
                plot_expression_class_rscu_heatmap(rp_rscu_tests_df, he_rscu_tests_df, p)
                outputs["expression_class_rscu_heatmap"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: expression_class_rscu_heatmap — %s", e)

        if enrichment_comp_df is not None and not enrichment_comp_df.empty:
            try:
                p = plot_dir / "enrichment_comparison"
                plot_enrichment_comparison(enrichment_comp_df, p)
                outputs["enrichment_comparison"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: enrichment_comparison — %s", e)

        # ── RP vs HE divergence by condition ─────────────────────────
        try:
            p = plot_dir / "rp_he_divergence_by_condition"
            plot_rp_he_divergence_by_condition(metrics_df, condition_col, p)
            outputs["rp_he_divergence_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: rp_he_divergence_by_condition — %s", e)

        # ── gRodon2 by condition ────────────────────────────────────
        if "grodon2_doubling_time_hours" in metrics_df.columns:
            try:
                p = plot_dir / "grodon2_by_condition"
                plot_grodon2_by_condition(metrics_df, condition_col, p)
                outputs["grodon2_by_condition"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Failed: grodon2_by_condition — %s", e)

        # ── MGE codon usage deviation by condition ───────────────────
        try:
            p = plot_dir / "mge_cu_deviation_by_condition"
            plot_mge_cu_deviation_by_condition(metrics_df, condition_col, p)
            outputs["mge_cu_deviation_by_condition"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Failed: mge_cu_deviation_by_condition — %s", e)

    logger.info("Generated %d comparative plots", len(outputs))
    return outputs
