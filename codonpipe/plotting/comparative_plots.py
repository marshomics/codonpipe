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


DPI = 300
FORMATS = ["png", "svg"]


def _apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "svg.fonttype": "none",
    })


def _save_fig(fig, base_path: Path, dpi: int = DPI):
    for fmt in FORMATS:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

    for j in range(i + 1, len(axes)):
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

    df = metrics_df.dropna(subset=rscu_cols + [condition_col])
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

    for i, metric in enumerate(sig_metrics):
        ax = axes[i]
        data = metrics_df.dropna(subset=[metric, condition_col])
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

    for j in range(i + 1, len(axes)):
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

        ax.axhline(-np.log10(0.05), color="blue", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(0.3, color="gray", linestyle=":", alpha=0.4)
        ax.axvline(-0.3, color="gray", linestyle=":", alpha=0.4)
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
        df = metrics_df.dropna(subset=rscu_cols + [condition_col])
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
) -> dict[str, Path]:
    """Generate all within- and between-condition comparative plots.

    Returns dict of plot paths.
    """
    plot_dir = output_dir / "comparative" / "plots"
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

    logger.info("Generated %d comparative plots", len(outputs))
    return outputs
