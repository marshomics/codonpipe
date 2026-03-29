"""Publication-ready plotting module for codon usage analysis.

Generates figures at 300 DPI in PNG and SVG 1.1 format.
SVG output uses embedded fonts for full editability in Adobe Illustrator.
All plots use a consistent style suitable for journal submission.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

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
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from codonpipe.utils.codon_tables import AMINO_ACID_FAMILIES, RSCU_COLUMN_NAMES

logger = logging.getLogger("codonpipe")


def _safe_label(value, fallback: str = "", maxlen: int = 50) -> str:
    """Return a truncated string label, safely handling NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return str(fallback)[:maxlen] if fallback else ""
    return str(value)[:maxlen]

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
    # SVG settings: embed fonts so glyphs render correctly in Illustrator
    # without requiring the font to be installed on the editing machine.
    "svg.fonttype": "none",  # output text as <text> elements (editable in AI)
}

DPI = 300
FORMATS = ["png", "svg"]


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

# Amino-acid property grouping for the rounded-cell RSCU heatmap
_THREE_TO_ONE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}
# Also handle split-family names (Ser4, Ser2, Leu4, Leu2, Arg4, Arg2)
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


def plot_rscu_heatmap_rounded(
    freq_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Rounded-cell RSCU heatmap with blue→white→red diverging palette centred at 1.0.

    Each amino acid is a row (grouped by biochemical property), each synonymous
    codon is a rounded tile whose colour encodes the RSCU value.  Values and
    codon labels are printed inside each cell.

    Args:
        freq_df: Output from ``compute_codon_frequency_table()`` with columns
            ``codon``, ``amino_acid``, ``rscu``.
        output_path: Base path for saving (extensions appended).
        sample_id: Sample identifier for the title.
    """
    _apply_style()
    if freq_df is None or freq_df.empty or "rscu" not in freq_df.columns:
        return

    # Exclude stop codons and single-codon families (Met, Trp)
    df = freq_df.dropna(subset=["rscu"]).copy()
    df = df[~df["amino_acid"].isin(("*", "Met", "Trp"))]
    if df.empty:
        return

    # Map amino acid names to single-letter codes
    df["AA"] = df["amino_acid"].map(_THREE_TO_ONE)
    df = df.dropna(subset=["AA"])

    # Order amino acids by biochemical group
    group_order = ["Nonpolar", "Polar", "Positive", "Negative"]
    aa_ordered = []
    present = set(df["AA"].unique())
    for grp in group_order:
        aa_ordered.extend(aa for aa in _AA_ORDER if _AA_TO_GROUP.get(aa) == grp and aa in present)

    if not aa_ordered:
        return

    # Diverging colormap: blue (low) → white (1.0) → red (high)
    max_rscu = df["RSCU"].max() if "RSCU" in df.columns else df["rscu"].max()
    rscu_col = "RSCU" if "RSCU" in df.columns else "rscu"
    vmax = min(max(max_rscu * 1.05, 3.0), 5.0)
    colors_below = plt.cm.Blues_r(np.linspace(0.0, 0.7, 128))
    colors_above = plt.cm.Reds(np.linspace(0.0, 0.75, 128))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rscu_diverging", np.vstack([colors_below, colors_above]),
    )
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)

    # Layout
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

    # Draw cells
    for aa in aa_ordered:
        sub = df[df["AA"] == aa].sort_values("codon")
        y = y_positions[aa]
        for j, (_, row) in enumerate(sub.iterrows()):
            rscu = row[rscu_col]
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

    # Y-axis labels
    for aa in aa_ordered:
        y = y_positions[aa]
        grp = _AA_TO_GROUP.get(aa, "")
        name = _AA_NAMES.get(aa, aa)
        ax.text(
            -0.3, y + 0.075 + cell_h / 2, f"{name} ({aa})",
            ha="right", va="center", fontsize=8.5, fontweight="bold",
            color=_GROUP_COLORS.get(grp, "#333333"),
        )

    # Group labels on far left
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

    title = "Relative Synonymous Codon Usage (RSCU)"
    if sample_id:
        title += f"\n{sample_id}"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Colorbar
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

    _save_fig(fig, output_path)


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
    score_candidates = [c for c in encprime_df.columns if c not in ("gene", "width")]
    if not score_candidates:
        logger.warning("ENCprime DataFrame has no score column; skipping plot")
        return
    score_col = score_candidates[0]

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

    score_candidates = [c for c in milc_df.columns if c not in ("gene", "width")]
    if not score_candidates:
        logger.warning("MILC DataFrame has no score column; skipping plot")
        return
    score_col = score_candidates[0]
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
    # NaN in RSCU means unobserved amino-acid family — treat as zero usage
    # for distance computation so hierarchical clustering doesn't crash.
    data = np.nan_to_num(data, nan=0.0)
    # Cap at 500 genes for readability (deterministic for reproducibility)
    if len(data) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(data), 500, replace=False)
        data = data[idx]

    fig = plt.figure(figsize=(14, min(10, max(5, len(data) // 30))))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Clustering large matrix")
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
    # fillna(0.0): NaN RSCU = unobserved family → zero for dimensionality reduction
    data = df[rscu_cols].fillna(0.0)
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

    pct1 = pca.explained_variance_ratio_[0] * 100
    pct2 = pca.explained_variance_ratio_[1] * 100
    cumulative = pct1 + pct2
    ax.set_xlabel(f"PC1 ({pct1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pct2:.1f}%)")
    ax.set_title(f"{title} (PC1+PC2 = {cumulative:.1f}% of variance)")
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
    # fillna(0.0): NaN RSCU = unobserved family → zero for dimensionality reduction
    data = df[rscu_cols].fillna(0.0)
    if len(data) < 10:
        logger.warning("Too few samples (%d) for UMAP", len(data))
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    min_dist = 0.1  # default UMAP parameter
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value")
        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, random_state=42)
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

    # Annotation with UMAP parameters
    ax.annotate(f"n_neighbors={n_neighbors}, min_dist={min_dist}",
                xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7, color="gray", fontstyle="italic")

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
            data=plot_data, x=group_col, y=col, hue=group_col,
            order=order, hue_order=order,
            ax=ax, fliersize=1, linewidth=0.8,
            palette="Set2", legend=False,
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
    # fillna(0.0): NaN RSCU = unobserved family → zero usage for clustering
    data = df[rscu_cols].fillna(0.0)
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

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Clustering large matrix")
        g = sns.clustermap(
            data,
            cmap="RdYlGn_r",
            center=1.0,
            method="complete",
            metric="cityblock",
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
        lambda r: _safe_label(r.get("pathway_name"), fallback=r.get("pathway", ""), maxlen=50),
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

    _axis1 = coa_inertia.loc[coa_inertia["axis"] == 1, "pct_inertia"].values
    _axis2 = coa_inertia.loc[coa_inertia["axis"] == 2, "pct_inertia"].values
    pct1 = _axis1[0] if len(_axis1) > 0 else 0
    pct2 = _axis2[0] if len(_axis2) > 0 else 0

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
    if len(valid) > 10 and valid[metric].nunique() > 1:
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
    if len(valid) > 10 and valid["GC3"].nunique() > 1:
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
    label_text = "high-expression genes" if rscu_col == "rscu_high_expr" else "all genes"
    ax.set_xlabel("tRNA Gene Copy Number")
    ax.set_ylabel(f"RSCU ({label_text})")
    if len(valid) > 5 and valid["tRNA_copy_number"].nunique() > 1 and valid[rscu_col].nunique() > 1:
        r, p_val = stats.spearmanr(valid["tRNA_copy_number"], valid[rscu_col])
        ax.set_title(f"r = {r:.3f}, p = {p_val:.2e}")

        # Regression line
        slope, intercept, _, _, _ = stats.linregress(valid["tRNA_copy_number"], valid[rscu_col])
        x_line = np.linspace(valid["tRNA_copy_number"].min(), valid["tRNA_copy_number"].max(), 50)
        ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=1, alpha=0.7)

    # Plot 2: compare high vs low expression (if available)
    if len(axes) > 1 and "rscu_low_expr" in trna_corr_df.columns:
        ax = axes[1]
        df2 = trna_corr_df.dropna(subset=["tRNA_copy_number", "rscu_high_expr", "rscu_low_expr"])
        valid2 = df2[df2["tRNA_copy_number"] > 0]
        if len(valid2) > 5 and valid2["tRNA_copy_number"].nunique() > 1:
            r_hi, p_hi = stats.spearmanr(valid2["tRNA_copy_number"], valid2["rscu_high_expr"])
            r_lo, p_lo = stats.spearmanr(valid2["tRNA_copy_number"], valid2["rscu_low_expr"])
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
        lambda r: f"{r['COG_category']} — {_safe_label(r.get('description'), maxlen=40)}"
        if pd.notna(r.get("description")) and r.get("description")
        else str(r.get("COG_category", "")),
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
        r, p_val = stats.spearmanr(length_kb, valid[metric])
        ax.set_title(f"{metric} (r = {r:.3f})")
        ax.set_xlabel("Gene Length (kb)")
        ax.set_ylabel(metric)

    if sample_id:
        fig.suptitle(f"Gene Length vs Codon Bias — {sample_id}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


# ─── KO Enrichment Plots ─────────────────────────────────────────────────────


def plot_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    output_path: Path,
    title: str = "KO Enrichment Dotplot",
    max_pathways: int = 20,
    sample_id: str = "",
):
    """Dotplot of KO enrichment results.

    Classic enrichment dot plot with x = gene ratio (k/K), y = pathway name.
    Dot size = gene count (k), color = -log10(fdr).

    Args:
        enrichment_df: DataFrame from enrichment analysis with columns:
            pathway, pathway_name, k (overlap count), K (pathway size),
            fdr, significant.
        output_path: Base path for saving.
        title: Plot title.
        max_pathways: Maximum number of pathways to show.
        sample_id: Sample identifier for title.
    """
    _apply_style()
    if enrichment_df.empty or "fdr" not in enrichment_df.columns:
        return

    # Filter to significant or top results
    sig = enrichment_df[enrichment_df["significant"]] if "significant" in enrichment_df.columns else enrichment_df.iloc[0:0].copy()
    if sig.empty:
        sig = enrichment_df.nlargest(max_pathways, "fdr").copy()
    else:
        sig = sig.nlargest(max_pathways, "fdr")

    if sig.empty:
        logger.warning("No pathways to plot in enrichment dotplot")
        return

    # Calculate gene ratio (k/N): hits in test set / total annotated in test set
    # Enrichment module uses: k=hits, n=pathway size, N=test set size, M=background size
    if "k" not in sig.columns or "N" not in sig.columns:
        logger.warning("SKIPPED: enrichment dotplot (missing k or N columns)")
        return
    sig["gene_ratio"] = sig["k"] / sig["N"].replace(0, np.nan)
    sig["neg_log_fdr"] = -np.log10(sig["fdr"].clip(lower=1e-300))

    # Sort by gene ratio for better visualization
    sig = sig.sort_values("gene_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(sig) * 0.35)))

    # Scatter plot: x = gene_ratio, y = pathway, size = k, color = -log10(fdr)
    scatter = ax.scatter(
        sig["gene_ratio"],
        range(len(sig)),
        s=sig["k"] * 20,  # Scale size by gene count
        c=sig["neg_log_fdr"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Labels and formatting
    ax.set_yticks(range(len(sig)))
    pathway_labels = sig.apply(
        lambda r: _safe_label(r.get("pathway_name"), fallback=r.get("pathway", ""), maxlen=40),
        axis=1,
    ).values
    ax.set_yticklabels(pathway_labels, fontsize=8)
    ax.set_xlabel("Gene Ratio (k/K)", fontsize=11)
    ax.set_ylabel("Pathway", fontsize=11)

    title_str = title
    if sample_id:
        title_str += f" — {sample_id}"
    ax.set_title(title_str, fontsize=12)

    # Colorbar for -log10(FDR)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("-log10(FDR)", fontsize=10)

    # Size legend
    for size_val in [5, 10, 20]:
        ax.scatter([], [], s=size_val * 20, c="gray", alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.legend(
        [plt.scatter([], [], s=s * 20, c="gray", alpha=0.6, edgecolors="black", linewidth=0.5) for s in [5, 10, 20]],
        ["5 genes", "10 genes", "20 genes"],
        scatterpoints=1,
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_enrichment_heatmap(
    enrichment_results: dict[str, pd.DataFrame],
    output_path: Path,
    sample_id: str = "",
    max_pathways: int = 15,
):
    """Heatmap of KO enrichment across multiple comparisons.

    Rows = top pathways (union of significant across all comparisons),
    columns = metric_tier (e.g., "CAI_high", "MELP_low").
    Cell value = -log10(fdr), with significance markers.

    Args:
        enrichment_results: Dict keyed like "enrichment_CAI_high" mapping
            to enrichment DataFrames.
        output_path: Base path for saving.
        sample_id: Sample identifier.
        max_pathways: Maximum number of pathways to include.
    """
    _apply_style()
    if not enrichment_results:
        return

    # Extract metric_tier from keys and prepare data
    comparisons = []
    all_pathways = set()

    for key, edf in enrichment_results.items():
        if edf.empty:
            continue
        # Extract metric_tier from key like "enrichment_CAI_high"
        metric_tier = key.replace("enrichment_", "")
        comparisons.append((metric_tier, edf))

        # Collect significant pathways
        sig = edf[edf["significant"]] if "significant" in edf.columns else edf.iloc[0:0]
        if "pathway" in sig.columns:
            all_pathways.update(sig["pathway"].dropna())

    if not comparisons or not all_pathways:
        logger.warning("No significant pathways for enrichment heatmap")
        return

    # Limit to top pathways
    top_n = min(len(all_pathways), max_pathways)
    all_pathways = list(all_pathways)[:top_n]

    # Build matrix: rows = pathways, columns = comparisons
    matrix = []
    pathway_names = []

    for pathway in all_pathways:
        row = []
        for metric_tier, edf in comparisons:
            match = edf[edf["pathway"] == pathway] if "pathway" in edf.columns else edf.iloc[0:0]
            if not match.empty:
                fdr = match.iloc[0].get("fdr", 1.0)
                neg_log_fdr = -np.log10(max(fdr, 1e-300))
                row.append(neg_log_fdr)
            else:
                row.append(np.nan)  # White for untested
        matrix.append(row)
        pathway_name = _safe_label(pathway, maxlen=30)
        pathway_names.append(pathway_name)

    if not matrix:
        logger.warning("No data for enrichment heatmap")
        return

    df_heatmap = pd.DataFrame(
        matrix,
        index=pathway_names,
        columns=[m for m, _ in comparisons],
    )

    fig, ax = plt.subplots(figsize=(6 + len(comparisons) * 0.5, max(6, len(pathway_names) * 0.3)))

    # Plot heatmap with annotations
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".1f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "-log10(FDR)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=df_heatmap.max().max() if not df_heatmap.empty else 1,
    )

    title_str = "KO Enrichment Heatmap"
    if sample_id:
        title_str += f" — {sample_id}"
    ax.set_title(title_str, fontsize=12)
    ax.set_xlabel("Comparison", fontsize=11)
    ax.set_ylabel("Pathway", fontsize=11)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_enrichment_summary_bar(
    enrichment_results: dict[str, pd.DataFrame],
    output_path: Path,
    sample_id: str = "",
    max_pathways: int = 20,
):
    """Grouped horizontal bar chart of top KO enrichment pathways.

    Shows top enriched pathways across all comparisons, with bars grouped
    by metric_tier and colored by metric.

    Args:
        enrichment_results: Dict keyed like "enrichment_CAI_high".
        output_path: Base path for saving.
        sample_id: Sample identifier.
        max_pathways: Maximum pathways per comparison.
    """
    _apply_style()
    if not enrichment_results:
        return

    # Collect top pathways from each comparison
    data = []
    colors_map = {}

    metric_colors = sns.color_palette("husl", len(enrichment_results))

    for (key, edf), color in zip(enrichment_results.items(), metric_colors):
        if edf.empty:
            continue

        metric_tier = key.replace("enrichment_", "")
        metric = metric_tier.split("_")[0]

        # Get top pathways by significance
        sig = edf[edf["significant"]] if "significant" in edf.columns else edf.iloc[0:0]
        if sig.empty:
            sig = edf.nlargest(max_pathways, "fdr")
        else:
            sig = sig.head(max_pathways)

        for _, row in sig.iterrows():
            pathway = _safe_label(row.get("pathway"), maxlen=35)
            pathway_name = _safe_label(row.get("pathway_name"), fallback=pathway, maxlen=35)
            fdr = row.get("fdr", 1.0)
            neg_log_fdr = -np.log10(max(fdr, 1e-300))

            data.append({
                "pathway": pathway_name,
                "metric_tier": metric_tier,
                "neg_log_fdr": neg_log_fdr,
                "color": color,
            })
            colors_map[metric_tier] = color

    if not data:
        logger.warning("No pathways for enrichment summary bar")
        return

    df_plot = pd.DataFrame(data)
    unique_pathways = df_plot["pathway"].unique()

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(unique_pathways) * 0.35)))

    y_pos = np.arange(len(unique_pathways))
    bar_height = 0.15
    tiers = df_plot["metric_tier"].unique()

    for i, tier in enumerate(sorted(tiers)):
        tier_data = df_plot[df_plot["metric_tier"] == tier]
        tier_dict = dict(zip(tier_data["pathway"], tier_data["neg_log_fdr"]))

        values = [tier_dict.get(p, 0) for p in unique_pathways]
        color = colors_map.get(tier, "gray")

        ax.barh(
            y_pos + i * bar_height,
            values,
            bar_height,
            label=tier,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )

    # FDR threshold line
    fdr_line = -np.log10(0.05)
    ax.axvline(fdr_line, color="red", linestyle="--", linewidth=1, alpha=0.7, label="FDR = 0.05")

    ax.set_yticks(y_pos + bar_height * (len(tiers) - 1) / 2)
    ax.set_yticklabels(unique_pathways, fontsize=8)
    ax.set_xlabel("-log10(FDR)", fontsize=11)
    ax.set_ylabel("Pathway", fontsize=11)

    title_str = "KO Enrichment Summary"
    if sample_id:
        title_str += f" — {sample_id}"
    ax.set_title(title_str, fontsize=12)

    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _save_fig(fig, output_path)


# ─── Bio/Ecology Plots ──────────────────────────────────────────────────────


def plot_hgt_scatter(
    hgt_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Scatter plot of HGT candidate detection.

    x = GC3 deviation from genome mean, y = Mahalanobis distance.
    Color by hgt_flag (True=red, False=gray).
    Markers by expression class if available.

    Args:
        hgt_df: DataFrame with gc3, mahalanobis_dist, hgt_flag, optional expression_class.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if hgt_df.empty or "mahalanobis_dist" not in hgt_df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Marker mapping
    marker_map = {"high": "^", "medium": "o", "low": "v"}
    size_map = {"high": 50, "medium": 30, "low": 50}

    has_expr = "expression_class" in hgt_df.columns

    # Plot non-HGT candidates
    non_hgt = hgt_df[~hgt_df["hgt_flag"]] if "hgt_flag" in hgt_df.columns else hgt_df
    if not non_hgt.empty:
        ax.scatter(
            non_hgt.get("gc3_deviation", 0),
            non_hgt["mahalanobis_dist"],
            s=30,
            c="lightgray",
            alpha=0.5,
            marker="o",
            label="Non-HGT",
            edgecolors="none",
        )

    # Plot HGT candidates
    hgt = hgt_df[hgt_df["hgt_flag"]] if "hgt_flag" in hgt_df.columns else pd.DataFrame()
    if not hgt.empty:
        if has_expr:
            for expr_class in ["high", "medium", "low"]:
                subset = hgt[hgt.get("expression_class", "") == expr_class]
                if not subset.empty:
                    ax.scatter(
                        subset.get("gc3_deviation", 0),
                        subset["mahalanobis_dist"],
                        s=size_map.get(expr_class, 30),
                        c="red",
                        alpha=0.7,
                        marker=marker_map.get(expr_class, "o"),
                        label=f"HGT ({expr_class})",
                        edgecolors="darkred",
                        linewidth=0.5,
                    )
        else:
            ax.scatter(
                hgt.get("gc3_deviation", 0),
                hgt["mahalanobis_dist"],
                s=50,
                c="red",
                alpha=0.7,
                marker="o",
                label="HGT candidates",
                edgecolors="darkred",
                linewidth=0.5,
            )

    # Mahalanobis distance threshold (chi2 p=0.001)
    # For 1 DOF: chi2(0.001) ≈ 10.83
    threshold = np.sqrt(10.83)
    ax.axhline(threshold, color="blue", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Mahal threshold (p=0.001)")

    ax.set_xlabel("GC3 Deviation from Genome Mean", fontsize=11)
    ax.set_ylabel("Mahalanobis Distance", fontsize=11)

    title_str = "HGT Candidate Detection"
    if sample_id:
        title_str += f" — {sample_id}"
    ax.set_title(title_str, fontsize=12)

    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_fop_gradient(
    fop_gradient_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Bar chart of mean Fop across expression quintiles.

    Shows expected increase in Fop with expression level if translational
    selection is operating.

    Args:
        fop_gradient_df: DataFrame with expr_quintile, mean_fop, std_fop.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if fop_gradient_df.empty or "mean_fop" not in fop_gradient_df.columns:
        return

    # Column may be named "quintile" (bio_ecology module) or "expr_quintile"
    quintile_col = "expr_quintile" if "expr_quintile" in fop_gradient_df.columns else "quintile"
    if quintile_col not in fop_gradient_df.columns:
        logger.warning("SKIPPED: FOP gradient plot (no quintile column)")
        return

    df = fop_gradient_df.sort_values(quintile_col).copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color gradient: blue (low) to red (high)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(df)))

    x_pos = df[quintile_col]
    y = df["mean_fop"]
    yerr = df.get("std_fop", None)

    bars = ax.bar(
        x_pos,
        y,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1,
        yerr=yerr,
        capsize=5,
        error_kw={"elinewidth": 1, "ecolor": "black"},
    )

    # Add trend line
    z = np.polyfit(x_pos, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x_pos.min(), x_pos.max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=2, alpha=0.7, label="Trend")

    # Spearman correlation
    if len(df) > 2:
        r_sp, p_val = stats.spearmanr(x_pos, y)
        ax.text(
            0.05, 0.95, f"Spearman r = {r_sp:.3f}\np = {p_val:.3e}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel("Expression Quintile", fontsize=11)
    ax.set_ylabel("Mean Fop", fontsize=11)
    ax.set_xticks(x_pos)

    title_str = "Fop Gradient Across Expression Levels"
    if sample_id:
        title_str += f" — {sample_id}"
    ax.set_title(title_str, fontsize=12)

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_position_effects(
    position_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Three-panel boxplot of Fop at 5', middle, 3' gene regions.

    Args:
        position_df: DataFrame with position (5prime/middle/3prime), fop,
            optional expression_class.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if position_df.empty:
        return

    # bio_ecology produces wide format (fop_5prime, fop_middle, fop_3prime);
    # convert to long format (position, fop) if needed
    if "fop" not in position_df.columns and "fop_5prime" in position_df.columns:
        id_vars = [c for c in position_df.columns if not c.startswith("fop_")]
        melted = position_df.melt(
            id_vars=id_vars,
            value_vars=[c for c in ("fop_5prime", "fop_middle", "fop_3prime") if c in position_df.columns],
            var_name="position",
            value_name="fop",
        )
        melted["position"] = melted["position"].str.replace("fop_", "")
        position_df = melted

    if "fop" not in position_df.columns:
        return

    has_expr = "expression_class" in position_df.columns

    if has_expr:
        expr_classes = sorted(position_df["expression_class"].unique())
        n_panels = len(expr_classes)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes = [ax]

    positions = ["5prime", "middle", "3prime"]
    position_labels = ["5'", "Middle", "3'"]

    for panel_idx, axis in enumerate(axes):
        if has_expr:
            expr_class = expr_classes[panel_idx]
            data = position_df[position_df["expression_class"] == expr_class]
            axis.set_title(f"{expr_class.capitalize()} expression", fontsize=11)
        else:
            data = position_df
            axis.set_title("Gene position effects on Fop", fontsize=11)

        # Create boxplot data
        if "position" not in data.columns:
            bp_data = [pd.Series(dtype=float) for _ in positions]
        else:
            bp_data = [data[data["position"] == pos]["fop"].dropna() for pos in positions]

        bp = axis.boxplot(
            bp_data,
            labels=position_labels,
            patch_artist=True,
            widths=0.6,
        )

        # Color boxes
        for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen", "lightcoral"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Mann-Whitney U test: 5' vs other regions (independent samples)
        if "position" not in data.columns:
            continue
        for pos2, label2 in zip(["middle", "3prime"], ["5' vs Middle", "5' vs 3'"]):
            data5 = data[data["position"] == "5prime"]["fop"].dropna()
            data2 = data[data["position"] == pos2]["fop"].dropna()

            if len(data5) > 2 and len(data2) > 2:
                try:
                    stat, pval = stats.mannwhitneyu(
                        data5, data2, alternative="two-sided"
                    )
                    sig_str = f"p={pval:.3e}"
                    axis.text(0.5, 0.95 - (0.1 if label2 == "5' vs 3'" else 0.05), sig_str,
                             transform=axis.transAxes, fontsize=8, ha="center",
                             bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3))
                except Exception:
                    pass

        axis.set_ylabel("Fop", fontsize=10)
        axis.grid(True, alpha=0.3, axis="y")

    if has_expr:
        fig.suptitle(f"Position Effects on Fop — {sample_id}" if sample_id else "Position Effects on Fop",
                    fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_strand_asymmetry(
    strand_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Bar chart of RSCU asymmetry between + and - strands.

    Groups codons by amino acid and shows mean RSCU on + strand vs - strand.
    Highlights codons with significant asymmetry (p < 0.05 after BH correction).

    Args:
        strand_df: DataFrame with codon, amino_acid, rscu_plus, rscu_minus, p_value.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    # bio_ecology uses "mean_rscu_plus/minus"; normalize to "rscu_plus/minus"
    strand_df = strand_df.copy()
    if "mean_rscu_plus" in strand_df.columns and "rscu_plus" not in strand_df.columns:
        strand_df.rename(columns={"mean_rscu_plus": "rscu_plus", "mean_rscu_minus": "rscu_minus"}, inplace=True)
    if strand_df.empty or "rscu_plus" not in strand_df.columns:
        return

    # Use pre-computed FDR-adjusted p-values from bio_ecology if available;
    # otherwise fall back to BH correction here for backward compatibility.
    if "p_adjusted" in strand_df.columns and "significant" in strand_df.columns:
        pass  # already corrected upstream
    elif "p_value" in strand_df.columns:
        from scipy.stats import rankdata
        n_tests = len(strand_df)
        ranks = rankdata(strand_df["p_value"])
        strand_df = strand_df.copy()
        strand_df["bh_threshold"] = (ranks / n_tests) * 0.05
        strand_df["significant"] = strand_df["p_value"] <= strand_df["bh_threshold"]
    else:
        strand_df = strand_df.copy()
        strand_df["significant"] = False

    # Group by amino acid
    aa_groups = strand_df.groupby("amino_acid", observed=True)

    n_aa = len(aa_groups)
    fig, axes = plt.subplots((n_aa + 1) // 2, 2 if n_aa > 1 else 1, figsize=(12, max(6, n_aa * 1.5)))

    if n_aa == 1:
        axes = [[axes]]
    elif n_aa == 2:
        axes = [[axes[0], axes[1]]]
    else:
        axes = axes.reshape(-1, 2)

    axes = axes.flatten()

    for panel_idx, (aa, aa_data) in enumerate(aa_groups):
        ax = axes[panel_idx]

        aa_data = aa_data.sort_values("codon")
        x_pos = np.arange(len(aa_data))
        width = 0.35

        plus = aa_data["rscu_plus"].values
        minus = aa_data["rscu_minus"].values
        colors_plus = ["red" if s else "steelblue" for s in aa_data["significant"]]
        colors_minus = ["darkred" if s else "slategray" for s in aa_data["significant"]]

        bars1 = ax.bar(x_pos - width / 2, plus, width, label="+", color=colors_plus, alpha=0.7, edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x_pos + width / 2, minus, width, label="-", color=colors_minus, alpha=0.7, edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Mean RSCU", fontsize=10)
        ax.set_title(f"{aa}", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(aa_data["codon"].values, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    # Hide extra subplots
    for panel_idx in range(n_aa, len(axes)):
        axes[panel_idx].set_visible(False)

    title_str = "Strand Asymmetry in Codon Usage"
    if sample_id:
        title_str += f" — {sample_id}"
    fig.suptitle(title_str, fontsize=12)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_operon_coadaptation(
    operon_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Histogram of RSCU distances between adjacent genes.

    Shows distribution of RSCU distances for same-strand adjacent genes
    with overlay of median shuffled distance.

    Args:
        operon_df: DataFrame with rscu_distance, intergenic_distance, same_operon_prediction.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    # bio_ecology uses "intergenic_bp"; normalize to "intergenic_distance"
    operon_df = operon_df.copy()
    if "intergenic_bp" in operon_df.columns and "intergenic_distance" not in operon_df.columns:
        operon_df.rename(columns={"intergenic_bp": "intergenic_distance"}, inplace=True)
    if operon_df.empty or "rscu_distance" not in operon_df.columns:
        return

    # Determine if we have the additional scatter plot data
    has_intergenic = "intergenic_distance" in operon_df.columns

    if has_intergenic:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(9, 6))

    # Histogram
    rscu_dist = operon_df["rscu_distance"].dropna()
    if len(rscu_dist) > 0:
        ax1.hist(rscu_dist, bins=30, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.8)

        # Median shuffled distance (if available)
        if "median_shuffled_distance" in operon_df.columns:
            median_shuffled = operon_df["median_shuffled_distance"].iloc[0] if not operon_df.empty else None
            if median_shuffled is not None:
                ax1.axvline(median_shuffled, color="red", linestyle="--", linewidth=2, label=f"Median shuffled: {median_shuffled:.3f}")
                ax1.legend(fontsize=9)

        ax1.set_xlabel("RSCU Distance", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.set_title("RSCU Distances Between Adjacent Genes", fontsize=11)
        ax1.grid(True, alpha=0.3)

    # Scatter plot if intergenic data available
    if has_intergenic:
        # Joint dropna to keep x and y aligned
        scatter_df = operon_df.dropna(subset=["intergenic_distance", "rscu_distance"])
        intergenic = scatter_df["intergenic_distance"]
        rscu_dist_scatter = scatter_df["rscu_distance"]
        has_prediction = "same_operon_prediction" in scatter_df.columns

        if len(intergenic) > 0:
            if has_prediction:
                prediction = scatter_df["same_operon_prediction"]
                for pred_val, color, label in [(True, "red", "Predicted same operon"), (False, "blue", "Predicted different")]:
                    mask = prediction == pred_val
                    if mask.any():
                        ax2.scatter(
                            intergenic[mask],
                            rscu_dist_scatter[mask],
                            alpha=0.5,
                            s=20,
                            color=color,
                            label=label,
                        )
                ax2.legend(fontsize=9)
            else:
                ax2.scatter(intergenic, rscu_dist_scatter, alpha=0.5, s=20, color="steelblue")

            ax2.set_xlabel("Intergenic Distance (bp)", fontsize=11)
            ax2.set_ylabel("RSCU Distance", fontsize=11)
            ax2.set_title("Intergenic Distance vs RSCU Distance", fontsize=11)
            ax2.grid(True, alpha=0.3)

    title_str = "Operon Co-adaptation Analysis"
    if sample_id:
        title_str += f" — {sample_id}"
    fig.suptitle(title_str, fontsize=12)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_growth_rate_gauge(
    growth_dict: dict,
    output_path: Path,
    sample_id: str = "",
):
    """Gauge/infographic plot showing growth rate and associated metrics.

    Displays predicted doubling time, mean CAI of RP genes, and growth class
    with color-coded zones: green (<2h fast), yellow (2-8h moderate), red (>8h slow).

    Args:
        growth_dict: Dict with keys doubling_time, mean_cai_rp, growth_class.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if not growth_dict:
        return

    doubling_time = growth_dict.get("predicted_doubling_time_hours",
                                     growth_dict.get("doubling_time", 0))
    mean_cai = growth_dict.get("mean_cai_rp", 0)
    growth_class = growth_dict.get("growth_class", "unknown")

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Main gauge: doubling time
    ax_gauge = fig.add_subplot(gs[0, :])

    # Color zones
    if doubling_time < 2:
        zone_color = "green"
        zone_label = "FAST"
    elif doubling_time < 8:
        zone_color = "yellow"
        zone_label = "MODERATE"
    else:
        zone_color = "red"
        zone_label = "SLOW"

    # Horizontal bar
    ax_gauge.barh([0], [doubling_time], height=0.3, color=zone_color, alpha=0.7, edgecolor="black", linewidth=2)
    ax_gauge.set_xlim(0, 20)
    ax_gauge.text(doubling_time + 0.3, 0, f"{doubling_time:.2f} hours", va="center", fontsize=11, weight="bold")

    # Zone boundaries
    ax_gauge.axvline(2, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax_gauge.axvline(8, color="orange", linestyle="--", alpha=0.5, linewidth=1)

    ax_gauge.text(1, -0.5, "FAST\n(<2h)", ha="center", fontsize=9, color="green", weight="bold")
    ax_gauge.text(5, -0.5, "MODERATE\n(2-8h)", ha="center", fontsize=9, color="orange", weight="bold")
    ax_gauge.text(14, -0.5, "SLOW\n(>8h)", ha="center", fontsize=9, color="red", weight="bold")

    ax_gauge.set_xlabel("Predicted Doubling Time (hours)", fontsize=11)
    ax_gauge.set_yticks([])
    ax_gauge.set_title(f"Growth Rate Gauge — {zone_label}", fontsize=12, weight="bold", color=zone_color)
    ax_gauge.grid(True, alpha=0.3, axis="x")

    # CAI panel
    ax_cai = fig.add_subplot(gs[1, 0])
    ax_cai.text(0.5, 0.6, f"{mean_cai:.3f}", ha="center", va="center", fontsize=20, weight="bold", transform=ax_cai.transAxes)
    ax_cai.text(0.5, 0.2, "Mean CAI\n(Ribosomal Proteins)", ha="center", va="center", fontsize=10, transform=ax_cai.transAxes)
    ax_cai.axis("off")

    # Growth class panel
    ax_class = fig.add_subplot(gs[1, 1])
    ax_class.text(0.5, 0.6, growth_class.upper(), ha="center", va="center", fontsize=16, weight="bold", transform=ax_class.transAxes)
    ax_class.text(0.5, 0.2, "Growth\nClass", ha="center", va="center", fontsize=10, transform=ax_class.transAxes)
    ax_class.axis("off")

    title_str = "Growth Rate Prediction"
    if sample_id:
        title_str += f" — {sample_id}"
    fig.suptitle(title_str, fontsize=13, weight="bold")

    # GridSpec already controls spacing; bbox_inches="tight" in _save_fig
    # handles final cropping, so tight_layout() is unnecessary here and
    # would warn about incompatible axes.
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_path)


# ─── gRodon2 growth prediction plots ────────────────────────────────────────


def plot_grodon2_summary(
    grodon_dict: dict,
    cai_growth_dict: dict | None,
    output_path: Path,
    sample_id: str = "",
):
    """Summary figure comparing gRodon2 and CAI-based growth predictions.

    Panel layout:
      Top row: doubling time comparison bar + confidence interval
      Bottom left: codon usage bias metrics (CUBHE, ConsistencyHE, CPB)
      Bottom right: growth class and model metadata

    Args:
        grodon_dict: gRodon2 result dict (d, CIs, CUBHE, ConsistencyHE, CPB).
        cai_growth_dict: Optional Vieira-Silva & Rocha result dict for comparison.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if not grodon_dict:
        return

    d = grodon_dict.get("predicted_doubling_time_hours", 0)
    lo = grodon_dict.get("lower_ci_hours", d)
    hi = grodon_dict.get("upper_ci_hours", d)
    cubhe = grodon_dict.get("CUBHE", 0)
    consistency = grodon_dict.get("ConsistencyHE", 0)
    cpb = grodon_dict.get("CPB")
    gc = grodon_dict.get("GC", 0)
    n_he = grodon_dict.get("n_highly_expressed", 0)
    growth_class = grodon_dict.get("growth_class", "unknown")

    cai_d = None
    if cai_growth_dict:
        cai_d = cai_growth_dict.get("predicted_doubling_time_hours")

    fig = plt.figure(figsize=(12, 7))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Top: doubling time comparison ────────────────────────────────
    ax_top = fig.add_subplot(gs[0, :])

    labels = []
    vals = []
    errs_lo = []
    errs_hi = []
    colors = []

    # gRodon2
    labels.append("gRodon2")
    vals.append(d)
    errs_lo.append(d - lo if lo else 0)
    errs_hi.append(hi - d if hi else 0)
    colors.append("#2196F3")

    # CAI method
    if cai_d is not None:
        labels.append("CAI\n(Vieira-Silva)")
        vals.append(cai_d)
        errs_lo.append(0)
        errs_hi.append(0)
        colors.append("#FF9800")

    y_pos = list(range(len(labels)))
    ax_top.barh(y_pos, vals, height=0.4, color=colors, alpha=0.8, edgecolor="black", linewidth=1)

    # CI whiskers for gRodon2
    if lo and hi:
        ax_top.errorbar(d, 0, xerr=[[d - lo], [hi - d]], fmt="none", ecolor="black",
                        capsize=5, capthick=1.5, linewidth=1.5)

    for i, v in enumerate(vals):
        ax_top.text(v + 0.15, i, f"{v:.2f} h", va="center", fontsize=10, weight="bold")

    # Zone shading
    x_max = max(max(vals) * 1.5, 10)
    ax_top.axvspan(0, 2, alpha=0.07, color="green")
    ax_top.axvspan(2, min(8, x_max), alpha=0.07, color="orange")
    if x_max > 8:
        ax_top.axvspan(8, x_max, alpha=0.07, color="red")
    ax_top.axvline(2, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_top.axvline(8, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)

    ax_top.set_xlim(0, x_max)
    ax_top.set_yticks(y_pos)
    ax_top.set_yticklabels(labels, fontsize=10)
    ax_top.set_xlabel("Predicted Minimum Doubling Time (hours)", fontsize=10)
    ax_top.set_title("Growth Rate Prediction Comparison", fontsize=11, weight="bold")
    ax_top.grid(True, alpha=0.3, axis="x")

    # ── Bottom left: CUB metrics ─────────────────────────────────────
    ax_cub = fig.add_subplot(gs[1, 0])
    metric_names = ["CUBHE", "Consistency\nHE"]
    metric_vals = [cubhe, consistency]
    bar_colors = ["#4CAF50", "#9C27B0"]

    if cpb is not None:
        metric_names.append("CPB")
        metric_vals.append(cpb)
        bar_colors.append("#E91E63")

    x_pos = list(range(len(metric_names)))
    bars = ax_cub.bar(x_pos, metric_vals, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, metric_vals):
        ax_cub.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, weight="bold")

    ax_cub.set_xticks(x_pos)
    ax_cub.set_xticklabels(metric_names, fontsize=9)
    ax_cub.set_ylabel("Value", fontsize=10)
    ax_cub.set_title("gRodon2 Codon Usage Metrics", fontsize=10, weight="bold")
    ax_cub.grid(True, alpha=0.3, axis="y")

    # ── Bottom right: info panel ─────────────────────────────────────
    ax_info = fig.add_subplot(gs[1, 1])
    ax_info.axis("off")

    info_lines = [
        f"Growth class: {growth_class.upper()}",
        f"GC content: {gc:.3f}",
        f"Ribosomal proteins: {n_he}",
        f"95% CI: [{lo:.2f}, {hi:.2f}] h",
        f"Model: full (Madin training set)",
    ]
    if grodon_dict.get("caveat"):
        info_lines.append(f"\n{grodon_dict['caveat']}")

    info_text = "\n".join(info_lines)
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=10,
                 va="top", ha="left", linespacing=1.6,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"))

    title_str = "gRodon2 Growth Rate Analysis"
    if sample_id:
        title_str += f" — {sample_id}"
    fig.suptitle(title_str, fontsize=13, weight="bold")
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    _save_fig(fig, output_path)


def plot_grodon2_batch_comparison(
    grodon_data: dict[str, dict],
    cai_data: dict[str, dict] | None,
    output_path: Path,
):
    """Compare gRodon2 predictions across multiple genomes.

    Three-panel figure:
      Left: horizontal bar chart of doubling times with CIs per genome
      Middle: scatter of CUBHE vs ConsistencyHE, colored by growth class
      Right: gRodon2 vs CAI predicted doubling time scatter (if CAI data present)

    Args:
        grodon_data: {sample_id: grodon2_result_dict}.
        cai_data: Optional {sample_id: cai_growth_rate_dict}.
        output_path: Base path for saving.
    """
    _apply_style()
    if not grodon_data:
        return

    sample_ids = sorted(grodon_data.keys())
    n = len(sample_ids)

    has_cai = cai_data is not None and len(cai_data) > 0
    n_panels = 3 if has_cai else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, max(4, n * 0.5 + 2)))
    if n_panels == 1:
        axes = [axes]

    # Color by growth class
    class_colors = {"fast": "#4CAF50", "moderate": "#FF9800", "slow": "#F44336", "very_slow": "#9E9E9E"}

    # ── Panel 1: doubling time bars with CI ──────────────────────────
    ax = axes[0]
    y_pos = list(range(n))
    d_vals = [grodon_data[s].get("predicted_doubling_time_hours", 0) for s in sample_ids]
    lo_vals = [grodon_data[s].get("lower_ci_hours", d_vals[i]) for i, s in enumerate(sample_ids)]
    hi_vals = [grodon_data[s].get("upper_ci_hours", d_vals[i]) for i, s in enumerate(sample_ids)]
    gc_vals = [grodon_data[s].get("growth_class", "unknown") for s in sample_ids]
    bar_colors = [class_colors.get(gc, "#9E9E9E") for gc in gc_vals]

    xerr_lo = [d_vals[i] - lo_vals[i] for i in range(n)]
    xerr_hi = [hi_vals[i] - d_vals[i] for i in range(n)]

    ax.barh(y_pos, d_vals, height=0.6, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.8)
    ax.errorbar(d_vals, y_pos, xerr=[xerr_lo, xerr_hi], fmt="none", ecolor="black",
                capsize=3, capthick=1, linewidth=1)

    ax.axvline(2, color="green", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(5, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sample_ids, fontsize=9)
    ax.set_xlabel("Predicted Doubling Time (h)", fontsize=10)
    ax.set_title("gRodon2 Doubling Time", fontsize=10, weight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # ── Panel 2: CUBHE vs ConsistencyHE scatter ──────────────────────
    ax2 = axes[1]
    cubhe_vals = [grodon_data[s].get("CUBHE", 0) for s in sample_ids]
    cons_vals = [grodon_data[s].get("ConsistencyHE", 0) for s in sample_ids]

    for i, s in enumerate(sample_ids):
        ax2.scatter(cubhe_vals[i], cons_vals[i], c=bar_colors[i], s=80,
                    edgecolors="black", linewidth=0.8, zorder=3)
        ax2.annotate(s, (cubhe_vals[i], cons_vals[i]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")

    ax2.set_xlabel("CUBHE (CUB of HE genes)", fontsize=10)
    ax2.set_ylabel("ConsistencyHE", fontsize=10)
    ax2.set_title("Codon Usage Bias Landscape", fontsize=10, weight="bold")
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: gRodon2 vs CAI scatter (if CAI available) ───────────
    if has_cai:
        ax3 = axes[2]
        grodon_d = []
        cai_d = []
        labels = []
        scatter_colors = []
        for s in sample_ids:
            if s in cai_data and cai_data[s]:
                gd = grodon_data[s].get("predicted_doubling_time_hours")
                cd = cai_data[s].get("predicted_doubling_time_hours")
                if gd is not None and cd is not None:
                    grodon_d.append(gd)
                    cai_d.append(cd)
                    labels.append(s)
                    scatter_colors.append(class_colors.get(
                        grodon_data[s].get("growth_class", "unknown"), "#9E9E9E"))

        if grodon_d:
            ax3.scatter(cai_d, grodon_d, c=scatter_colors, s=80,
                        edgecolors="black", linewidth=0.8, zorder=3)
            for i, s in enumerate(labels):
                ax3.annotate(s, (cai_d[i], grodon_d[i]), fontsize=7,
                             xytext=(4, 4), textcoords="offset points")

            # Diagonal
            lim_max = max(max(grodon_d), max(cai_d)) * 1.2
            ax3.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, linewidth=0.8)
            ax3.set_xlim(0, lim_max)
            ax3.set_ylim(0, lim_max)

        ax3.set_xlabel("CAI-based Doubling Time (h)", fontsize=10)
        ax3.set_ylabel("gRodon2 Doubling Time (h)", fontsize=10)
        ax3.set_title("gRodon2 vs CAI Model", fontsize=10, weight="bold")
        ax3.grid(True, alpha=0.3)

    fig.suptitle("gRodon2 Growth Rate Comparison Across Genomes", fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, output_path)


# ─── Ribosomal vs high-expression codon usage plots ─────────────────────────


def plot_rp_vs_he_rscu(
    rscu_gene_df: pd.DataFrame,
    rscu_rp: dict[str, float],
    expr_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Grouped bar chart comparing RSCU across ribosomal, high-expression, and genome-average gene sets.

    For each codon (grouped by amino acid family), three bars show:
      - genome-wide median RSCU (gray)
      - high-expression genes median RSCU (orange)
      - ribosomal protein RSCU (blue)

    This reveals which codons are differentially preferred by the
    translational apparatus vs. other highly expressed genes.

    Args:
        rscu_gene_df: Per-gene RSCU table (gene + 59 RSCU columns).
        rscu_rp: Dict of concatenated ribosomal protein RSCU values.
        expr_df: Expression table with gene and CAI_class columns.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns and c in rscu_rp]
    if len(rscu_cols) < 10:
        return

    # Compute genome-wide median RSCU
    genome_median = rscu_gene_df[rscu_cols].median()

    # Compute high-expression gene median RSCU
    he_genes = set()
    for col in ("CAI_class", "MELP_class"):
        if col in expr_df.columns:
            he_genes |= set(expr_df.loc[expr_df[col] == "high", "gene"])
    if not he_genes:
        return

    he_mask = rscu_gene_df["gene"].isin(he_genes)
    if he_mask.sum() < 3:
        return
    he_median = rscu_gene_df.loc[he_mask, rscu_cols].median()

    # Ribosomal RSCU
    rp_vals = pd.Series({c: rscu_rp[c] for c in rscu_cols})

    # Build grouped data
    n = len(rscu_cols)
    x = np.arange(n)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(max(14, n * 0.35), 6))

    ax.bar(x - bar_w, genome_median[rscu_cols].values, bar_w,
           label="Genome average", color="#b0b0b0", edgecolor="white", linewidth=0.3)
    ax.bar(x, he_median[rscu_cols].values, bar_w,
           label="High-expression", color="#e8853d", edgecolor="white", linewidth=0.3)
    ax.bar(x + bar_w, rp_vals[rscu_cols].values, bar_w,
           label="Ribosomal proteins", color="#4a90d9", edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("-")[-1] for c in rscu_cols], rotation=90, fontsize=6)
    ax.set_ylabel("RSCU")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Codon usage: ribosomal vs high-expression genes — {sample_id}")

    # Add amino acid family separators
    _add_aa_family_spans(ax, rscu_cols, n)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rp_he_rscu_scatter(
    rscu_gene_df: pd.DataFrame,
    rscu_rp: dict[str, float],
    expr_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Scatter of ribosomal vs high-expression RSCU per codon.

    Each point is one codon.  The diagonal means identical preference.
    Points off the diagonal highlight codons with divergent usage between
    the two gene classes.  Colored by amino acid family.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        rscu_rp: Concatenated ribosomal RSCU dict.
        expr_df: Expression table with CAI_class.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns and c in rscu_rp]
    if len(rscu_cols) < 10:
        return

    # High-expression gene median RSCU
    he_genes = set()
    for col in ("CAI_class", "MELP_class"):
        if col in expr_df.columns:
            he_genes |= set(expr_df.loc[expr_df[col] == "high", "gene"])
    if not he_genes:
        return
    he_mask = rscu_gene_df["gene"].isin(he_genes)
    if he_mask.sum() < 3:
        return
    he_median = rscu_gene_df.loc[he_mask, rscu_cols].median()
    rp_vals = pd.Series({c: rscu_rp[c] for c in rscu_cols})

    # Assign colors by amino acid family
    aa_families = {}
    for col in rscu_cols:
        aa = col.rsplit("-", 1)[0]
        aa_families.setdefault(aa, []).append(col)
    family_names = sorted(aa_families.keys())
    cmap = plt.cm.get_cmap("tab20", len(family_names))
    col_to_color = {}
    for i, fam in enumerate(family_names):
        for c in aa_families[fam]:
            col_to_color[c] = cmap(i)

    fig, ax = plt.subplots(figsize=(8, 8))
    for c in rscu_cols:
        ax.scatter(rp_vals[c], he_median[c], s=50, c=[col_to_color[c]],
                   edgecolors="black", linewidth=0.3, zorder=3)

    # Label outliers (codons where RP and HE differ most)
    diffs = abs(rp_vals - he_median)
    threshold = diffs.quantile(0.85)
    for c in rscu_cols:
        if diffs[c] >= threshold:
            codon_label = c.split("-")[-1]
            ax.annotate(codon_label, (rp_vals[c], he_median[c]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    # Diagonal
    lim_max = max(rp_vals.max(), he_median.max()) * 1.15
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)

    ax.set_xlabel("Ribosomal protein RSCU")
    ax.set_ylabel("High-expression gene RSCU")
    ax.set_title(f"Ribosomal vs high-expression codon preference — {sample_id}")

    # Spearman correlation
    rho, pval = stats.spearmanr(rp_vals[rscu_cols], he_median[rscu_cols])
    ax.text(0.05, 0.95, f"Spearman ρ = {rho:.3f}, p = {pval:.2e}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rp_he_delta_heatmap(
    rscu_gene_df: pd.DataFrame,
    rscu_rp: dict[str, float],
    expr_df: pd.DataFrame,
    output_path: Path,
    sample_id: str = "",
):
    """Heatmap of ΔRSCU (deviation from genome average) for ribosomal and high-expression genes.

    Two columns per amino acid family: one for ribosomal proteins, one for
    high-expression genes.  Blue = underused relative to genome, red = overused.
    Highlights codons under different selection pressures in each class.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        rscu_rp: Concatenated ribosomal RSCU dict.
        expr_df: Expression table with CAI_class.
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns and c in rscu_rp]
    if len(rscu_cols) < 10:
        return

    genome_median = rscu_gene_df[rscu_cols].median()

    he_genes = set()
    for col in ("CAI_class", "MELP_class"):
        if col in expr_df.columns:
            he_genes |= set(expr_df.loc[expr_df[col] == "high", "gene"])
    if not he_genes:
        return
    he_mask = rscu_gene_df["gene"].isin(he_genes)
    if he_mask.sum() < 3:
        return
    he_median = rscu_gene_df.loc[he_mask, rscu_cols].median()
    rp_vals = pd.Series({c: rscu_rp[c] for c in rscu_cols})

    delta_rp = rp_vals - genome_median
    delta_he = he_median - genome_median

    # Build matrix: rows = codons, columns = [RP, HE]
    mat = pd.DataFrame({
        "Ribosomal": delta_rp[rscu_cols].values,
        "High-expression": delta_he[rscu_cols].values,
    }, index=[c.split("-")[-1] for c in rscu_cols])

    vmax = max(abs(mat.values.min()), abs(mat.values.max()), 0.5)
    fig, ax = plt.subplots(figsize=(4, max(10, len(rscu_cols) * 0.22)))
    sns.heatmap(
        mat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "ΔRSCU (vs genome average)", "shrink": 0.6},
        annot=True, fmt=".2f", annot_kws={"size": 6},
    )
    ax.set_ylabel("Codon")
    ax.set_title(f"ΔRSCU from genome average — {sample_id}", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save_fig(fig, output_path)


def _add_aa_family_spans(ax, rscu_cols: list[str], n: int):
    """Add faint vertical spans to separate amino acid families on bar charts."""
    prev_aa = None
    span_start = 0
    colors = ["#f0f0f0", "#e0e8f0"]
    color_idx = 0
    for i, col in enumerate(rscu_cols):
        aa = col.rsplit("-", 1)[0]
        if aa != prev_aa and prev_aa is not None:
            ax.axvspan(span_start - 0.5, i - 0.5, color=colors[color_idx % 2], alpha=0.3, zorder=0)
            color_idx += 1
            span_start = i
        prev_aa = aa
    # Last family
    ax.axvspan(span_start - 0.5, n - 0.5, color=colors[color_idx % 2], alpha=0.3, zorder=0)


# ─── Genomic landscape and MGE codon usage plots ───────────────────────────


def plot_genomic_cu_landscape(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    hgt_df: pd.DataFrame | None,
    phage_mobile_df: pd.DataFrame | None,
    output_path: Path,
    sample_id: str = "",
    gff_path: Path | None = None,
):
    """Multi-track genomic landscape of codon usage variation.

    Upper track: Mahalanobis distance (if HGT data available) or ENC per gene.
    Lower track: GC3 content per gene.
    MGE/phage genes are highlighted with colored vertical bands.
    Gene positions come from GFF if available; otherwise gene order in the
    RSCU table is used as a positional proxy (Prokka produces genes in
    genome order).

    Args:
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC + GC3 per gene.
        hgt_df: HGT detection DataFrame (optional).
        phage_mobile_df: Phage/mobile element DataFrame (optional).
        output_path: Base path for saving.
        sample_id: Sample identifier.
        gff_path: GFF3 annotation file for genomic coordinates (optional).
    """
    _apply_style()
    if enc_df is None or enc_df.empty:
        return

    # Build positional index
    gene_positions = _get_gene_positions(rscu_gene_df, enc_df, gff_path)
    if gene_positions.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.08})
    ax_top, ax_bot = axes

    x = gene_positions["position"].values
    genes = gene_positions["gene"].values

    # --- Top track: codon usage deviation ---
    if hgt_df is not None and "mahalanobis_dist" in hgt_df.columns:
        merged = gene_positions.merge(hgt_df[["gene", "mahalanobis_dist"]], on="gene", how="left")
        y_top = merged["mahalanobis_dist"].values
        ax_top.set_ylabel("Mahalanobis\ndistance", fontsize=10)
    else:
        merged = gene_positions.merge(enc_df[["gene", "ENC"]], on="gene", how="left")
        y_top = merged["ENC"].values
        ax_top.set_ylabel("ENC", fontsize=10)

    ax_top.fill_between(x, 0, y_top, color="#4a90d9", alpha=0.4, linewidth=0)
    ax_top.plot(x, y_top, color="#2c5f8a", linewidth=0.5, alpha=0.7)

    # --- Bottom track: GC3 ---
    merged_gc3 = gene_positions.merge(enc_df[["gene", "GC3"]], on="gene", how="left")
    y_gc3 = merged_gc3["GC3"].values
    gc3_mean = np.nanmean(y_gc3)
    ax_bot.fill_between(x, gc3_mean, y_gc3, where=y_gc3 >= gc3_mean,
                        color="#e8853d", alpha=0.4, linewidth=0, interpolate=True)
    ax_bot.fill_between(x, gc3_mean, y_gc3, where=y_gc3 < gc3_mean,
                        color="#5fa85f", alpha=0.4, linewidth=0, interpolate=True)
    ax_bot.plot(x, y_gc3, color="#444444", linewidth=0.5, alpha=0.7)
    ax_bot.axhline(gc3_mean, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_bot.set_ylabel("GC3", fontsize=10)
    ax_bot.set_xlabel("Gene position (index)" if gff_path is None else "Genome position (kb)")

    # --- Highlight MGE/phage genes ---
    mge_genes = set()
    if phage_mobile_df is not None and not phage_mobile_df.empty:
        mob_col = next((c for c in ("is_mobilome", "mobilome") if c in phage_mobile_df.columns), None)
        phage_col = next((c for c in ("is_phage_related", "is_phage", "phage_related") if c in phage_mobile_df.columns), None)
        if mob_col:
            mge_genes |= set(phage_mobile_df.loc[phage_mobile_df[mob_col].astype(bool), "gene"])
        if phage_col:
            mge_genes |= set(phage_mobile_df.loc[phage_mobile_df[phage_col].astype(bool), "gene"])

    hgt_genes = set()
    if hgt_df is not None and "hgt_flag" in hgt_df.columns:
        hgt_genes = set(hgt_df.loc[hgt_df["hgt_flag"], "gene"])

    # Draw bands
    gene_to_x = dict(zip(genes, x))
    for g in mge_genes:
        if g in gene_to_x:
            xpos = gene_to_x[g]
            for a in (ax_top, ax_bot):
                a.axvline(xpos, color="#e84040", alpha=0.25, linewidth=1.5)
    for g in hgt_genes - mge_genes:
        if g in gene_to_x:
            xpos = gene_to_x[g]
            for a in (ax_top, ax_bot):
                a.axvline(xpos, color="#d9a032", alpha=0.2, linewidth=1.0)

    # Legend proxies
    from matplotlib.patches import Patch
    legend_elements = []
    if mge_genes:
        legend_elements.append(Patch(facecolor="#e84040", alpha=0.4, label="MGE/phage"))
    if hgt_genes - mge_genes:
        legend_elements.append(Patch(facecolor="#d9a032", alpha=0.4, label="HGT candidate"))
    if legend_elements:
        ax_top.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax_top.set_title(f"Codon usage landscape — {sample_id}", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_mge_vs_core_rscu(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    hgt_df: pd.DataFrame,
    phage_mobile_df: pd.DataFrame | None,
    expr_df: pd.DataFrame | None,
    output_path: Path,
    sample_id: str = "",
):
    """Violin plots comparing codon usage metrics between MGE-associated and core genome genes.

    Four panels: ENC, GC3, Mahalanobis distance, and CAI (if available).
    Genes are split into MGE (mobilome + phage), other HGT candidates, and
    core genome.  Mann-Whitney p-values annotated.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC + GC3 per gene.
        hgt_df: HGT detection DataFrame.
        phage_mobile_df: Phage/mobile element DataFrame (optional).
        expr_df: Expression table (optional, for CAI).
        output_path: Base path for saving.
        sample_id: Sample identifier.
    """
    _apply_style()
    if hgt_df is None or hgt_df.empty:
        return

    # Classify genes
    mge_genes = set()
    if phage_mobile_df is not None and not phage_mobile_df.empty:
        for col in ("is_mobilome", "is_phage_related", "is_phage", "phage_related", "mobilome"):
            if col in phage_mobile_df.columns:
                mge_genes |= set(phage_mobile_df.loc[phage_mobile_df[col].astype(bool), "gene"])

    hgt_only_genes = set()
    if "hgt_flag" in hgt_df.columns:
        hgt_only_genes = set(hgt_df.loc[hgt_df["hgt_flag"], "gene"]) - mge_genes

    all_genes = set(enc_df["gene"]) if "gene" in enc_df.columns else set()
    core_genes = all_genes - mge_genes - hgt_only_genes

    if not mge_genes and not hgt_only_genes:
        return

    # Build combined table
    merged = enc_df.copy()
    if "mahalanobis_dist" in hgt_df.columns:
        merged = merged.merge(hgt_df[["gene", "mahalanobis_dist"]], on="gene", how="left")
    if expr_df is not None and "CAI" in expr_df.columns:
        merged = merged.merge(expr_df[["gene", "CAI"]], on="gene", how="left")

    def _assign_class(gene):
        if gene in mge_genes:
            return "MGE/phage"
        elif gene in hgt_only_genes:
            return "HGT candidate"
        else:
            return "Core genome"

    merged["gene_class"] = merged["gene"].apply(_assign_class)

    # Determine panels
    panels = []
    if "ENC" in merged.columns:
        panels.append(("ENC", "ENC"))
    if "GC3" in merged.columns:
        panels.append(("GC3", "GC3"))
    if "mahalanobis_dist" in merged.columns:
        panels.append(("mahalanobis_dist", "Mahalanobis distance"))
    if "CAI" in merged.columns:
        panels.append(("CAI", "CAI"))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    order = ["Core genome", "HGT candidate", "MGE/phage"]
    order = [o for o in order if o in merged["gene_class"].unique()]
    palette = {"Core genome": "#b0b0b0", "HGT candidate": "#d9a032", "MGE/phage": "#e84040"}

    for ax, (col, label) in zip(axes, panels):
        plot_data = merged.dropna(subset=[col])
        if plot_data.empty:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=plot_data, x="gene_class", y=col, order=order,
            palette=palette, ax=ax, inner="box", cut=0, linewidth=0.8,
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)

        # Mann-Whitney U: MGE/phage vs core
        if "Core genome" in order and len(order) >= 2:
            test_class = "MGE/phage" if "MGE/phage" in order else order[-1]
            core_vals = plot_data.loc[plot_data["gene_class"] == "Core genome", col].dropna()
            test_vals = plot_data.loc[plot_data["gene_class"] == test_class, col].dropna()
            if len(core_vals) >= 3 and len(test_vals) >= 3:
                _, pval = stats.mannwhitneyu(core_vals, test_vals, alternative="two-sided")
                sig_str = f"p = {pval:.2e}"
                ax.set_title(sig_str, fontsize=8, fontstyle="italic")

    fig.suptitle(f"Codon usage: MGE vs core genome — {sample_id}", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def _get_gene_positions(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    gff_path: Path | None = None,
) -> pd.DataFrame:
    """Get gene positions from GFF or fall back to gene order.

    Returns DataFrame with columns: gene, position.
    """
    genes_ordered = rscu_gene_df["gene"].tolist() if "gene" in rscu_gene_df.columns else []

    if gff_path is not None and Path(gff_path).exists():
        try:
            positions = []
            with open(gff_path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 9 or parts[2] != "CDS":
                        continue
                    start = int(parts[3])
                    attrs = parts[8]
                    gene_id = None
                    for tag in ("ID=", "Name=", "locus_tag="):
                        if tag in attrs:
                            for field in attrs.split(";"):
                                if field.startswith(tag):
                                    gene_id = field.split("=", 1)[1]
                                    # Strip common prefixes
                                    for prefix in ("cds-", "cds_", "gene-", "gene_", "CDS:"):
                                        if gene_id.startswith(prefix):
                                            gene_id = gene_id[len(prefix):]
                                    break
                        if gene_id:
                            break
                    if gene_id:
                        positions.append({"gene": gene_id, "position": start / 1000.0})

            if positions:
                pos_df = pd.DataFrame(positions).drop_duplicates(subset="gene")
                # Only use genes present in our RSCU data
                gene_set = set(genes_ordered)
                pos_df = pos_df[pos_df["gene"].isin(gene_set)].sort_values("position")
                if len(pos_df) > 10:
                    return pos_df.reset_index(drop=True)
        except Exception:
            pass

    # Fallback: gene order as position
    if genes_ordered:
        return pd.DataFrame({
            "gene": genes_ordered,
            "position": np.arange(len(genes_ordered)),
        })
    return pd.DataFrame()


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
    bio_ecology_results: dict[str, pd.DataFrame | dict] | None = None,
    gff_path: Path | None = None,
) -> dict[str, Path]:
    """Generate all single-genome plots.

    Returns dict of plot paths.
    """
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Each plot is wrapped in try/except so one failure doesn't crash
    # the entire genome pipeline — analytical data is already saved to disk.

    if freq_df is not None and not freq_df.empty:
        try:
            p = plot_dir / f"{sample_id}_codon_frequency"
            plot_codon_frequency_bar(freq_df, p, sample_id)
            outputs["codon_frequency_bar"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Codon frequency bar plot failed: %s", e)
    else:
        logger.info("SKIPPED: codon frequency bar plot (no frequency data)")

    if freq_df is not None and not freq_df.empty:
        try:
            p = plot_dir / f"{sample_id}_rscu_heatmap_rounded"
            plot_rscu_heatmap_rounded(freq_df, p, sample_id)
            outputs["rscu_heatmap_rounded"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Rounded RSCU heatmap failed: %s", e)
    else:
        logger.info("SKIPPED: rounded RSCU heatmap (no frequency data)")

    if rscu_all is not None:
        try:
            p = plot_dir / f"{sample_id}_rscu_all"
            plot_rscu_bar(rscu_all, p, sample_id, "All CDS")
            outputs["rscu_bar_all"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RSCU bar plot (all CDS) failed: %s", e)
    else:
        logger.info("SKIPPED: RSCU bar plot — all CDS (no RSCU data)")

    if rscu_rp is not None:
        try:
            p = plot_dir / f"{sample_id}_rscu_ribosomal"
            plot_rscu_bar(rscu_rp, p, sample_id, "Ribosomal Proteins")
            outputs["rscu_bar_rp"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RSCU bar plot (ribosomal) failed: %s", e)
    else:
        logger.info("SKIPPED: RSCU bar plot — ribosomal proteins (no ribosomal RSCU data)")

    if rscu_gene_df is not None and not rscu_gene_df.empty:
        try:
            p = plot_dir / f"{sample_id}_rscu_heatmap"
            plot_rscu_heatmap_single(rscu_gene_df, p, sample_id)
            outputs["rscu_heatmap"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RSCU heatmap plot failed: %s", e)
    else:
        logger.info("SKIPPED: RSCU heatmap (no per-gene RSCU data)")

    if enc_df is not None and not enc_df.empty:
        try:
            p = plot_dir / f"{sample_id}_enc_gc3"
            plot_enc_gc3(enc_df, p, sample_id)
            outputs["enc_gc3"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("ENC-GC3 plot failed: %s", e)
    else:
        logger.info("SKIPPED: ENC vs GC3 plot (no ENC data)")

    if encprime_df is not None and not encprime_df.empty and enc_df is not None and not enc_df.empty:
        try:
            p = plot_dir / f"{sample_id}_encprime_gc3"
            plot_encprime_gc3(encprime_df, enc_df, p, sample_id)
            outputs["encprime_gc3"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("ENCprime-GC3 plot failed: %s", e)
    else:
        logger.info("SKIPPED: ENCprime vs GC3 plot (no ENCprime or ENC data)")

    if milc_df is not None and not milc_df.empty:
        try:
            p = plot_dir / f"{sample_id}_milc_dist"
            plot_milc_distribution(milc_df, p, sample_id)
            outputs["milc_dist"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("MILC distribution plot failed: %s", e)
    else:
        logger.info("SKIPPED: MILC distribution plot (no MILC data)")

    if expr_df is not None and not expr_df.empty:
        try:
            p = plot_dir / f"{sample_id}_expression_dist"
            plot_expression_distribution(expr_df, p, sample_id)
            outputs["expression_dist"] = p.with_suffix(".png")

            # Expression tier summary (stacked bars per metric)
            class_cols = [c for c in expr_df.columns if c.endswith("_class") and c != "expression_class"]
            if class_cols:
                p = plot_dir / f"{sample_id}_expression_tiers"
                plot_expression_tier_summary(expr_df, p, sample_id)
                outputs["expression_tiers"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Expression distribution/tier plot failed: %s", e)
    else:
        logger.info("SKIPPED: expression distribution and tier plots (no expression data)")

    # ── Ribosomal vs high-expression codon usage plots ──────────────
    if (rscu_gene_df is not None and not rscu_gene_df.empty
            and rscu_rp is not None
            and expr_df is not None and not expr_df.empty):
        try:
            p = plot_dir / f"{sample_id}_rp_vs_he_rscu"
            plot_rp_vs_he_rscu(rscu_gene_df, rscu_rp, expr_df, p, sample_id)
            outputs["rp_vs_he_rscu"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RP vs HE RSCU bar plot failed: %s", e)

        try:
            p = plot_dir / f"{sample_id}_rp_he_rscu_scatter"
            plot_rp_he_rscu_scatter(rscu_gene_df, rscu_rp, expr_df, p, sample_id)
            outputs["rp_he_rscu_scatter"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RP vs HE RSCU scatter failed: %s", e)

        try:
            p = plot_dir / f"{sample_id}_rp_he_delta_heatmap"
            plot_rp_he_delta_heatmap(rscu_gene_df, rscu_rp, expr_df, p, sample_id)
            outputs["rp_he_delta_heatmap"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("RP/HE ΔRSCU heatmap failed: %s", e)
    else:
        logger.info("SKIPPED: ribosomal vs high-expression plots (missing RSCU, RP, or expression data)")

    # Enrichment plots
    if enrichment_results:
        # Per-comparison bar plots
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

        # Additional enrichment summary plots
        # Dotplot
        if any(not edf.empty for edf in enrichment_results.values()):
            # Use the first non-empty DataFrame for dotplot
            for key, edf in enrichment_results.items():
                if not edf.empty:
                    p = plot_dir / f"{sample_id}_enrichment_dotplot"
                    plot_enrichment_dotplot(edf, p, sample_id=sample_id)
                    outputs["enrichment_dotplot"] = p.with_suffix(".png")
                    break

        # Heatmap (multi-comparison)
        non_empty = {k: v for k, v in enrichment_results.items() if not v.empty}
        if len(non_empty) > 1:
            p = plot_dir / f"{sample_id}_enrichment_heatmap"
            plot_enrichment_heatmap(non_empty, p, sample_id=sample_id)
            outputs["enrichment_heatmap"] = p.with_suffix(".png")
        else:
            logger.info("SKIPPED: enrichment heatmap (fewer than 2 non-empty comparisons)")

        # Summary bar chart
        if non_empty:
            p = plot_dir / f"{sample_id}_enrichment_summary"
            plot_enrichment_summary_bar(non_empty, p, sample_id=sample_id)
            outputs["enrichment_summary"] = p.with_suffix(".png")
    else:
        logger.info("SKIPPED: enrichment plots (no enrichment results)")

    # ── Advanced analysis plots ──────────────────────────────────────────
    if advanced_results:
        _generate_advanced_plots(
            advanced_results, plot_dir, sample_id, outputs,
            expr_df=expr_df,
        )
    else:
        logger.info("SKIPPED: advanced analysis plots (no advanced analysis data)")

    # ── Bio/Ecology plots ────────────────────────────────────────────────
    if bio_ecology_results:
        _generate_bio_ecology_plots(
            bio_ecology_results, plot_dir, sample_id, outputs,
            rscu_gene_df=rscu_gene_df, enc_df=enc_df,
            expr_df=expr_df, gff_path=gff_path,
        )
    else:
        logger.info("SKIPPED: bio/ecology plots (no bio/ecology analysis data)")

    return outputs


def _generate_advanced_plots(
    adv: dict,
    plot_dir: Path,
    sample_id: str,
    outputs: dict[str, Path],
    expr_df: pd.DataFrame | None = None,
):
    """Generate all advanced analysis plots from pre-computed data.

    Each plot is wrapped in try/except so one failure doesn't crash the
    entire genome pipeline — analytical data is already saved to disk.
    """

    # COA on RSCU
    if "coa_coords" in adv and "coa_inertia" in adv:
        try:
            coa_coords = adv["coa_coords"]
            coa_inertia = adv["coa_inertia"]
            color_col = "CAI_class" if "CAI_class" in coa_coords.columns else None
            p = plot_dir / f"{sample_id}_coa"
            plot_coa(coa_coords, coa_inertia, p, sample_id, color_col=color_col)
            outputs["coa"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("COA plot failed: %s", e)
    else:
        logger.info("SKIPPED: COA plot (no COA data)")

    if "coa_codon_coords" in adv:
        try:
            p = plot_dir / f"{sample_id}_coa_codons"
            plot_coa_codons(adv["coa_codon_coords"], p, sample_id)
            outputs["coa_codons"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("COA codon plot failed: %s", e)
    else:
        logger.info("SKIPPED: COA codon plot (no COA codon coordinates)")

    # S-value scatter
    if "s_value" in adv and expr_df is not None:
        try:
            for metric in ["CAI", "MELP", "Fop"]:
                if metric in expr_df.columns:
                    p = plot_dir / f"{sample_id}_s_value_vs_{metric.lower()}"
                    plot_s_value_scatter(adv["s_value"], expr_df, p, sample_id, metric)
                    outputs[f"s_value_vs_{metric.lower()}"] = p.with_suffix(".png")
                    break  # One S-value plot is enough
        except Exception as e:
            logger.warning("S-value scatter plot failed: %s", e)
    else:
        logger.info("SKIPPED: S-value scatter plot (no S-value or expression data)")

    # ENC - ENC' difference
    if "enc_diff" in adv:
        try:
            p = plot_dir / f"{sample_id}_enc_diff"
            plot_enc_diff(adv["enc_diff"], p, sample_id, expr_df=expr_df)
            outputs["enc_diff"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("ENC diff plot failed: %s", e)
    else:
        logger.info("SKIPPED: ENC-ENCprime difference plot (no ENC difference data)")

    # Neutrality plot
    if "gc12_gc3" in adv:
        try:
            p = plot_dir / f"{sample_id}_neutrality"
            plot_neutrality(adv["gc12_gc3"], p, sample_id)
            outputs["neutrality"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Neutrality plot failed: %s", e)
    else:
        logger.info("SKIPPED: neutrality plot (no GC12/GC3 data)")

    # PR2 plot
    if "pr2" in adv:
        try:
            p = plot_dir / f"{sample_id}_pr2"
            plot_pr2(adv["pr2"], p, sample_id, expr_df=expr_df)
            outputs["pr2"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("PR2 plot failed: %s", e)
    else:
        logger.info("SKIPPED: PR2 plot (no PR2 data)")

    # Delta RSCU heatmaps
    for metric in ["CAI", "MELP", "Fop"]:
        key = f"delta_rscu_{metric}"
        if key in adv:
            try:
                p = plot_dir / f"{sample_id}_delta_rscu_{metric.lower()}"
                plot_delta_rscu_heatmap(adv[key], p, sample_id, metric)
                outputs[f"delta_rscu_{metric.lower()}"] = p.with_suffix(".png")
            except Exception as e:
                logger.warning("Delta RSCU %s plot failed: %s", metric, e)

    # tRNA-codon correlation
    if "trna_codon_correlation" in adv:
        try:
            p = plot_dir / f"{sample_id}_trna_codon"
            plot_trna_codon_correlation(adv["trna_codon_correlation"], p, sample_id)
            outputs["trna_codon"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("tRNA-codon correlation plot failed: %s", e)
    else:
        logger.info("SKIPPED: tRNA-codon correlation plot (no tRNA data)")

    # COG enrichment
    if "cog_enrichment" in adv:
        try:
            p = plot_dir / f"{sample_id}_cog_enrichment"
            plot_cog_enrichment(adv["cog_enrichment"], p, sample_id)
            outputs["cog_enrichment"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("COG enrichment plot failed: %s", e)
    else:
        logger.info("SKIPPED: COG enrichment plot (no COG enrichment data)")

    # Gene length vs bias
    if "gene_length_bias" in adv:
        try:
            p = plot_dir / f"{sample_id}_gene_length_bias"
            plot_gene_length_vs_bias(adv["gene_length_bias"], p, sample_id)
            outputs["gene_length_bias"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Gene length bias plot failed: %s", e)
    else:
        logger.info("SKIPPED: gene length vs bias plot (no gene length bias data)")


def _generate_bio_ecology_plots(
    bio: dict[str, pd.DataFrame | dict],
    plot_dir: Path,
    sample_id: str,
    outputs: dict[str, Path],
    rscu_gene_df: pd.DataFrame | None = None,
    enc_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    gff_path: Path | None = None,
):
    """Generate bio/ecology analysis plots from pre-computed data.

    Expected keys: hgt, growth_rate, optimal_codons, fop_gradient,
    position_effects, phage_mobile, strand_asymmetry, operon_coadaptation.
    """

    # Each plot is wrapped in try/except so one failure doesn't crash
    # the entire genome pipeline — analytical data is already saved to disk.

    # HGT scatter
    if "hgt" in bio and isinstance(bio["hgt"], pd.DataFrame) and not bio["hgt"].empty:
        try:
            data = bio["hgt"]
            p = plot_dir / f"{sample_id}_hgt_scatter"
            plot_hgt_scatter(data, p, sample_id)
            outputs["hgt_scatter"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("HGT scatter plot failed: %s", e)
    else:
        logger.info("SKIPPED: HGT scatter plot (no HGT data)")

    # Fop gradient
    if "fop_gradient" in bio and isinstance(bio["fop_gradient"], pd.DataFrame) and not bio["fop_gradient"].empty:
        try:
            data = bio["fop_gradient"]
            p = plot_dir / f"{sample_id}_fop_gradient"
            plot_fop_gradient(data, p, sample_id)
            outputs["fop_gradient"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Fop gradient plot failed: %s", e)
    else:
        logger.info("SKIPPED: Fop gradient plot (no Fop gradient data)")

    # Position effects
    if "position_effects" in bio and isinstance(bio["position_effects"], pd.DataFrame) and not bio["position_effects"].empty:
        try:
            data = bio["position_effects"]
            p = plot_dir / f"{sample_id}_position_effects"
            plot_position_effects(data, p, sample_id)
            outputs["position_effects"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Position effects plot failed: %s", e)
    else:
        logger.info("SKIPPED: codon position effects plot (no position effects data)")

    # Strand asymmetry
    if "strand_asymmetry" in bio and isinstance(bio["strand_asymmetry"], pd.DataFrame) and not bio["strand_asymmetry"].empty:
        try:
            data = bio["strand_asymmetry"]
            p = plot_dir / f"{sample_id}_strand_asymmetry"
            plot_strand_asymmetry(data, p, sample_id)
            outputs["strand_asymmetry"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Strand asymmetry plot failed: %s", e)
    else:
        logger.info("SKIPPED: strand asymmetry plot (no strand asymmetry data)")

    # Operon co-adaptation
    if "operon_coadaptation" in bio and isinstance(bio["operon_coadaptation"], pd.DataFrame) and not bio["operon_coadaptation"].empty:
        try:
            data = bio["operon_coadaptation"]
            p = plot_dir / f"{sample_id}_operon_coadaptation"
            plot_operon_coadaptation(data, p, sample_id)
            outputs["operon_coadaptation"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Operon coadaptation plot failed: %s", e)
    else:
        logger.info("SKIPPED: operon coadaptation plot (no operon coadaptation data)")

    # Growth rate gauge
    if "growth_rate" in bio and isinstance(bio["growth_rate"], dict) and bio["growth_rate"]:
        try:
            data = bio["growth_rate"]
            p = plot_dir / f"{sample_id}_growth_rate_gauge"
            plot_growth_rate_gauge(data, p, sample_id)
            outputs["growth_rate_gauge"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Growth rate gauge plot failed: %s", e)
    else:
        logger.info("SKIPPED: growth rate gauge plot (no growth rate data)")

    # gRodon2 summary
    grodon_data = bio.get("grodon2_prediction")
    if isinstance(grodon_data, dict) and grodon_data:
        try:
            cai_data = bio.get("growth_rate") if isinstance(bio.get("growth_rate"), dict) else None
            p = plot_dir / f"{sample_id}_grodon2_summary"
            plot_grodon2_summary(grodon_data, cai_data, p, sample_id)
            outputs["grodon2_summary"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("gRodon2 summary plot failed: %s", e)
    else:
        logger.info("SKIPPED: gRodon2 summary plot (no gRodon2 data)")

    # Genomic CU landscape
    hgt_data = bio.get("hgt") if isinstance(bio.get("hgt"), pd.DataFrame) else None
    phage_data = bio.get("phage_mobile") if isinstance(bio.get("phage_mobile"), pd.DataFrame) else None
    if rscu_gene_df is not None and enc_df is not None:
        try:
            p = plot_dir / f"{sample_id}_genomic_cu_landscape"
            plot_genomic_cu_landscape(
                rscu_gene_df, enc_df, hgt_data, phage_data,
                p, sample_id, gff_path=gff_path,
            )
            outputs["genomic_cu_landscape"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("Genomic CU landscape plot failed: %s", e)
    else:
        logger.info("SKIPPED: genomic CU landscape plot (no RSCU/ENC data)")

    # MGE vs core genome codon usage
    if hgt_data is not None and not hgt_data.empty and rscu_gene_df is not None and enc_df is not None:
        try:
            p = plot_dir / f"{sample_id}_mge_vs_core_rscu"
            plot_mge_vs_core_rscu(
                rscu_gene_df, enc_df, hgt_data, phage_data, expr_df,
                p, sample_id,
            )
            outputs["mge_vs_core_rscu"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("MGE vs core RSCU plot failed: %s", e)
    else:
        logger.info("SKIPPED: MGE vs core RSCU plot (no HGT data)")


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


# ─── Qualitative pairwise / small-batch comparison plots ───────────────────
#
# These plots work with any number of samples >= 2, without statistical tests.
# They provide visual comparisons that are useful when sample sizes are too
# small for formal hypothesis testing.


def _load_sample_data(
    sample_outputs: dict[str, dict[str, Path]],
) -> dict[str, dict]:
    """Load per-sample TSV data for pairwise comparison plots.

    Returns a dict keyed by sample_id, each containing DataFrames and dicts
    for the available analyses.
    """
    data: dict[str, dict] = {}
    for sid, paths in sample_outputs.items():
        entry: dict = {"sample_id": sid}

        # RSCU median
        p = paths.get("rscu_median")
        if p and Path(p).exists():
            try:
                entry["rscu_median"] = pd.read_csv(p, sep="\t")
            except Exception:
                pass

        # Ribosomal RSCU
        p = paths.get("rscu_ribosomal")
        if p and Path(p).exists():
            try:
                entry["rscu_ribosomal"] = pd.read_csv(p, sep="\t")
            except Exception:
                pass

        # ENC
        p = paths.get("enc")
        if p and Path(p).exists():
            try:
                entry["enc"] = pd.read_csv(p, sep="\t")
            except Exception:
                pass

        # Expression
        p = paths.get("expression_combined")
        if p and Path(p).exists():
            try:
                entry["expression"] = pd.read_csv(p, sep="\t")
            except Exception:
                pass

        # Enrichment results
        enrich = {}
        for key, ep in paths.items():
            if key.startswith("enrichment_") and ep and Path(ep).exists():
                try:
                    enrich[key] = pd.read_csv(ep, sep="\t")
                except Exception:
                    pass
        if enrich:
            entry["enrichment"] = enrich

        # HGT candidates
        for key in ("bio_hgt_candidates_path", "bio_hgt_candidates", "hgt_candidates"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    entry["hgt"] = pd.read_csv(p, sep="\t")
                    break
                except Exception:
                    pass

        # Phage/mobile elements
        for key in ("bio_phage_mobile_elements_path", "bio_phage_mobile_elements", "phage_mobile_elements"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    entry["phage_mobile"] = pd.read_csv(p, sep="\t")
                    break
                except Exception:
                    pass

        # Strand asymmetry
        for key in ("bio_strand_asymmetry_path", "bio_strand_asymmetry", "strand_asymmetry"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    entry["strand_asymmetry"] = pd.read_csv(p, sep="\t")
                    break
                except Exception:
                    pass

        # Operon coadaptation
        for key in ("bio_operon_coadaptation_path", "bio_operon_coadaptation", "operon_coadaptation"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    entry["operon_coadaptation"] = pd.read_csv(p, sep="\t")
                    break
                except Exception:
                    pass

        # Growth rate
        for key in ("bio_growth_rate_prediction_path", "bio_growth_rate_prediction"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    import json as _json
                    entry["growth_rate"] = _json.loads(Path(p).read_text())
                    break
                except Exception:
                    # Growth rate TSV is also common
                    try:
                        _gr_df = pd.read_csv(p, sep="\t")
                        if not _gr_df.empty:
                            entry["growth_rate"] = _gr_df.iloc[0].to_dict()
                        break
                    except Exception:
                        pass

        # gRodon2 prediction
        for key in ("bio_grodon2_prediction_path", "bio_grodon2_prediction"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    _gd_df = pd.read_csv(p, sep="\t")
                    if not _gd_df.empty:
                        entry["grodon2"] = _gd_df.iloc[0].to_dict()
                    break
                except Exception:
                    pass

        # Translational selection - optimal codons
        for key in ("bio_trans_sel_optimal_codons_path", "bio_trans_sel_optimal_codons"):
            p = paths.get(key)
            if p and Path(p).exists():
                try:
                    entry["optimal_codons"] = pd.read_csv(p, sep="\t")
                    break
                except Exception:
                    pass

        # Codon frequency table
        p = paths.get("codon_frequency")
        if p and Path(p).exists():
            try:
                entry["codon_frequency"] = pd.read_csv(p, sep="\t")
            except Exception:
                pass

        data[sid] = entry

    return data


def plot_pairwise_rscu_overlay(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Overlaid RSCU bar chart comparing all genomes.

    Each genome's genome-wide median RSCU is plotted as a semi-transparent
    colored bar.  Shared preferences appear as tall stacked bars; divergent
    codons show up as mismatched heights.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    profiles = {}
    for sid in sids:
        df = sample_data[sid].get("rscu_median")
        if df is not None:
            rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]
            if rscu_cols and len(df) > 0:
                profiles[sid] = {c: df[c].iloc[0] for c in rscu_cols}

    if len(profiles) < 2:
        return

    common_cols = sorted(set.intersection(*[set(p.keys()) for p in profiles.values()]))
    if len(common_cols) < 10:
        return

    n = len(common_cols)
    n_samples = len(profiles)
    bar_w = 0.8 / n_samples
    x = np.arange(n)
    palette = sns.color_palette("Set2", n_samples)

    fig, ax = plt.subplots(figsize=(max(14, n * 0.35), 6))
    for i, (sid, vals) in enumerate(profiles.items()):
        y = [vals.get(c, 0) for c in common_cols]
        offset = (i - n_samples / 2 + 0.5) * bar_w
        ax.bar(x + offset, y, bar_w, label=sid, color=palette[i],
               edgecolor="white", linewidth=0.3, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("-")[-1] for c in common_cols], rotation=90, fontsize=6)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("RSCU")
    ax.set_title("Genome-wide RSCU comparison")
    ax.legend(fontsize=7, loc="upper right", ncol=min(n_samples, 4))

    _add_aa_family_spans(ax, common_cols, n)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_pairwise_rscu_delta_heatmap(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Heatmap of ΔRSCU between genomes.

    Rows = codons (grouped by amino acid).  For 2 genomes the single column
    shows the signed difference.  For >2 genomes each column is a pairwise
    comparison.  Codons with large absolute differences are the most
    discriminating markers.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    profiles = {}
    for sid in sids:
        df = sample_data[sid].get("rscu_median")
        if df is not None:
            rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in df.columns]
            if rscu_cols and len(df) > 0:
                profiles[sid] = {c: df[c].iloc[0] for c in rscu_cols}

    if len(profiles) < 2:
        return

    common_cols = sorted(set.intersection(*[set(p.keys()) for p in profiles.values()]))
    if len(common_cols) < 10:
        return

    # Build pairwise difference columns
    pairs = []
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            if sids[i] in profiles and sids[j] in profiles:
                pairs.append((sids[i], sids[j]))

    if not pairs:
        return

    # Limit to 10 pairwise comparisons for readability
    pairs = pairs[:10]

    mat_data = {}
    for s1, s2 in pairs:
        label = f"{s1}\nvs\n{s2}" if len(pairs) <= 5 else f"{s1} vs {s2}"
        mat_data[label] = [profiles[s1].get(c, 0) - profiles[s2].get(c, 0) for c in common_cols]

    mat = pd.DataFrame(mat_data, index=[c.split("-")[-1] for c in common_cols])

    vmax = max(abs(mat.values.min()), abs(mat.values.max()), 0.3)
    fig_w = max(4, len(pairs) * 1.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, max(10, len(common_cols) * 0.22)))
    sns.heatmap(
        mat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "ΔRSCU", "shrink": 0.6},
        annot=len(pairs) <= 3, fmt=".2f", annot_kws={"size": 5},
    )
    ax.set_ylabel("Codon")
    ax.set_title("Pairwise RSCU differences")
    ax.tick_params(axis="y", labelsize=6)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_genome_metrics_comparison(
    sample_data: dict[str, dict],
    combined_rscu: pd.DataFrame,
    output_path: Path,
):
    """Multi-panel dot plot comparing key genome-level metrics across samples.

    Panels: median ENC, median GC3, median CAI, predicted doubling time,
    HGT gene count, MGE/phage gene count.  Each sample is a labeled dot.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        combined_rscu: Combined RSCU table (for IQR-based metrics).
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    rows = []
    for sid in sids:
        row = {"sample_id": sid}
        enc_df = sample_data[sid].get("enc")
        if enc_df is not None and "ENC" in enc_df.columns:
            row["Median ENC"] = enc_df["ENC"].median()
            row["Median GC3"] = enc_df["GC3"].median() if "GC3" in enc_df.columns else np.nan

        expr = sample_data[sid].get("expression")
        if expr is not None:
            for m in ("CAI", "MELP", "Fop"):
                if m in expr.columns:
                    row[f"Median {m}"] = expr[m].median()

        gr = sample_data[sid].get("growth_rate")
        if isinstance(gr, dict) and "predicted_doubling_time_hours" in gr:
            row["Doubling time\n(CAI, h)"] = gr["predicted_doubling_time_hours"]

        grodon = sample_data[sid].get("grodon2")
        if isinstance(grodon, dict) and "predicted_doubling_time_hours" in grodon:
            row["Doubling time\n(gRodon2, h)"] = grodon["predicted_doubling_time_hours"]
            if "CUBHE" in grodon:
                row["CUBHE"] = grodon["CUBHE"]
            if "ConsistencyHE" in grodon:
                row["ConsistencyHE"] = grodon["ConsistencyHE"]

        hgt = sample_data[sid].get("hgt")
        if hgt is not None:
            if "hgt_flag" in hgt.columns:
                row["HGT genes"] = int(hgt["hgt_flag"].sum())
            row["Total genes"] = len(hgt)

        phage = sample_data[sid].get("phage_mobile")
        if phage is not None:
            for col in ("is_mobilome", "mobilome"):
                if col in phage.columns:
                    row["Mobilome genes"] = int(phage[col].astype(bool).sum())
                    break
            for col in ("is_phage_related", "is_phage", "phage_related"):
                if col in phage.columns:
                    row["Phage genes"] = int(phage[col].astype(bool).sum())
                    break

        operon = sample_data[sid].get("operon_coadaptation")
        if operon is not None and not operon.empty:
            if "mean_rscu_distance" in operon.columns:
                row["Operon CU\ncoadaptation"] = operon["mean_rscu_distance"].median()

        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metric_cols = [c for c in metrics_df.columns if c != "sample_id"
                   and metrics_df[c].notna().sum() >= 2]
    if not metric_cols:
        return

    n_panels = len(metric_cols)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    palette = sns.color_palette("Set2", len(sids))
    sid_colors = dict(zip(sids, palette))

    for i, col in enumerate(metric_cols):
        ax = axes_flat[i]
        vals = metrics_df.dropna(subset=[col])
        for _, row in vals.iterrows():
            ax.scatter(row[col], row["sample_id"], s=80,
                       color=sid_colors[row["sample_id"]],
                       edgecolors="black", linewidth=0.5, zorder=3)
        ax.set_xlabel(col, fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=0.3)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Genome metric comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_expression_tier_comparison(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Stacked bar chart comparing expression tier proportions across genomes.

    For each expression metric (CAI, MELP, Fop), shows the fraction of genes
    in high/medium/low tiers per genome.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    tier_data = []
    for sid in sids:
        expr = sample_data[sid].get("expression")
        if expr is None:
            continue
        for metric in ("CAI", "MELP", "Fop"):
            col = f"{metric}_class"
            if col not in expr.columns:
                continue
            counts = expr[col].value_counts()
            total = counts.sum()
            for tier in ("high", "medium", "low"):
                tier_data.append({
                    "sample_id": sid, "metric": metric, "tier": tier,
                    "fraction": counts.get(tier, 0) / total if total > 0 else 0,
                })

    if not tier_data:
        return

    df = pd.DataFrame(tier_data)
    metrics = df["metric"].unique()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics + 1, max(4, len(sids) * 0.6 + 1)))
    if n_metrics == 1:
        axes = [axes]

    tier_colors = {"high": "#e8853d", "medium": "#b0b0b0", "low": "#4a90d9"}

    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        sample_order = sids
        bottom = np.zeros(len(sample_order))
        for tier in ("high", "medium", "low"):
            tier_vals = []
            for sid in sample_order:
                match = sub[(sub["sample_id"] == sid) & (sub["tier"] == tier)]
                tier_vals.append(match["fraction"].iloc[0] if len(match) > 0 else 0)
            ax.barh(sample_order, tier_vals, left=bottom, height=0.6,
                    color=tier_colors[tier], label=tier, edgecolor="white", linewidth=0.3)
            bottom += np.array(tier_vals)
        ax.set_xlabel("Fraction of genes")
        ax.set_title(metric, fontsize=10)
        ax.set_xlim(0, 1)
        if ax == axes[0]:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Expression tier distribution", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_enrichment_comparison_heatmap(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Heatmap of pathway enrichment significance across genomes.

    Rows = KEGG pathways (union across all samples), columns = genomes.
    Cell colour = −log10(FDR).  Shows shared and genome-specific enrichments.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())

    # Collect all enrichment results, keeping only significant or top pathways
    all_rows = []
    for sid in sids:
        enrichments = sample_data[sid].get("enrichment", {})
        for key, edf in enrichments.items():
            if edf.empty or "fdr" not in edf.columns:
                continue
            metric_tier = key.replace("enrichment_", "")
            for _, row in edf.iterrows():
                pw_name = row.get("pathway_name", "") or row.get("pathway", "")
                all_rows.append({
                    "sample_id": sid,
                    "pathway": pw_name if pw_name else row.get("pathway", ""),
                    "fdr": row["fdr"],
                    "metric_tier": metric_tier,
                })

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    df["neg_log_fdr"] = -np.log10(df["fdr"].clip(lower=1e-20))

    # Pivot: pathway × sample_id, taking max significance across metric_tiers
    pivot = df.pivot_table(
        index="pathway", columns="sample_id", values="neg_log_fdr", aggfunc="max",
    ).reindex(columns=sids)

    # Keep top 30 pathways by max significance
    max_sig = pivot.max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[max_sig.head(30).index]

    if pivot.empty or pivot.shape[0] < 1:
        return

    fig_h = max(5, pivot.shape[0] * 0.35 + 2)
    fig_w = max(5, len(sids) * 1.5 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sig_line = -np.log10(0.05)
    sns.heatmap(
        pivot.fillna(0), ax=ax, cmap="YlOrRd", vmin=0,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "−log₁₀(FDR)", "shrink": 0.6},
        annot=pivot.shape[1] <= 5, fmt=".1f", annot_kws={"size": 7},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Pathway enrichment comparison (red line: FDR = 0.05)")
    ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_bio_ecology_comparison(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Multi-panel comparison of bio/ecology features across genomes.

    Panels: HGT Mahalanobis distribution, strand asymmetry significant codons,
    operon coadaptation distances, and optimal codon counts.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    palette = sns.color_palette("Set2", len(sids))
    sid_colors = dict(zip(sids, palette))

    panels = []

    # Panel 1: HGT Mahalanobis distance distributions
    hgt_data = []
    for sid in sids:
        hgt = sample_data[sid].get("hgt")
        if hgt is not None and "mahalanobis_dist" in hgt.columns:
            for v in hgt["mahalanobis_dist"].dropna():
                hgt_data.append({"sample_id": sid, "mahalanobis_dist": v})
    if len(hgt_data) > 0:
        panels.append(("hgt", pd.DataFrame(hgt_data)))

    # Panel 2: Strand asymmetry — count of significant codons per sample
    asym_data = []
    for sid in sids:
        sa = sample_data[sid].get("strand_asymmetry")
        if sa is not None and "significant" in sa.columns:
            n_sig = sa["significant"].sum()
            n_total = len(sa)
            asym_data.append({
                "sample_id": sid,
                "n_significant": int(n_sig),
                "fraction_significant": n_sig / n_total if n_total > 0 else 0,
            })
    if asym_data:
        panels.append(("strand_asym", pd.DataFrame(asym_data)))

    # Panel 3: Operon coadaptation — distribution of mean RSCU distances
    operon_data = []
    for sid in sids:
        op = sample_data[sid].get("operon_coadaptation")
        if op is not None and "mean_rscu_distance" in op.columns:
            for v in op["mean_rscu_distance"].dropna():
                operon_data.append({"sample_id": sid, "mean_rscu_distance": v})
    if operon_data:
        panels.append(("operon", pd.DataFrame(operon_data)))

    # Panel 4: Optimal codons — count per sample
    opt_data = []
    for sid in sids:
        oc = sample_data[sid].get("optimal_codons")
        if oc is not None and "is_optimal" in oc.columns:
            n_optimal = (oc["is_optimal"] >= 1).sum()
            n_top = (oc["is_optimal"] >= 2).sum()
            opt_data.append({
                "sample_id": sid,
                "n_optimal": int(n_optimal),
                "n_top_optimal": int(n_top),
            })
    if opt_data:
        panels.append(("optimal", pd.DataFrame(opt_data)))

    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, (ptype, pdf) in zip(axes, panels):
        if ptype == "hgt":
            sns.violinplot(data=pdf, x="sample_id", y="mahalanobis_dist",
                           palette=sid_colors, ax=ax, inner="box", cut=0, linewidth=0.8)
            ax.set_ylabel("Mahalanobis distance")
            ax.set_xlabel("")
            ax.set_title("HGT detection\n(CU deviation)", fontsize=10)
        elif ptype == "strand_asym":
            bars = [pdf.loc[pdf["sample_id"] == s, "n_significant"].values[0]
                    for s in sids if s in pdf["sample_id"].values]
            bar_sids = [s for s in sids if s in pdf["sample_id"].values]
            ax.bar(bar_sids, bars, color=[sid_colors[s] for s in bar_sids],
                   edgecolor="white", linewidth=0.5)
            ax.set_ylabel("Significant codons")
            ax.set_xlabel("")
            ax.set_title("Strand asymmetry", fontsize=10)
        elif ptype == "operon":
            sns.violinplot(data=pdf, x="sample_id", y="mean_rscu_distance",
                           palette=sid_colors, ax=ax, inner="box", cut=0, linewidth=0.8)
            ax.set_ylabel("Mean RSCU distance")
            ax.set_xlabel("")
            ax.set_title("Operon codon\ncoadaptation", fontsize=10)
        elif ptype == "optimal":
            bar_data = {s: 0 for s in sids}
            for _, row in pdf.iterrows():
                bar_data[row["sample_id"]] = row["n_optimal"]
            ax.bar(list(bar_data.keys()), list(bar_data.values()),
                   color=[sid_colors[s] for s in bar_data],
                   edgecolor="white", linewidth=0.5)
            ax.set_ylabel("Optimal codons")
            ax.set_xlabel("")
            ax.set_title("Translational\nselection", fontsize=10)

        ax.tick_params(axis="x", rotation=30, labelsize=8)

    fig.suptitle("Biological & ecological feature comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_hgt_mge_comparison(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Grouped bar chart comparing HGT, mobilome, and phage gene counts.

    Three bar groups per genome: HGT candidates, mobilome genes, phage genes.
    Provides a single-figure overview of foreign DNA burden across genomes.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    rows = []
    for sid in sids:
        row = {"sample_id": sid, "HGT candidates": 0, "Mobilome": 0, "Phage-related": 0}
        hgt = sample_data[sid].get("hgt")
        if hgt is not None and "hgt_flag" in hgt.columns:
            row["HGT candidates"] = int(hgt["hgt_flag"].sum())
        phage = sample_data[sid].get("phage_mobile")
        if phage is not None:
            for col in ("is_mobilome", "mobilome"):
                if col in phage.columns:
                    row["Mobilome"] = int(phage[col].astype(bool).sum())
                    break
            for col in ("is_phage_related", "is_phage", "phage_related"):
                if col in phage.columns:
                    row["Phage-related"] = int(phage[col].astype(bool).sum())
                    break
        rows.append(row)

    df = pd.DataFrame(rows)
    categories = ["HGT candidates", "Mobilome", "Phage-related"]
    has_data = any(df[c].sum() > 0 for c in categories)
    if not has_data:
        return

    x = np.arange(len(sids))
    bar_w = 0.25
    cat_colors = {"HGT candidates": "#d9a032", "Mobilome": "#e84040", "Phage-related": "#9467bd"}

    fig, ax = plt.subplots(figsize=(max(5, len(sids) * 2), 5))
    for i, cat in enumerate(categories):
        offset = (i - 1) * bar_w
        ax.bar(x + offset, df[cat].values, bar_w, label=cat,
               color=cat_colors[cat], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sids, rotation=30, ha="right")
    ax.set_ylabel("Gene count")
    ax.set_title("HGT and mobile genetic element comparison")
    ax.legend(fontsize=8)

    fig.tight_layout()
    _save_fig(fig, output_path)


def plot_rp_he_comparison_across_genomes(
    sample_data: dict[str, dict],
    output_path: Path,
):
    """Compare RP and high-expression RSCU profiles across genomes.

    For each genome, shows the Euclidean distance between its ribosomal
    and high-expression RSCU profiles as a horizontal bar, plus the
    correlation coefficient.  Reveals whether the RP/HE relationship
    is conserved or divergent across genomes.

    Args:
        sample_data: Per-sample data from _load_sample_data().
        output_path: Base path for saving.
    """
    _apply_style()
    sids = sorted(sample_data.keys())
    rows = []
    for sid in sids:
        rp_df = sample_data[sid].get("rscu_ribosomal")
        expr = sample_data[sid].get("expression")
        enc_data = sample_data[sid].get("enc")

        if rp_df is None or expr is None:
            continue

        rp_cols = [c for c in RSCU_COLUMN_NAMES if c in rp_df.columns]
        if len(rp_cols) < 10 or len(rp_df) == 0:
            continue

        rp_vals = pd.Series({c: rp_df[c].iloc[0] for c in rp_cols})

        # Load per-gene RSCU to compute HE median
        rscu_all_path = sample_data[sid].get("rscu_median")
        # We need per-gene RSCU to compute HE median — use rscu_median as genome reference
        genome_vals = None
        rscu_median_df = sample_data[sid].get("rscu_median")
        if rscu_median_df is not None and len(rscu_median_df) > 0:
            genome_vals = pd.Series({c: rscu_median_df[c].iloc[0] for c in rp_cols
                                     if c in rscu_median_df.columns})

        if genome_vals is None:
            continue

        # Euclidean distance RP vs genome
        shared = sorted(set(rp_vals.index) & set(genome_vals.index))
        if len(shared) < 10:
            continue

        rp_v = np.array([rp_vals[c] for c in shared])
        gn_v = np.array([genome_vals[c] for c in shared])
        mask = np.isfinite(rp_v) & np.isfinite(gn_v)

        if mask.sum() < 10:
            continue

        dist = np.sqrt(np.sum((rp_v[mask] - gn_v[mask]) ** 2))
        rho, _ = stats.spearmanr(rp_v[mask], gn_v[mask])

        rows.append({
            "sample_id": sid,
            "rp_genome_distance": dist,
            "rp_genome_rho": rho,
        })

    if len(rows) < 2:
        return

    df = pd.DataFrame(rows).sort_values("rp_genome_distance")
    palette = sns.color_palette("Set2", len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, max(4, len(df) * 0.5 + 1)))

    # Distance bars
    ax1.barh(df["sample_id"], df["rp_genome_distance"],
             color=palette, edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Euclidean distance\n(RP vs genome RSCU)")
    ax1.set_title("RP codon usage\ndivergence from genome")

    # Correlation dots
    ax2.scatter(df["rp_genome_rho"], df["sample_id"], s=80,
                color=palette, edgecolors="black", linewidth=0.5, zorder=3)
    ax2.axvline(1.0, color="gray", linestyle=":", alpha=0.4)
    ax2.set_xlabel("Spearman ρ\n(RP vs genome RSCU)")
    ax2.set_title("RP–genome\ncorrelation")
    ax2.set_xlim(min(df["rp_genome_rho"].min() - 0.05, 0.5), 1.05)

    fig.suptitle("Ribosomal protein codon usage across genomes", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, output_path)


def generate_pairwise_comparison_plots(
    sample_outputs: dict[str, dict[str, Path]],
    combined_rscu: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate qualitative comparison plots for small batches (≥2 genomes).

    These plots do not require statistical tests or condition columns.
    They provide visual comparisons of codon usage, expression, bio/ecology,
    and enrichment across all samples.

    Args:
        sample_outputs: Per-sample pipeline output paths.
        combined_rscu: Combined genome-level RSCU table.
        output_dir: Base output directory.

    Returns:
        Dict of output plot paths.
    """
    plot_dir = output_dir / "comparison" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    sample_data = _load_sample_data(sample_outputs)
    if len(sample_data) < 2:
        logger.info("SKIPPED: pairwise comparison plots (fewer than 2 samples)")
        return outputs

    logger.info("Generating qualitative comparison plots for %d genomes", len(sample_data))

    # RSCU overlay comparison
    try:
        p = plot_dir / "rscu_overlay"
        plot_pairwise_rscu_overlay(sample_data, p)
        outputs["rscu_overlay"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("RSCU overlay plot failed: %s", e)

    # RSCU difference heatmap
    try:
        p = plot_dir / "rscu_delta_heatmap"
        plot_pairwise_rscu_delta_heatmap(sample_data, p)
        outputs["rscu_delta_heatmap"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("RSCU delta heatmap failed: %s", e)

    # Genome metrics comparison
    try:
        p = plot_dir / "genome_metrics"
        plot_genome_metrics_comparison(sample_data, combined_rscu, p)
        outputs["genome_metrics"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Genome metrics comparison failed: %s", e)

    # Expression tier comparison
    try:
        p = plot_dir / "expression_tiers"
        plot_expression_tier_comparison(sample_data, p)
        outputs["expression_tiers"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Expression tier comparison failed: %s", e)

    # Enrichment comparison heatmap
    try:
        p = plot_dir / "enrichment_comparison"
        plot_enrichment_comparison_heatmap(sample_data, p)
        outputs["enrichment_comparison"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Enrichment comparison heatmap failed: %s", e)

    # Bio/ecology multi-panel comparison
    try:
        p = plot_dir / "bio_ecology_comparison"
        plot_bio_ecology_comparison(sample_data, p)
        outputs["bio_ecology_comparison"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("Bio/ecology comparison failed: %s", e)

    # HGT/MGE/phage comparison
    try:
        p = plot_dir / "hgt_mge_comparison"
        plot_hgt_mge_comparison(sample_data, p)
        outputs["hgt_mge_comparison"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("HGT/MGE comparison failed: %s", e)

    # RP vs HE across genomes
    try:
        p = plot_dir / "rp_he_across_genomes"
        plot_rp_he_comparison_across_genomes(sample_data, p)
        outputs["rp_he_across_genomes"] = p.with_suffix(".png")
    except Exception as e:
        logger.warning("RP/HE comparison across genomes failed: %s", e)

    # gRodon2 batch comparison
    grodon_data = {
        sid: d["grodon2"]
        for sid, d in sample_data.items()
        if "grodon2" in d and isinstance(d["grodon2"], dict)
    }
    if len(grodon_data) >= 2:
        try:
            cai_data = {
                sid: d["growth_rate"]
                for sid, d in sample_data.items()
                if "growth_rate" in d and isinstance(d["growth_rate"], dict)
            }
            p = plot_dir / "grodon2_comparison"
            plot_grodon2_batch_comparison(
                grodon_data, cai_data if cai_data else None, p,
            )
            outputs["grodon2_comparison"] = p.with_suffix(".png")
        except Exception as e:
            logger.warning("gRodon2 batch comparison plot failed: %s", e)
    else:
        logger.info("SKIPPED: gRodon2 batch comparison plot (fewer than 2 genomes with gRodon2 data)")

    logger.info("Generated %d pairwise comparison plots", len(outputs))
    return outputs
