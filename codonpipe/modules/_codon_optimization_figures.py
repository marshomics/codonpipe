"""Figures for the three-way codon-optimization comparison.

Three figures, each focused on one part of the synthesis-design narrative:

  render_three_way_rscu(...)
    Per-codon RSCU bars: genome / RP / Mahal-cluster, with disagreeing
    families highlighted. The headline visualization for "where do the
    three references disagree?".

  render_optimization_agreement(...)
    Compact per-AA-family table showing the RP-optimal codon vs the
    Mahal-optimal codon side by side, with cells colored by the magnitude
    of disagreement. Reads at a glance.

  render_optimization_gain(...)
    Per-gene cbi_rp vs cbi_mahal scatter, colored by Mahal cluster
    membership_score, with the diagonal line. Genes above the diagonal
    benefit more from Mahal-style optimization than RP-style. Plus a
    histogram of the gain distribution and the top-N genes-most-improved.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("codonpipe")


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        logger.warning("matplotlib unavailable; skipping figure (%s)", e)
        return None


def render_three_way_rscu(
    output_dir: Path,
    sample_id: str,
    table: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    """Per-codon RSCU bar chart with three references overlaid."""
    plt = _plt()
    if plt is None or table is None or table.empty:
        return (None, None)

    fig, axes = plt.subplots(
        nrows=2, ncols=1,
        figsize=(15, 7.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axTop, axBot = axes

    # Order codons by family then by codon name for consistent reading
    ordered = table.copy().reset_index(drop=True)
    ordered = ordered.sort_values(["family", "codon"]).reset_index(drop=True)
    n = len(ordered)
    x = np.arange(n)
    width = 0.27

    # Three sets of bars
    axTop.bar(x - width, ordered["genome_rscu"], width=width,
              color="#888888", edgecolor="none", label="genome mean")
    axTop.bar(x,         ordered["rp_rscu"], width=width,
              color="#1f77b4", edgecolor="none", label="ribosomal proteins")
    axTop.bar(x + width, ordered["mahal_rscu"], width=width,
              color="#2ca02c", edgecolor="none", label="Mahal cluster")

    # Highlight family backgrounds when RP and Mahal disagree on the optimum
    disagree_families = set(summary.loc[~summary["agree"], "family"]) \
        if (summary is not None and not summary.empty) else set()
    for family in disagree_families:
        idx = ordered.index[ordered["family"] == family].tolist()
        if not idx:
            continue
        x_lo = idx[0] - 0.5
        x_hi = idx[-1] + 0.5
        axTop.axvspan(x_lo, x_hi, color="#fff0d6", alpha=0.6, zorder=0)

    axTop.set_xticks(x)
    axTop.set_xticklabels(ordered["codon_col"], rotation=90, fontsize=6)
    axTop.set_ylabel("RSCU")
    axTop.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
    axTop.set_title(
        f"Three-way RSCU comparison — {sample_id}  "
        f"(orange-shaded families: RP and Mahal disagree on the optimal codon)",
        fontsize=11, fontweight="bold",
    )
    axTop.legend(loc="upper right", fontsize=8, frameon=False)
    for s in ("top", "right"):
        axTop.spines[s].set_visible(False)

    # Bottom: per-codon Δw (Mahal − RP), with a clear sign convention
    axBot.bar(
        x, ordered["delta_w_mahal_minus_rp"], width=0.7,
        color=["#2ca02c" if v >= 0 else "#1f77b4"
               for v in ordered["delta_w_mahal_minus_rp"].fillna(0)],
        edgecolor="none",
    )
    axBot.axhline(0, color="black", linewidth=0.5)
    axBot.set_xticks(x)
    axBot.set_xticklabels(ordered["codon_col"], rotation=90, fontsize=6)
    axBot.set_ylabel("Δw (Mahal − RP)")
    axBot.set_title(
        "Per-codon shift in relative adaptiveness when switching reference frame "
        "(positive = Mahal favours the codon more than RP does)",
        fontsize=10,
    )
    for s in ("top", "right"):
        axBot.spines[s].set_visible(False)

    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_three_way_rscu.png"
    svg = out_dir / f"{sample_id}_three_way_rscu.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


def render_optimization_agreement(
    output_dir: Path,
    sample_id: str,
    summary: pd.DataFrame,
    table: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    """Compact per-AA-family agreement display.

    Two-row mini-heatmaps per family showing the chosen codon under each
    reference frame, plus the magnitude of the disagreement. Disagreeing
    families are highlighted on the right with a coloured strip.
    """
    plt = _plt()
    if plt is None or summary is None or summary.empty:
        return (None, None)

    fams = summary.copy().sort_values(
        ["agree", "max_codon_w_shift"], ascending=[True, False],
    ).reset_index(drop=True)

    n = len(fams)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.35 + 1)),
                           constrained_layout=True)

    ypos = np.arange(n)[::-1]
    cmap = plt.get_cmap("Reds")
    max_shift = max(0.01, fams["max_codon_w_shift"].max(skipna=True) or 0.01)

    for i, (_, r) in enumerate(fams.iterrows()):
        y = ypos[i]
        # Family label
        ax.text(-0.02, y, f"{r['family']} ({r['amino_acid']})",
                ha="right", va="center", fontsize=8, fontweight="bold",
                transform=ax.get_yaxis_transform())
        # RP-optimal box
        ax.add_patch(plt.Rectangle((0.02, y - 0.35), 0.18, 0.7,
                                    color="#1f77b4", alpha=0.85,
                                    transform=ax.get_yaxis_transform(), clip_on=False))
        ax.text(0.11, y, f"RP: {r['rp_optimal_codon']}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold",
                transform=ax.get_yaxis_transform())
        # Mahal-optimal box
        ax.add_patch(plt.Rectangle((0.22, y - 0.35), 0.18, 0.7,
                                    color="#2ca02c", alpha=0.85,
                                    transform=ax.get_yaxis_transform(), clip_on=False))
        ax.text(0.31, y, f"Mahal: {r['mahal_optimal_codon']}",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold",
                transform=ax.get_yaxis_transform())
        # Agreement marker
        symbol = "✓" if r["agree"] else "✗"
        agree_color = "#2ca02c" if r["agree"] else "#d62728"
        ax.text(0.43, y, symbol, ha="center", va="center",
                fontsize=14, fontweight="bold", color=agree_color,
                transform=ax.get_yaxis_transform())
        # Shift magnitude bar
        shift_norm = (r["max_codon_w_shift"] / max_shift) if max_shift > 0 else 0
        ax.add_patch(plt.Rectangle(
            (0.48, y - 0.3), 0.4 * shift_norm, 0.6,
            color=cmap(0.3 + 0.7 * shift_norm),
            transform=ax.get_yaxis_transform(), clip_on=False,
        ))
        ax.text(0.89, y, f"max Δw = {r['max_codon_w_shift']:.2f}",
                ha="left", va="center", fontsize=7,
                transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    n_disagree = int((~fams["agree"]).sum())
    ax.set_title(
        f"Per-AA-family optimal codon: RP vs Mahal-cluster — {sample_id}\n"
        f"{n_disagree} of {n} families disagree on the optimum"
        + ("; rows ordered with disagreements first" if n_disagree else ""),
        fontsize=11, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_optimization_agreement.png"
    svg = out_dir / f"{sample_id}_optimization_agreement.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)


def render_optimization_gain(
    output_dir: Path,
    sample_id: str,
    gain_df: pd.DataFrame,
    *,
    top_n: int = 15,
) -> tuple[Path | None, Path | None]:
    """Per-gene optimization-gain visualization.

    Three panels:
      A. Scatter of cbi_rp vs cbi_mahal, colored by membership_score when
         available (else by in_optimized_set). Diagonal line shows where
         the two CBIs agree. Genes above the line gain from Mahal-style
         optimization.
      B. Histogram of (cbi_mahal − cbi_rp) with the median, IQR, and the
         fraction of genes that benefit from Mahal optimization.
      C. Bar chart of the top-N genes with the largest positive gain.
    """
    plt = _plt()
    if plt is None or gain_df is None or gain_df.empty:
        return (None, None)
    if "cbi_rp" not in gain_df.columns or "cbi_mahal" not in gain_df.columns:
        return (None, None)

    df = gain_df.dropna(subset=["cbi_rp", "cbi_mahal"]).copy()
    if df.empty:
        return (None, None)

    fig = plt.figure(figsize=(15, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.3])
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # Panel A: scatter
    color_col = None
    if "membership_score" in df.columns and df["membership_score"].notna().any():
        color_col = "membership_score"
    elif "in_optimized_set" in df.columns:
        color_col = "in_optimized_set"

    if color_col == "membership_score":
        vmin, vmax = np.nanpercentile(df["membership_score"].values, [2, 98])
        sc = axA.scatter(
            df["cbi_rp"], df["cbi_mahal"],
            c=df["membership_score"], cmap="viridis", s=14,
            alpha=0.75, edgecolor="black", linewidth=0.15,
            vmin=vmin, vmax=vmax,
        )
        fig.colorbar(sc, ax=axA, fraction=0.045, pad=0.02,
                     label="Mahal cluster membership_score")
    elif color_col == "in_optimized_set":
        in_set = df["in_optimized_set"].fillna(False).astype(bool)
        axA.scatter(df.loc[~in_set, "cbi_rp"], df.loc[~in_set, "cbi_mahal"],
                    s=12, alpha=0.6, color="#888888", edgecolor="none",
                    label=f"out of cluster (n={int((~in_set).sum())})")
        axA.scatter(df.loc[in_set, "cbi_rp"], df.loc[in_set, "cbi_mahal"],
                    s=20, alpha=0.85, color="#2ca02c",
                    edgecolor="black", linewidth=0.3,
                    label=f"in Mahal cluster (n={int(in_set.sum())})")
        axA.legend(loc="best", fontsize=7, frameon=False)
    else:
        axA.scatter(df["cbi_rp"], df["cbi_mahal"], s=10, alpha=0.5, color="#1f77b4")

    # Diagonal reference
    lo = float(min(df["cbi_rp"].min(), df["cbi_mahal"].min()))
    hi = float(max(df["cbi_rp"].max(), df["cbi_mahal"].max()))
    axA.plot([lo, hi], [lo, hi], color="black", linewidth=0.6, linestyle="--",
             alpha=0.6, label="cbi_rp = cbi_mahal")
    axA.set_xlabel("CBI (RP-derived optimal codons)")
    axA.set_ylabel("CBI (Mahal-cluster-derived optimal codons)")
    axA.set_title("A. Per-gene CBI: which reference fits better?", fontsize=10)
    for s in ("top", "right"):
        axA.spines[s].set_visible(False)

    # Panel B: gain distribution
    gains = df["gain_mahal_minus_rp"].dropna()
    if len(gains) > 0:
        axB.hist(gains, bins=40, color="#9467bd", edgecolor="white", linewidth=0.4)
        axB.axvline(0, color="black", linewidth=0.6)
        med = float(gains.median())
        axB.axvline(med, color="#d62728", linewidth=1.2, linestyle="--",
                    label=f"median = {med:+.3f}")
        frac_pos = float((gains > 0).mean()) * 100
        axB.text(
            0.02, 0.98,
            f"{frac_pos:.1f}% of genes benefit from\nMahal-style optimization",
            transform=axB.transAxes, ha="left", va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
        )
        axB.set_xlabel("CBI gain from Mahal vs RP\n(cbi_mahal − cbi_rp)")
        axB.set_ylabel("Number of genes")
        axB.set_title("B. Distribution of optimization gain", fontsize=10)
        axB.legend(loc="upper right", fontsize=8, frameon=False)
        for s in ("top", "right"):
            axB.spines[s].set_visible(False)

    # Panel C: top-N most-improved genes
    df_sorted = df.dropna(subset=["gain_mahal_minus_rp"]).sort_values(
        "gain_mahal_minus_rp", ascending=False,
    ).head(top_n)
    if not df_sorted.empty:
        ypos = np.arange(len(df_sorted))[::-1]
        axC.barh(ypos, df_sorted["gain_mahal_minus_rp"],
                 color="#2ca02c", edgecolor="black", linewidth=0.4)
        axC.set_yticks(ypos)
        axC.set_yticklabels(df_sorted["gene"].astype(str), fontsize=7)
        axC.axvline(0, color="black", linewidth=0.5)
        axC.set_xlabel("CBI gain (cbi_mahal − cbi_rp)")
        axC.set_title(f"C. Top {len(df_sorted)} genes most improved by Mahal optimization",
                      fontsize=10)
        for s in ("top", "right", "left"):
            axC.spines[s].set_visible(False)

    fig.suptitle(
        f"Codon-optimization gain — {sample_id}\n"
        "Mahal-derived optimal codons vs RP-derived optimal codons",
        fontsize=12, fontweight="bold",
    )

    out_dir = Path(output_dir)
    png = out_dir / f"{sample_id}_optimization_gain.png"
    svg = out_dir / f"{sample_id}_optimization_gain.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)
