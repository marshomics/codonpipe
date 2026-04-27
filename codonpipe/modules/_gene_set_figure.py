"""Seven-panel summary figure for the gene_set module.

Kept in its own file so the analytic core (gene_set.py) doesn't import
matplotlib at module load time. Importing matplotlib is heavy and not
something we want as a side effect of running the test suite.

Panels (2 rows × 4 columns; the heatmap spans the last two cells of the
bottom row to give the column labels enough horizontal space):

    A — Concatenated/mean RSCU bar chart: GOI vs genome vs RP vs Mahal.
    B — Effect-size forest plot (Cliff's delta with 95% bootstrap CI).
    C — Correspondence-analysis biplot with GOI highlighted.
    D — Wright ENC-vs-GC3 plot with the neutral curve.
    E — Genome-centroid Mahalanobis distance distribution; GOI marked.
    F — Genome-centroid vs cluster-centroid Mahalanobis biplot. Separates
        the "unusual relative to the bulk genome" axis from the "near the
        translationally optimized core" axis.
    G — Per-GOI heatmap of percentile-rank metrics with a Mahal-cluster
        membership strip.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe.utils.codon_tables import (
    AMINO_ACID_FAMILIES,
    COL_GENE,
    RSCU_COLUMN_NAMES,
)

logger = logging.getLogger("codonpipe")


def _wright_enc(gc3: np.ndarray) -> np.ndarray:
    """Wright (1990) neutral expectation: Nc(GC3) = 2 + GC3 + 29 / (GC3^2 + (1-GC3)^2)."""
    s = np.clip(gc3, 1e-6, 1 - 1e-6)
    return 2 + s + 29.0 / (s * s + (1 - s) * (1 - s))


def render_summary_figure(
    output_dir: Path,
    sample_id: str,
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame | None,
    expr_df: pd.DataFrame | None,
    hgt_df: pd.DataFrame | None,
    summary_df: pd.DataFrame,
    scalar_tests: pd.DataFrame,
    codon_tests: pd.DataFrame,
    goi_ids: set[str],
    rscu_genome: dict[str, float] | None,
    rscu_rp: dict[str, float] | None,
    rscu_mahal_cluster: dict[str, float] | None,
    mahal_cluster_df: pd.DataFrame | None = None,
) -> tuple[Path | None, Path | None]:
    """Render the 7-panel figure to PNG + SVG. Returns (png_path, svg_path) or (None, None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning("matplotlib unavailable; skipping figure (%s)", e)
        return (None, None)

    fig = plt.figure(figsize=(22, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    # Top row: A–D (RSCU bars, forest, COA, Wright)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[0, 3])
    # Bottom row: E (Mahal hist), F (NEW Mahal biplot), G (heatmap, double-wide)
    axE = fig.add_subplot(gs[1, 0])
    axF = fig.add_subplot(gs[1, 1])
    axG = fig.add_subplot(gs[1, 2:])

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    in_goi = rscu_gene_df[COL_GENE].isin(goi_ids)
    goi_rscu = rscu_gene_df[in_goi]
    bg_rscu = rscu_gene_df[~in_goi]

    # ── Panel A: RSCU bar chart, GOI vs genome vs RP vs Mahal ─────────────
    try:
        _panel_rscu_bars(axA, goi_rscu, rscu_genome, rscu_rp, rscu_mahal_cluster, rscu_cols)
    except Exception as e:
        logger.warning("Panel A (RSCU bars) failed: %s", e)
        axA.text(0.5, 0.5, "Panel A unavailable", ha="center", va="center", transform=axA.transAxes)

    # ── Panel B: ECDF overlay of scalar metrics ───────────────────────────
    try:
        _panel_ecdfs(axB, summary_df, scalar_tests, enc_df, expr_df, hgt_df, in_goi, rscu_gene_df)
    except Exception as e:
        logger.warning("Panel B (ECDFs) failed: %s", e)
        axB.text(0.5, 0.5, "Panel B unavailable", ha="center", va="center", transform=axB.transAxes)

    # ── Panel C: COA biplot with GOI highlighted ──────────────────────────
    try:
        _panel_coa(axC, rscu_gene_df, in_goi)
    except Exception as e:
        logger.warning("Panel C (COA) failed: %s", e)
        axC.text(0.5, 0.5, "Panel C unavailable", ha="center", va="center", transform=axC.transAxes)

    # ── Panel D: Wright ENC vs GC3 ────────────────────────────────────────
    try:
        _panel_wright(axD, enc_df, in_goi, rscu_gene_df)
    except Exception as e:
        logger.warning("Panel D (Wright) failed: %s", e)
        axD.text(0.5, 0.5, "Panel D unavailable", ha="center", va="center", transform=axD.transAxes)

    # ── Panel E: Mahalanobis distance histogram ───────────────────────────
    try:
        _panel_mahal(axE, hgt_df, in_goi, rscu_gene_df)
    except Exception as e:
        logger.warning("Panel E (Mahalanobis) failed: %s", e)
        axE.text(0.5, 0.5, "Panel E unavailable", ha="center", va="center", transform=axE.transAxes)

    # ── Panel F: Mahalanobis biplot (genome vs cluster centroid) ──────────
    try:
        _panel_mahal_biplot(axF, hgt_df, mahal_cluster_df, in_goi, rscu_gene_df)
    except Exception as e:
        logger.warning("Panel F (Mahal biplot) failed: %s", e)
        axF.text(0.5, 0.5, "Panel F unavailable", ha="center", va="center", transform=axF.transAxes)

    # ── Panel G: Per-GOI heatmap ──────────────────────────────────────────
    try:
        _panel_heatmap(axG, summary_df, fig)
    except Exception as e:
        logger.warning("Panel G (heatmap) failed: %s", e)
        axG.text(0.5, 0.5, "Panel G unavailable", ha="center", va="center", transform=axG.transAxes)

    fig.suptitle(
        f"Gene-set vs genome — {sample_id}  (n_goi = {int(in_goi.sum())}, "
        f"n_background = {int((~in_goi).sum())})",
        fontsize=14, fontweight="bold",
    )

    png_path = Path(output_dir) / f"{sample_id}_goi_panel.png"
    svg_path = Path(output_dir) / f"{sample_id}_goi_panel.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote summary figure: %s", png_path)
    return (png_path, svg_path)


# ──────────────────────────────────────────────────────────────────────────────
# Panel implementations
# ──────────────────────────────────────────────────────────────────────────────


def _panel_rscu_bars(ax, goi_rscu_df, ref_genome, ref_rp, ref_mahal, rscu_cols):
    """Stacked-by-AA bar chart of GOI mean vs reference RSCU values."""
    if goi_rscu_df.empty or not rscu_cols:
        ax.text(0.5, 0.5, "No RSCU data", ha="center", va="center", transform=ax.transAxes)
        return

    # GOI mean per codon
    goi_mean = goi_rscu_df[rscu_cols].mean()

    # Build long-form DataFrame for plotting
    series = {"GOI": goi_mean}
    if ref_genome is not None:
        series["genome"] = pd.Series({c: ref_genome.get(c, np.nan) for c in rscu_cols})
    if ref_rp is not None:
        series["RP"] = pd.Series({c: ref_rp.get(c, np.nan) for c in rscu_cols})
    if ref_mahal is not None:
        series["Mahal"] = pd.Series({c: ref_mahal.get(c, np.nan) for c in rscu_cols})

    n_series = len(series)
    x_pos = np.arange(len(rscu_cols))
    bar_w = 0.8 / n_series
    colors = {"GOI": "#d62728", "genome": "#7f7f7f", "RP": "#1f77b4", "Mahal": "#2ca02c"}
    for i, (label, vec) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * bar_w
        ax.bar(
            x_pos + offset, vec.values, width=bar_w,
            label=label, color=colors.get(label, "#888888"), edgecolor="none",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(rscu_cols, rotation=90, fontsize=6)
    ax.set_ylabel("Mean RSCU")
    ax.set_title("A. RSCU: GOI vs reference sets")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)


def _panel_ecdfs(ax, summary_df, scalar_tests, enc_df, expr_df, hgt_df, in_goi, rscu_gene_df):
    """Forest plot of Cliff's delta with 95% CI per metric."""
    if scalar_tests is None or scalar_tests.empty:
        ax.text(0.5, 0.5, "No scalar tests", ha="center", va="center", transform=ax.transAxes)
        return

    df = scalar_tests.sort_values("cliffs_delta", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    delta = df["cliffs_delta"].values
    lo = df["delta_ci_low"].values
    hi = df["delta_ci_high"].values
    sig = df["significant"].values if "significant" in df.columns else np.zeros(len(df), bool)

    colors = ["#d62728" if s else "#888888" for s in sig]
    # matplotlib's errorbar accepts a single ecolor; loop per row to honour
    # the per-point significance colouring while keeping CIs as proper bars.
    for yi, di, lo_i, hi_i, c in zip(y, delta, lo, hi, colors):
        ax.errorbar(
            di, yi,
            xerr=[[di - lo_i], [hi_i - di]],
            fmt="none", ecolor=c, capsize=3, linewidth=1.2,
        )
    ax.scatter(delta, y, color=colors, s=30, zorder=3)
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(df["metric"], fontsize=8)
    ax.set_xlabel("Cliff's delta (GOI − background) ± 95% bootstrap CI")
    ax.set_title("B. Effect size per metric (red = BH-significant)")
    # Reference effect-size thresholds
    for thr, lbl in [(0.147, "small"), (0.33, "medium"), (0.474, "large")]:
        ax.axvline(thr, color="black", linewidth=0.3, alpha=0.3)
        ax.axvline(-thr, color="black", linewidth=0.3, alpha=0.3)


def _panel_coa(ax, rscu_gene_df, in_goi):
    """COA biplot, GOI overlaid in red on background scatter."""
    from codonpipe.modules.advanced_analyses import compute_coa_on_rscu

    coa = compute_coa_on_rscu(rscu_gene_df)
    if not coa or "coa_coords" not in coa or coa["coa_coords"].empty:
        ax.text(0.5, 0.5, "COA not computable", ha="center", va="center", transform=ax.transAxes)
        return

    coords = coa["coa_coords"].copy()
    inertia = coa.get("coa_inertia")
    pct = inertia["pct_inertia"].values if inertia is not None and not inertia.empty else [0, 0]

    coords["in_goi"] = coords[COL_GENE].isin(rscu_gene_df.loc[in_goi, COL_GENE])
    bg = coords[~coords["in_goi"]]
    g = coords[coords["in_goi"]]
    ax.scatter(bg["Axis1"], bg["Axis2"], s=4, alpha=0.25, color="#888888", label="background")
    ax.scatter(g["Axis1"], g["Axis2"], s=30, alpha=0.9, color="#d62728",
               edgecolor="black", linewidth=0.4, label="GOI")
    ax.set_xlabel(f"Axis 1 ({pct[0]:.1f}% of inertia)")
    ax.set_ylabel(f"Axis 2 ({pct[1]:.1f}% of inertia)" if len(pct) > 1 else "Axis 2")
    ax.set_title("C. Correspondence-analysis biplot")
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.axhline(0, color="black", linewidth=0.3, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.3, alpha=0.4)


def _panel_wright(ax, enc_df, in_goi, rscu_gene_df):
    """ENC vs GC3 with Wright neutrality curve and GOI highlighted."""
    if enc_df is None or enc_df.empty:
        ax.text(0.5, 0.5, "No ENC data", ha="center", va="center", transform=ax.transAxes)
        return
    enc_df = enc_df.copy()
    goi_genes = set(rscu_gene_df.loc[in_goi, COL_GENE])
    enc_df["in_goi"] = enc_df[COL_GENE].isin(goi_genes)

    bg = enc_df[~enc_df["in_goi"]]
    g = enc_df[enc_df["in_goi"]]
    ax.scatter(bg["GC3"], bg["ENC"], s=4, alpha=0.2, color="#888888", label="background")
    ax.scatter(g["GC3"], g["ENC"], s=30, alpha=0.9, color="#d62728",
               edgecolor="black", linewidth=0.4, label="GOI")

    xs = np.linspace(0.05, 0.95, 200)
    ax.plot(xs, _wright_enc(xs), color="black", linewidth=1.0,
            linestyle="--", label="Wright 1990 neutral")
    ax.set_xlabel("GC3")
    ax.set_ylabel("ENC")
    ax.set_ylim(20, 62)
    ax.set_xlim(0.05, 0.95)
    ax.set_title("D. ENC vs GC3 (Wright plot)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def _panel_mahal(ax, hgt_df, in_goi, rscu_gene_df):
    """Mahalanobis distance histogram with GOI marked."""
    if hgt_df is None or hgt_df.empty or "mahalanobis_dist" not in hgt_df.columns:
        ax.text(0.5, 0.5, "No HGT data", ha="center", va="center", transform=ax.transAxes)
        return
    hgt = hgt_df.copy()
    goi_genes = set(rscu_gene_df.loc[in_goi, COL_GENE])
    hgt["in_goi"] = hgt[COL_GENE].isin(goi_genes)
    bg_dists = hgt.loc[~hgt["in_goi"], "mahalanobis_dist"].dropna().values
    g_dists = hgt.loc[hgt["in_goi"], "mahalanobis_dist"].dropna().values

    if len(bg_dists) == 0:
        ax.text(0.5, 0.5, "No background distances", ha="center", va="center", transform=ax.transAxes)
        return

    bins = np.linspace(0, max(bg_dists.max(), g_dists.max() if len(g_dists) else 0) + 0.5, 50)
    ax.hist(bg_dists, bins=bins, color="#888888", alpha=0.65, label="background", edgecolor="none")
    if len(g_dists):
        # Overlay GOI as a rug + transparent overlay histogram
        ax.hist(g_dists, bins=bins, color="#d62728", alpha=0.55, label="GOI", edgecolor="none")
        ymax = ax.get_ylim()[1]
        ax.vlines(g_dists, ymin=0, ymax=ymax * 0.05, color="#d62728", linewidth=1.2)
    # 95th percentile of background as a cutoff hint
    cutoff = np.quantile(bg_dists, 0.95)
    ax.axvline(cutoff, color="black", linewidth=0.6, linestyle="--",
               label=f"95th pctile ({cutoff:.2f})")
    ax.set_xlabel("Mahalanobis distance from genome centroid")
    ax.set_ylabel("# genes")
    ax.set_title("E. Mahalanobis distance distribution")
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def _panel_mahal_biplot(ax, hgt_df, mahal_cluster_df, in_goi, rscu_gene_df):
    """Genome-centroid Mahalanobis vs cluster-centroid Mahalanobis biplot.

    Separates the two questions that the codebase used to conflate:
        x-axis: how unusual is this gene's codon usage relative to the bulk
                genome? (the HGT-detector signal)
        y-axis: how far is this gene from the translationally optimized
                cluster centroid? (the Mahal-clustering signal)
    Background genes are coloured by in_optimized_set, GOIs are overlaid in red.
    """
    if hgt_df is None or hgt_df.empty or "mahalanobis_dist" not in hgt_df.columns:
        ax.text(0.5, 0.5, "HGT distances unavailable",
                ha="center", va="center", transform=ax.transAxes)
        return
    if mahal_cluster_df is None or mahal_cluster_df.empty or \
            "mahal_cluster_distance" not in mahal_cluster_df.columns:
        ax.text(0.5, 0.5, "Mahal-cluster distances unavailable",
                ha="center", va="center", transform=ax.transAxes)
        return

    merged = hgt_df[[COL_GENE, "mahalanobis_dist"]].merge(
        mahal_cluster_df[[COL_GENE, "mahal_cluster_distance", "in_optimized_set"]],
        on=COL_GENE, how="inner",
    )
    merged = merged.dropna(subset=["mahalanobis_dist", "mahal_cluster_distance"])
    if merged.empty:
        ax.text(0.5, 0.5, "No genes with both distances",
                ha="center", va="center", transform=ax.transAxes)
        return

    goi_genes = set(rscu_gene_df.loc[in_goi, COL_GENE])
    merged["in_goi"] = merged[COL_GENE].isin(goi_genes)

    # Background genes split by Mahal-cluster membership
    bg = merged[~merged["in_goi"]]
    bg_in = bg[bg["in_optimized_set"].fillna(False).astype(bool)]
    bg_out = bg[~bg["in_optimized_set"].fillna(False).astype(bool)]
    g = merged[merged["in_goi"]]

    ax.scatter(
        bg_out["mahalanobis_dist"], bg_out["mahal_cluster_distance"],
        s=4, alpha=0.25, color="#888888", label="bg, not in cluster",
    )
    ax.scatter(
        bg_in["mahalanobis_dist"], bg_in["mahal_cluster_distance"],
        s=6, alpha=0.55, color="#2ca02c", label="bg, in cluster",
    )
    if not g.empty:
        # Plot GOI with edge-color encoding cluster membership so the user
        # sees both "is this a GOI?" and "is the GOI in the cluster?" at once.
        g_in = g[g["in_optimized_set"].fillna(False).astype(bool)]
        g_out = g[~g["in_optimized_set"].fillna(False).astype(bool)]
        if not g_out.empty:
            ax.scatter(
                g_out["mahalanobis_dist"], g_out["mahal_cluster_distance"],
                s=50, color="#d62728", edgecolor="black", linewidth=0.8,
                label="GOI, not in cluster", zorder=3,
            )
        if not g_in.empty:
            ax.scatter(
                g_in["mahalanobis_dist"], g_in["mahal_cluster_distance"],
                s=60, marker="*", color="#d62728", edgecolor="black", linewidth=0.8,
                label="GOI, in cluster", zorder=4,
            )

    # Reference lines: 95th percentile of bg genome-centroid Mahalanobis
    # (the HGT-detector neighbourhood) and 95th percentile of bg cluster-
    # centroid Mahalanobis (the optimized-core neighbourhood).
    if len(bg) > 10:
        x95 = float(np.quantile(bg["mahalanobis_dist"], 0.95))
        y95 = float(np.quantile(bg["mahal_cluster_distance"], 0.95))
        ax.axvline(x95, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axhline(y95, color="black", linewidth=0.5, linestyle=":", alpha=0.6)
        ax.text(
            x95, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 1,
            f" 95% genome", fontsize=7, color="black", alpha=0.7,
            ha="left", va="top",
        )

    ax.set_xlabel("Mahalanobis dist. (genome centroid)\n← typical          unusual →")
    ax.set_ylabel("Mahalanobis dist. (cluster centroid)\n← in core          far from core →")
    ax.set_title("F. Genome- vs cluster-centroid distance biplot")
    ax.legend(loc="best", fontsize=7, frameon=False)


def _panel_heatmap(ax, summary_df, fig):
    """Per-GOI heatmap of z-scored percentile-rank columns.

    When 'in_optimized_set' is in summary_df, an annotation strip is drawn
    on the right edge marking which GOI rows belong to the Mahalanobis
    optimized cluster — a quick visual answer to "which of these genes are
    part of the translationally optimized core?".
    """
    if summary_df is None or summary_df.empty:
        ax.text(0.5, 0.5, "No GOI summary data", ha="center", va="center", transform=ax.transAxes)
        return

    pctile_cols = [c for c in summary_df.columns if c.endswith("_pctile")]
    if not pctile_cols:
        ax.text(0.5, 0.5, "No percentile columns", ha="center", va="center", transform=ax.transAxes)
        return

    # Limit to a sensible subset of metrics for readability
    preferred = [
        "CAI_pctile", "MELP_pctile", "Fop_pctile",
        "ENC_pctile", "ENCprime_pctile", "MILC_pctile", "GC3_pctile",
        "mahalanobis_dist_pctile",
        "mahal_cluster_distance_pctile",
        "membership_score_pctile",
        "cbi_rp_pctile", "cbi_mahal_pctile",
        "length_pctile",
    ]
    cols = [c for c in preferred if c in pctile_cols] or pctile_cols
    mat = summary_df[cols].values.astype(float)
    if mat.size == 0:
        ax.text(0.5, 0.5, "Empty matrix", ha="center", va="center", transform=ax.transAxes)
        return

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_pctile", "") for c in cols],
                       rotation=45, ha="right", fontsize=8)
    gene_labels = summary_df[COL_GENE].astype(str).values
    if len(gene_labels) <= 30:
        ax.set_yticks(np.arange(len(gene_labels)))
        ax.set_yticklabels(gene_labels, fontsize=7)
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(gene_labels)} GOI genes")

    # Mahal-cluster membership strip
    if "in_optimized_set" in summary_df.columns:
        membership = summary_df["in_optimized_set"].fillna(False).astype(bool).values
        # Draw markers at x = right edge (just past the heatmap) with green for in, grey for out
        x_pos = len(cols) + 0.5
        for i, m in enumerate(membership):
            color = "#2ca02c" if m else "#cccccc"
            ax.scatter(x_pos, i, marker="s", s=40, color=color,
                       edgecolor="black", linewidth=0.4, clip_on=False)
        ax.text(x_pos, -0.7, "in\nMahal", ha="center", va="bottom", fontsize=7)

    ax.set_title("F. Per-GOI percentile rank (right strip: in Mahal cluster)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="percentile rank")
