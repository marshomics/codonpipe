"""Adaptive Codon Enrichment (ACE) iterative convergence module.

Derives an endogenous, genome-specific "core selected" codon usage table
from CodonPipe analysis outputs, without requiring external reference
databases or organism-specific tuning.

Three independent seeds are used and their convergence is checked:
  (a) Top genes by ENC' residual (composition-corrected bias)
  (b) Top genes by tRNA adaptation index (if tRNA counts available)
  (c) Annotated ribosomal protein genes (if RP gene list available)

If all seeds converge to the same table (cosine similarity > 0.95),
the consensus is reported.  If they diverge, the genome likely has
multi-modal codon usage, and all tables are reported with a warning.
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
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("codonpipe")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum gene length (nt) to include.  Matches the pipeline-wide default;
# kept here as a module-level constant so callers can override via the
# run_ace_convergence() ``min_gene_length`` parameter.
_DEFAULT_MIN_GENE_LENGTH = 240

# Mahalanobis percentile cutoff for HGT pre-filtering.
_MAHA_PERCENTILE = 0.95

# Convergence criteria
_MAX_ITER = 30
_JACCARD_THRESHOLD = 0.95   # S_k vs S_{k+1} Jaccard similarity
_COSINE_CONVERGENCE = 0.95  # cross-seed agreement threshold


# ---------------------------------------------------------------------------
# RSCU matrix helpers
# ---------------------------------------------------------------------------

def _parse_codon_col(col: str) -> tuple[str, str]:
    """Parse 'Phe-UUU' -> ('Phe', 'UUU')."""
    parts = col.split("-")
    return parts[0], parts[-1]


def _group_codons_by_aa(codon_cols: list[str]) -> dict[str, list[int]]:
    """Map amino acid family -> list of column indices."""
    groups: dict[str, list[int]] = {}
    for i, col in enumerate(codon_cols):
        aa, _ = _parse_codon_col(col)
        groups.setdefault(aa, []).append(i)
    return groups


def _extract_rscu_matrix(
    rscu_gene_df: pd.DataFrame,
    min_gene_length: int = _DEFAULT_MIN_GENE_LENGTH,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract numeric RSCU matrix from the per-gene DataFrame.

    Args:
        rscu_gene_df: DataFrame with 'gene', 'length', and codon columns
            (e.g. 'Phe-UUU').
        min_gene_length: Exclude genes shorter than this (nucleotides).

    Returns:
        rscu_mat:     (n_genes, n_codons) float array
        gene_lengths: (n_genes,) int array
        gene_ids:     list of gene identifier strings
        codon_cols:   list of RSCU column names
    """
    df = rscu_gene_df.copy()
    if "length" in df.columns:
        df = df[df["length"] >= min_gene_length].reset_index(drop=True)

    gene_ids = df["gene"].astype(str).tolist()
    gene_lengths = df["length"].values.astype(int) if "length" in df.columns else np.ones(len(df), dtype=int)

    codon_cols = [c for c in df.columns if "-" in c and c not in ("gene", "length")]
    rscu_mat = df[codon_cols].apply(pd.to_numeric, errors="coerce").values

    return rscu_mat, gene_lengths, gene_ids, codon_cols


# ---------------------------------------------------------------------------
# Codon weight table construction
# ---------------------------------------------------------------------------

def build_weight_table(
    rscu_mat: np.ndarray,
    gene_lengths: np.ndarray,
    gene_mask: np.ndarray,
    codon_cols: list[str],
) -> np.ndarray:
    """Build a codon weight table from a subset of genes.

    For each amino acid family, the weight of each synonym is its relative
    frequency within the family, computed from the length-weighted mean RSCU
    across the selected genes.  Weights sum to 1.0 within each family.

    Returns:
        weights: (n_codons,) array, values in [0, 1] summing to 1 per AA family
    """
    subset = rscu_mat[gene_mask]
    lengths = gene_lengths[gene_mask].astype(float)

    clean = np.nan_to_num(subset, nan=0.0)
    weighted_sum = (clean * lengths[:, np.newaxis]).sum(axis=0)
    total_weight = np.where(np.isnan(subset), 0.0, 1.0)
    total_weight = (total_weight * lengths[:, np.newaxis]).sum(axis=0)
    mean_rscu = np.where(total_weight > 0, weighted_sum / total_weight, 0.0)

    aa_groups = _group_codons_by_aa(codon_cols)
    weights = np.zeros(len(codon_cols))
    for aa, indices in aa_groups.items():
        family_sum = mean_rscu[indices].sum()
        if family_sum > 0:
            weights[indices] = mean_rscu[indices] / family_sum
        else:
            weights[indices] = 1.0 / len(indices)

    return weights


# ---------------------------------------------------------------------------
# Gene scoring functions
# ---------------------------------------------------------------------------

def score_genes_cai(
    rscu_mat: np.ndarray,
    weights: np.ndarray,
    codon_cols: list[str],
) -> np.ndarray:
    """Compute a CAI-like score for each gene against a weight table.

    Uses the geometric mean of codon weights (the standard CAI formulation).
    Codons with weight 0 get a small floor (1e-4) to avoid log(0).

    Returns:
        scores: (n_genes,) array in [0, 1]
    """
    w = np.maximum(weights, 1e-4)
    log_w = np.log(w)

    rscu_clean = np.nan_to_num(rscu_mat, nan=0.0)
    numerator = (rscu_clean * log_w[np.newaxis, :]).sum(axis=1)
    denominator = rscu_clean.sum(axis=1)
    denominator = np.where(denominator > 0, denominator, 1.0)

    log_cai = numerator / denominator
    return np.exp(log_cai)


def score_genes_cosine(
    rscu_mat: np.ndarray,
    weights: np.ndarray,
    codon_cols: list[str],
) -> np.ndarray:
    """Score genes by cosine similarity to the weight table.

    Cosine similarity captures directional agreement independent of magnitude,
    making it robust to GC-content effects.

    Returns:
        scores: (n_genes,) array in [-1, 1], higher = more similar
    """
    w_norm = np.sqrt(np.sum(weights ** 2))
    if w_norm < 1e-9:
        return np.zeros(rscu_mat.shape[0])
    w_unit = weights / w_norm

    rscu_clean = np.nan_to_num(rscu_mat, nan=0.0)
    row_norms = np.sqrt(np.sum(rscu_clean ** 2, axis=1))
    row_norms = np.where(row_norms > 1e-9, row_norms, 1.0)
    rscu_unit = rscu_clean / row_norms[:, np.newaxis]

    return rscu_unit @ w_unit


def score_genes_tai(
    rscu_mat: np.ndarray,
    codon_cols: list[str],
    trna_weights: dict[str, float],
) -> np.ndarray:
    """Score genes by tRNA Adaptation Index.

    Uses tRNA gene copy numbers to weight each codon, then computes the
    geometric mean across codons in each gene.

    Args:
        trna_weights: {codon(RNA): tRNA_copy_number} mapping

    Returns:
        scores: (n_genes,) array
    """
    w = np.zeros(len(codon_cols))
    for i, col in enumerate(codon_cols):
        _, codon = _parse_codon_col(col)
        rna_codon = codon.replace("T", "U")
        w[i] = trna_weights.get(rna_codon, 0.0)

    aa_groups = _group_codons_by_aa(codon_cols)
    for aa, indices in aa_groups.items():
        fam_max = w[indices].max()
        if fam_max > 0:
            w[indices] = w[indices] / fam_max
        else:
            w[indices] = 1.0 / len(indices)

    return score_genes_cai(rscu_mat, w, codon_cols)


# ---------------------------------------------------------------------------
# HGT pre-filter (LedoitWolf regularized covariance)
# ---------------------------------------------------------------------------

def _compute_mahalanobis_mask(
    rscu_mat: np.ndarray,
    percentile: float = _MAHA_PERCENTILE,
) -> np.ndarray:
    """Return boolean mask of genes to KEEP (below Mahalanobis cutoff).

    Uses LedoitWolf shrinkage for robust covariance estimation, consistent
    with the HGT detection in bio_ecology.py.
    """
    from scipy.stats import chi2

    clean = np.nan_to_num(rscu_mat, nan=0.0)
    col_std = clean.std(axis=0)
    keep_cols = col_std > 1e-9
    X = clean[:, keep_cols]

    if X.shape[0] < X.shape[1] + 2:
        logger.warning(
            "Too few genes (%d) for Mahalanobis filter (%d features); "
            "skipping HGT pre-filter",
            X.shape[0], X.shape[1],
        )
        return np.ones(rscu_mat.shape[0], dtype=bool)

    try:
        lw = LedoitWolf().fit(X)
        cov_inv = np.linalg.inv(lw.covariance_)
        mean = lw.location_
    except Exception as e:
        logger.warning("LedoitWolf covariance failed (%s); skipping HGT filter", e)
        return np.ones(rscu_mat.shape[0], dtype=bool)

    diff = X - mean
    maha_sq = np.sum(diff @ cov_inv * diff, axis=1)
    maha_sq = np.maximum(maha_sq, 0.0)

    d = X.shape[1]
    threshold = chi2.ppf(percentile, df=d)

    mask = maha_sq <= threshold
    n_removed = (~mask).sum()
    logger.info(
        "ACE HGT pre-filter: removed %d/%d genes (%.1f%%)",
        n_removed, len(mask), 100 * n_removed / len(mask),
    )
    return mask


# ---------------------------------------------------------------------------
# Iterative convergence
# ---------------------------------------------------------------------------

def _iterate_to_convergence(
    rscu_mat: np.ndarray,
    gene_lengths: np.ndarray,
    gene_ids: list[str],
    codon_cols: list[str],
    initial_ranking: np.ndarray,
    candidate_mask: np.ndarray,
    top_pct: float = 5.0,
    max_iter: int = _MAX_ITER,
    jaccard_threshold: float = _JACCARD_THRESHOLD,
    seed_label: str = "unknown",
) -> dict:
    """Run the iterative ACE convergence procedure for one seed.

    The convergence loop always uses cosine similarity in RSCU space as
    the scoring function.  Cosine normalizes out magnitude differences
    driven by base composition, so it doesn't conflate "GC-rich" with
    "translationally optimized" the way CAI does.  Benchmarking confirmed
    this: in high-GC organisms CAI-scored loops converge on the wrong
    gene set, while cosine-scored loops converge stably across all three
    seeds (cross-seed cosine > 0.999).

    Args:
        rscu_mat:        (n_genes, n_codons) array
        gene_lengths:    (n_genes,) array
        gene_ids:        gene identifier list
        codon_cols:      codon column names
        initial_ranking: (n_genes,) array (higher = more biased/selected)
        candidate_mask:  boolean mask of eligible genes (post filters)
        top_pct:         percentage of genes to select each iteration
        max_iter:        maximum iterations
        jaccard_threshold: convergence criterion
        seed_label:      descriptive name for logging

    Returns:
        dict with: weights, gene_mask, gene_set, n_iter, history, converged,
        seed_label
    """
    n_genes = rscu_mat.shape[0]
    n_select = max(10, int(round(candidate_mask.sum() * top_pct / 100)))

    masked_ranking = np.where(candidate_mask, initial_ranking, -np.inf)
    top_indices = np.argsort(masked_ranking)[::-1][:n_select]
    current_set = set(top_indices)
    current_mask = np.zeros(n_genes, dtype=bool)
    current_mask[list(current_set)] = True

    history = []
    logger.info(
        "  ACE seed '%s': selecting top %d genes (%.1f%% of %d candidates)",
        seed_label, n_select, top_pct, candidate_mask.sum(),
    )

    for iteration in range(max_iter):
        weights = build_weight_table(rscu_mat, gene_lengths, current_mask, codon_cols)
        scores = score_genes_cosine(rscu_mat, weights, codon_cols)
        masked_scores = np.where(candidate_mask, scores, -np.inf)

        new_indices = np.argsort(masked_scores)[::-1][:n_select]
        new_set = set(new_indices)
        new_mask = np.zeros(n_genes, dtype=bool)
        new_mask[list(new_set)] = True

        intersection = len(current_set & new_set)
        union = len(current_set | new_set)
        jaccard = intersection / union if union > 0 else 0.0

        history.append({
            "iteration": iteration + 1,
            "jaccard": jaccard,
            "set_size": len(new_set),
            "n_changed": len(new_set - current_set),
        })

        logger.debug(
            "    ACE iter %d: Jaccard=%.4f, changed=%d/%d",
            iteration + 1, jaccard, len(new_set - current_set), len(new_set),
        )

        current_set = new_set
        current_mask = new_mask

        if jaccard >= jaccard_threshold:
            logger.info(
                "  ACE seed '%s' converged at iteration %d (J=%.4f)",
                seed_label, iteration + 1, jaccard,
            )
            break
    else:
        logger.warning(
            "  ACE seed '%s' did NOT converge after %d iterations (final J=%.4f)",
            seed_label, max_iter, jaccard,
        )

    final_weights = build_weight_table(rscu_mat, gene_lengths, current_mask, codon_cols)
    gene_set = {gene_ids[i] for i in current_set}

    return {
        "weights": final_weights,
        "gene_mask": current_mask,
        "gene_set": gene_set,
        "n_iter": len(history),
        "history": history,
        "converged": len(history) > 0 and history[-1]["jaccard"] >= jaccard_threshold,
        "seed_label": seed_label,
    }


# ---------------------------------------------------------------------------
# Seed builders (accept DataFrames rather than reading files)
# ---------------------------------------------------------------------------

def _seed_enc_residual(
    enc_df: pd.DataFrame | None,
    encprime_df: pd.DataFrame | None,
    gene_ids: list[str],
) -> np.ndarray | None:
    """Rank genes by ENC' residual.  Most negative = most selected."""
    if enc_df is None or encprime_df is None:
        return None

    try:
        from codonpipe.modules.advanced_analyses import compute_enc_diff
        diff_df = compute_enc_diff(enc_df, encprime_df)
    except Exception:
        diff_df = None

    if diff_df is None or diff_df.empty or "ENC_diff" not in diff_df.columns:
        return None

    enc_map = dict(zip(diff_df["gene"].astype(str), diff_df["ENC_diff"]))
    ranking = np.array([-enc_map.get(g, 0.0) for g in gene_ids])

    n_found = sum(1 for g in gene_ids if g in enc_map)
    logger.info("  ACE ENC' residual seed: %d/%d genes with data", n_found, len(gene_ids))
    return ranking


def _seed_trna_adaptation(
    advanced_results: dict,
    rscu_mat: np.ndarray,
    codon_cols: list[str],
) -> np.ndarray | None:
    """Rank genes by tRNA Adaptation Index from advanced analysis results."""
    trna_df = advanced_results.get("trna_counts")
    if trna_df is None:
        return None
    if not isinstance(trna_df, pd.DataFrame):
        return None
    if "codon" not in trna_df.columns or "tRNA_copy_number" not in trna_df.columns:
        return None

    trna_weights = dict(
        zip(trna_df["codon"].astype(str), trna_df["tRNA_copy_number"].astype(float))
    )
    if sum(v > 0 for v in trna_weights.values()) < 10:
        logger.info("  ACE tRNA seed: too few tRNA genes (%d), skipping",
                     sum(v > 0 for v in trna_weights.values()))
        return None

    scores = score_genes_tai(rscu_mat, codon_cols, trna_weights)
    logger.info("  ACE tRNA seed: %d codons with tRNA data",
                sum(v > 0 for v in trna_weights.values()))
    return scores


def _seed_ribosomal_proteins(
    rscu_rp: pd.DataFrame | None,
    gene_ids: list[str],
    rscu_mat: np.ndarray,
    gene_lengths: np.ndarray,
    codon_cols: list[str],
) -> np.ndarray | None:
    """Rank genes by similarity to the RP codon usage profile."""
    if rscu_rp is None:
        return None

    # rscu_rp is the ribosomal protein RSCU (may be a summary row or per-gene)
    # We need RP gene IDs; try to extract them from the DataFrame
    if isinstance(rscu_rp, pd.DataFrame) and "gene" in rscu_rp.columns:
        rp_genes = set(rscu_rp["gene"].astype(str))
    elif isinstance(rscu_rp, dict):
        # It's a summary dict (genome-level RP RSCU); can't identify genes
        return None
    else:
        return None

    rp_mask = np.array([g in rp_genes for g in gene_ids])
    if rp_mask.sum() < 5:
        logger.info("  ACE RP seed: too few RP genes matched (%d), skipping", rp_mask.sum())
        return None

    weights = build_weight_table(rscu_mat, gene_lengths, rp_mask, codon_cols)
    scores = score_genes_cosine(rscu_mat, weights, codon_cols)

    logger.info("  ACE RP seed: %d ribosomal protein genes identified", rp_mask.sum())
    return scores


# ---------------------------------------------------------------------------
# Cross-seed comparison and consensus
# ---------------------------------------------------------------------------

def _compare_weight_tables(results: list[dict]) -> pd.DataFrame | None:
    """Compute pairwise cosine similarity between converged weight tables."""
    if len(results) < 2:
        return None

    n = len(results)
    labels = [r["seed_label"] for r in results]
    sim_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = 1.0 - cosine_dist(results[i]["weights"], results[j]["weights"])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

    return pd.DataFrame(sim_matrix, index=labels, columns=labels)


def _build_consensus(
    results: list[dict],
    codon_cols: list[str],
) -> tuple[np.ndarray, set[str], bool]:
    """Build consensus weight table from converged seeds.

    If all seeds agree (cosine > threshold), returns the mean weight table.
    Otherwise returns the ENC' residual seed table and flags disagreement.
    """
    if len(results) == 0:
        raise ValueError("No converged results to build consensus from")

    if len(results) == 1:
        r = results[0]
        return r["weights"], r["gene_set"], r["converged"]

    sim_df = _compare_weight_tables(results)
    upper = sim_df.values[np.triu_indices(len(results), k=1)]
    min_sim = upper.min()

    logger.info("ACE cross-seed min cosine similarity: %.4f (threshold: %.2f)",
                min_sim, _COSINE_CONVERGENCE)

    consensus_reached = min_sim >= _COSINE_CONVERGENCE

    if consensus_reached:
        mean_weights = np.mean([r["weights"] for r in results], axis=0)
        aa_groups = _group_codons_by_aa(codon_cols)
        for aa, indices in aa_groups.items():
            fam_sum = mean_weights[indices].sum()
            if fam_sum > 0:
                mean_weights[indices] /= fam_sum

        gene_sets = [r["gene_set"] for r in results]
        consensus_genes = gene_sets[0]
        for gs in gene_sets[1:]:
            consensus_genes = consensus_genes & gs

        logger.info("ACE consensus reached: %d genes in intersection of %d seeds",
                     len(consensus_genes), len(results))
        return mean_weights, consensus_genes, True
    else:
        enc_result = next(
            (r for r in results if "ENC" in r["seed_label"]), results[0])
        logger.warning(
            "ACE seeds DIVERGED (min cosine=%.4f < %.2f). "
            "Reporting ENC'-residual table as primary. "
            "This genome may have multi-modal codon usage.",
            min_sim, _COSINE_CONVERGENCE,
        )
        return enc_result["weights"], enc_result["gene_set"], False


# ---------------------------------------------------------------------------
# Diagnostic figure
# ---------------------------------------------------------------------------

def _plot_convergence(
    results: list[dict],
    sim_df: pd.DataFrame | None,
    consensus_weights: np.ndarray,
    codon_cols: list[str],
    sample_id: str,
    out_path: Path,
) -> None:
    """Generate a diagnostic figure for the convergence procedure."""
    n_seeds = len(results)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    colors = ["#E74C3C", "#2980B9", "#27AE60", "#F39C12", "#8E44AD"]

    # (A) Convergence traces
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(results):
        iters = [h["iteration"] for h in r["history"]]
        jaccards = [h["jaccard"] for h in r["history"]]
        ax.plot(iters, jaccards, "o-", color=colors[i % len(colors)],
                label=r["seed_label"], linewidth=2, markersize=4)
    ax.axhline(_JACCARD_THRESHOLD, color="gray", linestyle="--", linewidth=1,
               label=f"Threshold ({_JACCARD_THRESHOLD})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Jaccard similarity")
    ax.set_title("(A) Convergence Traces")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)

    # (B) Genes changed per iteration
    ax = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(results):
        iters = [h["iteration"] for h in r["history"]]
        changed = [h["n_changed"] for h in r["history"]]
        ax.plot(iters, changed, "o-", color=colors[i % len(colors)],
                label=r["seed_label"], linewidth=2, markersize=4)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Genes replaced")
    ax.set_title("(B) Set Stability")
    ax.legend(fontsize=7)

    # (C) Cross-seed similarity heatmap
    ax = fig.add_subplot(gs[0, 2])
    if sim_df is not None and len(sim_df) > 1:
        im = ax.imshow(sim_df.values, cmap="RdYlGn", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(sim_df)))
        ax.set_xticklabels(sim_df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sim_df)))
        ax.set_yticklabels(sim_df.index, fontsize=8)
        for i in range(len(sim_df)):
            for j in range(len(sim_df)):
                ax.text(j, i, f"{sim_df.values[i, j]:.3f}", ha="center",
                        va="center", fontsize=8,
                        color="white" if sim_df.values[i, j] < 0.75 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
        ax.set_title("(C) Cross-Seed Agreement")
    else:
        ax.text(0.5, 0.5, "Single seed\n(no comparison)", ha="center",
                va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("(C) Cross-Seed Agreement")
        ax.axis("off")

    # (D) Converged weight table (top 30 codons)
    ax = fig.add_subplot(gs[1, 0:2])
    sorted_idx = np.argsort(consensus_weights)[::-1][:30]
    labels = [codon_cols[i].split("-")[-1] for i in sorted_idx]
    aa_labels = [codon_cols[i].split("-")[0] for i in sorted_idx]
    bar_labels = [f"{l}\n({a})" for l, a in zip(labels, aa_labels)]
    ax.bar(range(len(sorted_idx)), consensus_weights[sorted_idx],
           color="#2980B9", alpha=0.8)
    ax.set_xticks(range(len(sorted_idx)))
    ax.set_xticklabels(bar_labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Weight (within AA family)")
    ax.set_title("(D) Converged Codon Preference Table (top 30)")

    # (E) Summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    text_lines = [f"{sample_id} ACE Convergence Summary\n"]
    for r in results:
        status = "converged" if r["converged"] else "NOT converged"
        text_lines.append(
            f"{r['seed_label']}: {len(r['gene_set'])} genes, "
            f"{r['n_iter']} iter, {status}")

    if len(results) > 1:
        all_sets = [r["gene_set"] for r in results]
        intersection = all_sets[0]
        union_set = all_sets[0]
        for gs_ in all_sets[1:]:
            intersection = intersection & gs_
            union_set = union_set | gs_
        text_lines.append(f"\nIntersection: {len(intersection)} genes")
        text_lines.append(f"Union: {len(union_set)} genes")
        if len(union_set) > 0:
            text_lines.append(f"Overlap: {len(intersection)/len(union_set)*100:.1f}%")

    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))
    ax.set_title("(E) Summary", fontsize=10)

    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ACE convergence figure: %s", out_path.with_suffix(".png"))


# ---------------------------------------------------------------------------
# ACE-specific codon table outputs
# ---------------------------------------------------------------------------

def _build_ace_codon_tables(
    consensus_weights: np.ndarray,
    codon_cols: list[str],
    rscu_mat: np.ndarray,
    gene_lengths: np.ndarray,
    core_mask: np.ndarray,
    gene_ids: list[str],
) -> dict[str, pd.DataFrame]:
    """Build ACE-specific codon usage tables from the converged weight table.

    Returns a dict of DataFrames keyed by table type:
        ace_weights:    Per-codon weight within each AA family (the converged table)
        ace_w_values:   Relative adaptiveness (w = weight / max_weight per family)
        ace_adaptation: Log-ratio weights (ACE core vs genome background)
        ace_optimal:    Optimal codon set derived from ACE convergence
        ace_scores:     Per-gene scores: ace_melp (cosine, primary), ace_cai
                        (secondary), and ace_expression_class derived from MELP
    """
    tables: dict[str, pd.DataFrame] = {}
    aa_groups = _group_codons_by_aa(codon_cols)

    # 1. ACE weight table
    rows = []
    for i, col in enumerate(codon_cols):
        aa, codon = _parse_codon_col(col)
        rows.append({
            "codon": codon,
            "amino_acid_family": aa,
            "ace_weight": round(consensus_weights[i], 6),
            "column_name": col,
        })
    tables["ace_weights"] = pd.DataFrame(rows)

    # 2. ACE relative adaptiveness (w values): weight / max_weight per family
    w_rows = []
    for i, col in enumerate(codon_cols):
        aa, codon = _parse_codon_col(col)
        family_indices = aa_groups[aa]
        max_w = max(consensus_weights[j] for j in family_indices)
        w_val = consensus_weights[i] / max_w if max_w > 0 else np.nan
        w_rows.append({
            "codon": codon,
            "amino_acid_family": aa,
            "ace_weight": round(consensus_weights[i], 6),
            "ace_w_value": round(w_val, 6) if not np.isnan(w_val) else np.nan,
        })
    tables["ace_w_values"] = pd.DataFrame(w_rows)

    # 3. ACE adaptation weights: ln(RSCU_core / RSCU_genome)
    #    Core set RSCU = length-weighted mean from converged genes
    #    Genome RSCU = length-weighted mean from all genes
    all_mask = np.ones(rscu_mat.shape[0], dtype=bool)
    core_weights_vec = build_weight_table(rscu_mat, gene_lengths, core_mask, codon_cols)
    genome_weights_vec = build_weight_table(rscu_mat, gene_lengths, all_mask, codon_cols)

    adapt_rows = []
    for i, col in enumerate(codon_cols):
        aa, codon = _parse_codon_col(col)
        pseudocount = 1e-4
        log_ratio = np.log(
            (core_weights_vec[i] + pseudocount) / (genome_weights_vec[i] + pseudocount)
        )
        is_optimal = log_ratio > 0
        adapt_rows.append({
            "codon": codon,
            "amino_acid_family": aa,
            "ace_core_weight": round(core_weights_vec[i], 6),
            "genome_weight": round(genome_weights_vec[i], 6),
            "ace_adaptation_weight": round(log_ratio, 6),
            "ace_optimal": is_optimal,
        })
    tables["ace_adaptation"] = pd.DataFrame(adapt_rows)

    # 4. Optimal codon set
    adapt_df = tables["ace_adaptation"]
    optimal_df = adapt_df[adapt_df["ace_optimal"]].copy()
    optimal_df = optimal_df.sort_values("ace_adaptation_weight", ascending=False)
    # Pick top codon per AA family
    optimal_per_aa = optimal_df.drop_duplicates(subset=["amino_acid_family"])
    tables["ace_optimal"] = optimal_per_aa[
        ["amino_acid_family", "codon", "ace_adaptation_weight"]
    ].reset_index(drop=True)

    # 5. Per-gene scores against ACE weight table
    #
    # ace_melp (cosine similarity) is the primary expression metric.
    # Cosine in RSCU space measures directional agreement with the
    # converged reference, independent of magnitude differences driven
    # by base composition.  This avoids CAI's known failure mode in
    # high-GC organisms where dynamic range compresses.
    #
    # ace_cai is retained as a secondary output for literature
    # comparability, but expression_class derives from ace_melp.
    cai_scores = score_genes_cai(rscu_mat, consensus_weights, codon_cols)
    melp_scores = score_genes_cosine(rscu_mat, consensus_weights, codon_cols)

    # Classify expression tiers from MELP (top/bottom 10%)
    melp_series = pd.Series(melp_scores)
    valid = melp_series.dropna()
    if len(valid) > 0:
        hi_thresh = np.nanpercentile(valid, 90)
        lo_thresh = np.nanpercentile(valid, 10)
    else:
        hi_thresh = lo_thresh = 0.0

    core_gene_set = set() if core_mask is None else {
        gene_ids[j] for j in np.where(core_mask)[0]
    }

    score_rows = []
    for idx, gid in enumerate(gene_ids):
        melp_val = float(melp_scores[idx])
        if np.isnan(melp_val):
            expr_class = "unknown"
        elif melp_val >= hi_thresh:
            expr_class = "high"
        elif melp_val <= lo_thresh:
            expr_class = "low"
        else:
            expr_class = "medium"

        score_rows.append({
            "gene": gid,
            "ace_melp": round(melp_val, 6),
            "ace_cai": round(float(cai_scores[idx]), 6),
            "ace_expression_class": expr_class,
            "in_ace_core": gid in core_gene_set,
        })
    tables["ace_scores"] = pd.DataFrame(score_rows)

    return tables


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_outputs(
    consensus_weights: np.ndarray,
    core_genes: set[str],
    results: list[dict],
    sim_df: pd.DataFrame | None,
    consensus_reached: bool,
    codon_cols: list[str],
    gene_ids: list[str],
    ace_tables: dict[str, pd.DataFrame],
    rscu_mat: np.ndarray,
    gene_lengths: np.ndarray,
    core_mask: np.ndarray,
    sample_id: str,
    output_dir: Path,
    expr_df: pd.DataFrame | None,
) -> dict[str, Path]:
    """Write all ACE outputs and return path dict."""
    ace_dir = output_dir / "ace_convergence"
    ace_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Converged weight table
    wt_path = ace_dir / f"{sample_id}_ace_converged_table.tsv"
    ace_tables["ace_weights"].to_csv(wt_path, sep="\t", index=False, float_format="%.6f")
    outputs["ace_converged_table"] = wt_path

    # Core gene list (with expression annotations if available)
    core_path = ace_dir / f"{sample_id}_ace_core_genes.tsv"
    core_rows = []
    expr_map = {}
    if expr_df is not None and not expr_df.empty:
        gene_col = next(
            (c for c in ("gene", "gene_id", "gene_name") if c in expr_df.columns),
            expr_df.columns[0],
        )
        for _, row in expr_df.iterrows():
            expr_map[str(row[gene_col])] = {
                "CAI": row.get("CAI", np.nan),
                "MELP": row.get("MELP", np.nan),
                "Fop": row.get("Fop", np.nan),
                "expression_class": row.get("expression_class", ""),
            }
    for g in gene_ids:
        if g in core_genes:
            row_data: dict = {"gene": g, "in_ace_core": True}
            if g in expr_map:
                row_data.update(expr_map[g])
            core_rows.append(row_data)
    core_df = pd.DataFrame(core_rows)
    core_df.to_csv(core_path, sep="\t", index=False, float_format="%.4f")
    outputs["ace_core_genes"] = core_path

    # Convergence report
    report_path = ace_dir / f"{sample_id}_ace_convergence_report.tsv"
    report_rows = []
    for r in results:
        report_rows.append({
            "seed": r["seed_label"],
            "converged": r["converged"],
            "n_iterations": r["n_iter"],
            "final_jaccard": r["history"][-1]["jaccard"] if r["history"] else 0,
            "n_core_genes": len(r["gene_set"]),
        })
    report_df = pd.DataFrame(report_rows)

    with open(report_path, "w") as f:
        f.write(f"# ACE Convergence Report\n")
        f.write(f"# Consensus reached: {consensus_reached}\n")
        if sim_df is not None:
            f.write(f"# Cross-seed cosine similarities:\n")
            for line in sim_df.to_string().split("\n"):
                f.write(f"#   {line}\n")
        f.write("#\n")
        report_df.to_csv(f, sep="\t", index=False)
    outputs["ace_convergence_report"] = report_path

    # ACE-specific codon tables
    for table_name, table_df in ace_tables.items():
        if table_name == "ace_weights":
            continue  # already written above
        tbl_path = ace_dir / f"{sample_id}_{table_name}.tsv"
        table_df.to_csv(tbl_path, sep="\t", index=False, float_format="%.6f")
        outputs[table_name] = tbl_path

    # Diagnostic figure
    fig_path = ace_dir / f"{sample_id}_ace_convergence"
    _plot_convergence(results, sim_df, consensus_weights, codon_cols,
                      sample_id, fig_path)
    outputs["ace_convergence_png"] = fig_path.with_suffix(".png")
    outputs["ace_convergence_svg"] = fig_path.with_suffix(".svg")

    return outputs


# ---------------------------------------------------------------------------
# Public API — called by pipeline.py
# ---------------------------------------------------------------------------

def run_ace_convergence(
    rscu_gene_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    enc_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    rscu_rp: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    advanced_results: dict | None = None,
    top_pct: float = 5.0,
    min_gene_length: int = _DEFAULT_MIN_GENE_LENGTH,
) -> dict:
    """Run ACE convergence and produce all outputs.

    This is the main entry point called by the pipeline orchestrator.
    Accepts DataFrames that are already in memory rather than re-reading
    files from disk.

    The convergence loop uses cosine similarity exclusively.  CAI is
    computed as a secondary output for literature comparability, but the
    primary expression metric (``ace_melp``) and the derived
    ``ace_expression_class`` come from cosine similarity against the
    converged reference.

    Args:
        rscu_gene_df: Per-gene RSCU matrix (from compute_rscu_per_gene).
        output_dir:   Sample output directory.
        sample_id:    Sample identifier.
        enc_df:       ENC + GC3 per gene (from compute_enc).
        encprime_df:  ENCprime per gene (from run_cu_statistics).
        rscu_rp:      Ribosomal protein per-gene RSCU DataFrame (optional).
        expr_df:      Expression predictions (optional, used for annotation).
        advanced_results: Dict of DataFrames from run_advanced_analyses
            (used for tRNA counts seed).
        top_pct:      Percentage of genes to select each iteration.
        min_gene_length: Minimum gene length filter (nucleotides).

    Returns:
        Dict with keys:
            - File paths (str -> Path) for all outputs
            - 'ace_weights_array': the raw (n_codons,) consensus weight array
            - 'ace_core_gene_set': set of gene IDs in the converged core
            - 'ace_consensus_reached': bool
            - 'ace_codon_cols': list of codon column names
            - 'ace_tables': dict of ACE-specific codon table DataFrames
            - 'ace_scores_df': DataFrame with per-gene ACE MELP/CAI scores
              and ace_expression_class
    """
    logger.info("Running ACE iterative convergence (top_pct=%.1f%%, scoring=cosine)",
                top_pct)

    if rscu_gene_df is None or rscu_gene_df.empty:
        logger.warning("ACE: no RSCU data available, skipping")
        return {}

    # Extract numeric matrix
    rscu_mat, gene_lengths, gene_ids, codon_cols = _extract_rscu_matrix(
        rscu_gene_df, min_gene_length,
    )

    if len(gene_ids) < 50:
        logger.warning("ACE: too few genes (%d) for meaningful convergence, skipping",
                        len(gene_ids))
        return {}

    # Pre-filter: Mahalanobis HGT exclusion
    maha_mask = _compute_mahalanobis_mask(rscu_mat)
    candidate_mask = maha_mask  # length filter already applied in _extract_rscu_matrix

    logger.info("ACE candidate pool: %d genes (%.1f%% of total after filters)",
                candidate_mask.sum(), 100 * candidate_mask.sum() / len(gene_ids))

    # Build seeds
    seeds: list[tuple[str, np.ndarray]] = []

    enc_ranking = _seed_enc_residual(enc_df, encprime_df, gene_ids)
    if enc_ranking is not None:
        seeds.append(("ENC' residual", enc_ranking))

    if advanced_results is not None:
        tai_ranking = _seed_trna_adaptation(advanced_results, rscu_mat, codon_cols)
        if tai_ranking is not None:
            seeds.append(("tRNA adaptation", tai_ranking))

    rp_ranking = _seed_ribosomal_proteins(
        rscu_rp, gene_ids, rscu_mat, gene_lengths, codon_cols,
    )
    if rp_ranking is not None:
        seeds.append(("Ribosomal proteins", rp_ranking))

    if not seeds:
        logger.warning("ACE: no valid seeds could be constructed, skipping")
        return {}

    logger.info("ACE: running %d seed(s): %s",
                len(seeds), ", ".join(s[0] for s in seeds))

    # Run convergence for each seed
    results = []
    for label, ranking in seeds:
        result = _iterate_to_convergence(
            rscu_mat=rscu_mat,
            gene_lengths=gene_lengths,
            gene_ids=gene_ids,
            codon_cols=codon_cols,
            initial_ranking=ranking,
            candidate_mask=candidate_mask,
            top_pct=top_pct,
            seed_label=label,
        )
        results.append(result)

    # Cross-seed comparison
    sim_df = _compare_weight_tables(results)

    # Build consensus
    consensus_weights, core_genes, consensus_reached = _build_consensus(
        results, codon_cols,
    )

    # Build core gene mask for table generation
    core_mask = np.array([g in core_genes for g in gene_ids])

    # Build ACE-specific codon tables
    ace_tables = _build_ace_codon_tables(
        consensus_weights, codon_cols, rscu_mat, gene_lengths,
        core_mask, gene_ids,
    )

    # Write all outputs
    file_outputs = _write_outputs(
        consensus_weights=consensus_weights,
        core_genes=core_genes,
        results=results,
        sim_df=sim_df,
        consensus_reached=consensus_reached,
        codon_cols=codon_cols,
        gene_ids=gene_ids,
        ace_tables=ace_tables,
        rscu_mat=rscu_mat,
        gene_lengths=gene_lengths,
        core_mask=core_mask,
        sample_id=sample_id,
        output_dir=output_dir,
        expr_df=expr_df,
    )

    logger.info(
        "ACE complete. Consensus: %s | Core genes: %d | Seeds: %d | Outputs: %d",
        "YES" if consensus_reached else "NO (divergent)",
        len(core_genes), len(results), len(file_outputs),
    )

    # Return both file paths and in-memory objects for downstream use
    return {
        **file_outputs,
        "ace_weights_array": consensus_weights,
        "ace_core_gene_set": core_genes,
        "ace_consensus_reached": consensus_reached,
        "ace_codon_cols": codon_cols,
        "ace_tables": ace_tables,
        "ace_scores_df": ace_tables.get("ace_scores"),
    }
