"""Gene-set vs genome codon-usage comparison.

Takes a list of genes-of-interest (GOI), specified as Prokka locus tags, and
positions them within the existing per-genome distributions produced by the
core CodonPipe pipeline. Three biological questions get answered with the
same data:

1. Are these genes translationally optimized? Mann-Whitney + Cliff's delta
   on each scalar metric (CAI, MELP, Fop, ENC, ENCprime, MILC, GC3,
   mahalanobis_dist), tested against rest-of-genome.

2. Are these genes horizontally acquired? Position GOI in the existing 38-d
   independent-RSCU Mahalanobis space (the bio_ecology HGT detector's
   output), report each gene's distance, FDR flag, and gc3_outlier flag,
   plus a one-sided permutation test on the GOI's HGT-flag rate vs
   length-matched control sets.

3. Do these genes share a codon-usage signature distinct from references?
   Concatenated GOI mean-RSCU vector compared to genome / RP / Mahal-cluster
   reference vectors via Aitchison distance (CLR + Euclidean), with a
   length-matched permutation null. Per-codon Mann-Whitney with global BH
   FDR identifies which codons drive the signal.

Outputs (all written to *output_dir*):
    goi_summary.tsv             — per-GOI metrics + percentile ranks + flags
    goi_distribution_tests.tsv  — per (metric, reference) Mann-Whitney
    goi_rscu_comparison.tsv     — per (codon, reference) Mann-Whitney
    goi_aitchison_perm.tsv      — Aitchison distance + permutation null
    goi_panel.png / .svg        — six-panel summary figure

The module is self-contained: analyze_gene_set takes already-loaded DataFrames
so it can be called from a notebook or test, and load_and_analyze provides a
thin wrapper that reads the standard per-sample output directory layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    AMINO_ACID_FAMILIES,
    COL_GENE,
    RSCU_COL_TO_CODON,
    RSCU_COLUMN_NAMES,
)
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = (#(x>y) - #(x<y)) / (n_x * n_y)."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Vectorised pairwise comparison; O(nx*ny) memory but typical n_x<=200,
    # n_y<=10000 so peak ~2M floats which is fine.
    diff = x[:, None] - y[None, :]
    more = (diff > 0).sum()
    less = (diff < 0).sum()
    return (more - less) / (nx * ny)


def _bootstrap_cliffs_delta_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Cliff's delta. Resamples x and y separately."""
    if rng is None:
        rng = np.random.default_rng(0)
    nx, ny = len(x), len(y)
    if nx < 3 or ny < 3:
        return (float("nan"), float("nan"))
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        xb = rng.choice(x, size=nx, replace=True)
        yb = rng.choice(y, size=ny, replace=True)
        deltas[i] = _cliffs_delta(xb, yb)
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def _effect_label(abs_delta: float) -> str:
    """Romano et al. 2006 thresholds for Cliff's delta magnitude."""
    if abs_delta < 0.147:
        return "negligible"
    if abs_delta < 0.33:
        return "small"
    if abs_delta < 0.474:
        return "medium"
    return "large"


def clr_transform(rscu_vec: np.ndarray, pseudocount: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform on a single RSCU vector.

    Per-AA-family sum constraints make raw Euclidean distance on RSCU
    sensitive to total magnitude. CLR + Euclidean gives Aitchison distance
    (Aitchison 1986), which is the standard for compositional comparison.
    """
    a = np.where(np.isnan(rscu_vec) | (rscu_vec <= 0), pseudocount, rscu_vec)
    log_a = np.log(a)
    return log_a - log_a.mean()


def aitchison_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Aitchison distance = Euclidean of CLR-transformed vectors."""
    return float(np.linalg.norm(clr_transform(a) - clr_transform(b)))


# Backward-compat aliases for internal callers / tests
_clr_transform = clr_transform
_aitchison_distance = aitchison_distance


def _length_quartile_bins(lengths: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Right-edges of length quartile bins for digitize()."""
    edges = np.percentile(lengths, np.linspace(0, 100, n_bins + 1))
    # Use interior edges only; np.digitize maps below edges[1:-1] → bin index 0..n_bins-1
    return edges[1:-1]


def _length_matched_indices(
    bg_lengths: np.ndarray,
    goi_lengths: np.ndarray,
    rng: np.random.Generator,
    n_bins: int = 4,
) -> np.ndarray:
    """Sample indices into bg_lengths matching goi_lengths' length distribution.

    Uses quartile bins computed from the *background* length distribution.
    Per-bin counts in GOI are reproduced in the sample. If a background bin
    has fewer genes than the GOI requires (rare), samples with replacement
    from that bin and logs a warning.
    """
    bg_lengths = np.asarray(bg_lengths)
    goi_lengths = np.asarray(goi_lengths)
    edges = _length_quartile_bins(bg_lengths, n_bins)
    bg_bins = np.digitize(bg_lengths, edges)
    goi_bins = np.digitize(goi_lengths, edges)

    sampled: list[int] = []
    for b in range(n_bins):
        n_needed = int((goi_bins == b).sum())
        if n_needed == 0:
            continue
        pool = np.where(bg_bins == b)[0]
        if len(pool) >= n_needed:
            sampled.extend(rng.choice(pool, size=n_needed, replace=False).tolist())
        else:
            # Fall back to replacement; tag elsewhere if this matters in your
            # context. Typical background size dwarfs GOI so this is rare.
            sampled.extend(rng.choice(pool, size=n_needed, replace=True).tolist())
    return np.asarray(sampled, dtype=int)


def _drop_redundant_codon_per_family(rscu_cols: list[str]) -> list[str]:
    """Drop one codon per amino-acid family for compositional analyses.

    Mirrors bio_ecology._independent_rscu_columns: per-family RSCU sums are
    constrained, so without dropping one column per family the covariance
    is rank-deficient. We use the same deterministic rule (drop the
    lexicographically last codon in each family) so HGT and gene-set
    analyses speak the same coordinate system.
    """
    cols_set = set(rscu_cols)
    drop: set[str] = set()
    for aa_cols in AMINO_ACID_FAMILIES.values():
        family_cols = [c for c in aa_cols if c in cols_set]
        # Family-by-family in the AMINO_ACID_FAMILIES sense pools subfamily
        # codons (e.g. all 6 Leu codons); split into Leu4 vs Leu2 by prefix
        # before picking the column to drop, so the constraint structure
        # matches AA_CODON_GROUPS_RSCU.
        by_prefix: dict[str, list[str]] = {}
        for c in family_cols:
            by_prefix.setdefault(c.split("-")[0], []).append(c)
        for sub_cols in by_prefix.values():
            if len(sub_cols) >= 2:
                drop.add(sorted(sub_cols)[-1])
    return [c for c in rscu_cols if c not in drop]


# ──────────────────────────────────────────────────────────────────────────────
# Public API: per-GOI summary table
# ──────────────────────────────────────────────────────────────────────────────


def _percentile_rank(value: float, dist: np.ndarray) -> float:
    """Empirical percentile rank of *value* within *dist* (0..100, NaN-safe)."""
    if np.isnan(value):
        return float("nan")
    finite = dist[np.isfinite(dist)]
    if len(finite) == 0:
        return float("nan")
    return float((finite < value).sum() + 0.5 * (finite == value).sum()) / len(finite) * 100


def _build_summary_table(
    goi_ids: set[str],
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None,
    encprime_df: pd.DataFrame | None,
    milc_df: pd.DataFrame | None,
    hgt_df: pd.DataFrame | None,
    mahal_cluster_df: pd.DataFrame | None = None,
    cbi_rp_df: pd.DataFrame | None = None,
    cbi_mahal_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-GOI table with all available metrics + percentile ranks + flags.

    Background for percentile rank is the *rest of the genome* (genes not in
    *goi_ids*), so a GOI gene's rank reports its position relative to the
    other genome genes — never relative to itself.

    Mahal-cluster columns (when *mahal_cluster_df* is provided) are kept
    separate from the HGT detector's columns. HGT mahalanobis_dist is from
    the *genome centroid*; mahal_cluster_distance is from the *optimized
    cluster centroid*. They answer different questions: HGT distance is
    "how unusual is this gene's codon usage relative to the bulk genome?"
    while mahal_cluster_distance is "how far is this gene from the
    translationally optimized core?".
    """
    # Anchor on rscu_gene_df: it has the canonical gene list and lengths.
    base = rscu_gene_df[[COL_GENE, "length"]].copy()
    if enc_df is not None and not enc_df.empty:
        enc_cols = [c for c in (COL_GENE, "ENC", "GC3") if c in enc_df.columns]
        base = base.merge(enc_df[enc_cols], on=COL_GENE, how="left")
    for df, score_cols in (
        (expr_df, ["MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop"]),
        (encprime_df, ["ENCprime"]),
        (milc_df, ["MILC"]),
        (hgt_df, ["mahalanobis_dist", "gc3_deviation", "p_adjusted",
                  "hgt_flag_fdr", "hgt_flag_adaptive", "gc3_outlier",
                  "hgt_flag_combined"]),
        (mahal_cluster_df, ["mahal_cluster_distance", "membership_score",
                            "in_optimized_set", "is_ribosomal_protein"]),
        (cbi_rp_df, ["cbi_rp"]),
        (cbi_mahal_df, ["cbi_mahal"]),
    ):
        if df is None or df.empty:
            continue
        cols = [COL_GENE] + [c for c in score_cols if c in df.columns]
        if len(cols) > 1:
            base = base.merge(df[cols], on=COL_GENE, how="left")

    base["in_goi"] = base[COL_GENE].isin(goi_ids)

    # Filter to GOI rows but compute percentile ranks against rest-of-genome
    rest = base[~base["in_goi"]]
    metric_cols = [c for c in (
        "length", "ENC", "GC3", "ENCprime", "MILC",
        "MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop",
        "mahalanobis_dist", "gc3_deviation",
        "mahal_cluster_distance", "membership_score",
        "cbi_rp", "cbi_mahal",
    ) if c in base.columns]

    goi_rows = base[base["in_goi"]].copy()
    for col in metric_cols:
        bg_dist = rest[col].dropna().values
        goi_rows[f"{col}_pctile"] = goi_rows[col].apply(
            lambda v, d=bg_dist: _percentile_rank(v, d) if not np.isnan(v) else np.nan
        )
    return goi_rows.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Scalar-metric distribution tests
# ──────────────────────────────────────────────────────────────────────────────


def _scalar_metric_tests(
    goi_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    metrics: list[str],
    rng: np.random.Generator,
    n_boot_ci: int = 500,
) -> pd.DataFrame:
    """Mann-Whitney U + KS + Cliff's delta with bootstrap CI per metric.

    BH FDR is applied across the whole result table after this returns;
    don't apply it here, callers may want to combine with other tests.
    """
    rows = []
    for m in metrics:
        if m not in goi_df.columns or m not in bg_df.columns:
            continue
        x = goi_df[m].dropna().values
        y = bg_df[m].dropna().values
        if len(x) < 5 or len(y) < 5:
            # Match the project-wide n>=5 threshold (audit fix #6)
            continue
        u_stat, u_p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        ks_d, ks_p = sp_stats.ks_2samp(x, y)
        delta = _cliffs_delta(x, y)
        delta_lo, delta_hi = _bootstrap_cliffs_delta_ci(x, y, n_boot=n_boot_ci, rng=rng)
        rows.append({
            "metric": m,
            "n_goi": len(x),
            "n_background": len(y),
            "median_goi": float(np.median(x)),
            "median_background": float(np.median(y)),
            "U_statistic": round(float(u_stat), 4),
            "p_value": float(u_p),
            "KS_D": round(float(ks_d), 4),
            "KS_p": float(ks_p),
            "cliffs_delta": round(float(delta), 4),
            "delta_ci_low": round(delta_lo, 4),
            "delta_ci_high": round(delta_hi, 4),
            "effect_label": _effect_label(abs(delta)),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-codon RSCU comparison
# ──────────────────────────────────────────────────────────────────────────────


def _codon_tests_against_reference(
    goi_rscu: pd.DataFrame,
    bg_rscu: pd.DataFrame,
    reference_label: str,
    reference_vec: dict[str, float] | None,
    rscu_cols: list[str],
) -> pd.DataFrame:
    """Per-codon Mann-Whitney of GOI vs background RSCU, plus delta vs reference.

    Returns one row per codon. The reference (genome / rp / mahal) is used
    only to compute *delta_to_ref* (mean GOI − reference); the inferential
    test stays GOI vs background since that's the well-defined population
    comparison.
    """
    rows = []
    for col in rscu_cols:
        if col not in goi_rscu.columns or col not in bg_rscu.columns:
            continue
        x = goi_rscu[col].dropna().values
        y = bg_rscu[col].dropna().values
        if len(x) < 5 or len(y) < 5:
            continue
        try:
            u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            continue
        delta = _cliffs_delta(x, y)
        codon = RSCU_COL_TO_CODON.get(col, col.split("-")[-1])
        aa = col.split("-")[0]
        ref_val = reference_vec.get(col, float("nan")) if reference_vec is not None else float("nan")
        rows.append({
            "reference": reference_label,
            "codon_col": col,
            "amino_acid": aa,
            "codon": codon,
            "mean_goi": round(float(np.mean(x)), 4),
            "mean_background": round(float(np.mean(y)), 4),
            "ref_value": round(ref_val, 4) if not np.isnan(ref_val) else np.nan,
            "delta_to_ref": round(float(np.mean(x)) - ref_val, 4) if not np.isnan(ref_val) else np.nan,
            "U_statistic": round(float(u), 4),
            "p_value": float(p),
            "cliffs_delta": round(float(delta), 4),
        })
    return pd.DataFrame(rows)


def _codon_tests_all_references(
    goi_rscu: pd.DataFrame,
    bg_rscu: pd.DataFrame,
    references: dict[str, dict[str, float] | None],
    rscu_cols: list[str],
) -> pd.DataFrame:
    """Run per-codon tests against every available reference, then BH-correct globally."""
    pieces = []
    for label, ref_vec in references.items():
        piece = _codon_tests_against_reference(goi_rscu, bg_rscu, label, ref_vec, rscu_cols)
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    result = pd.concat(pieces, ignore_index=True)
    # Global BH across (codon × reference) — same convention as
    # comparative.between_condition_rscu_tests after audit fix #5.
    result["p_adjusted"] = benjamini_hochberg(result["p_value"].values)
    result["significant"] = result["p_adjusted"] < 0.05
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Aitchison-distance permutation test
# ──────────────────────────────────────────────────────────────────────────────


def _aitchison_perm_test(
    goi_rscu: pd.DataFrame,
    bg_rscu: pd.DataFrame,
    references: dict[str, dict[str, float] | None],
    rscu_cols_indep: list[str],
    rng: np.random.Generator,
    n_perm: int = 999,
    length_matched: bool = True,
) -> pd.DataFrame:
    """Aitchison distance from GOI mean-RSCU to reference, with permutation null.

    For each reference vector, computes:
      * obs_distance: Aitchison distance from GOI's mean RSCU to ref
      * perm_distances: for n_perm random gene sets of the same size as GOI
        (length-matched if length_matched=True), the same distance.
      * p-value: (n_perm where perm >= obs) + 1) / (n_perm + 1)

    Uses the independent RSCU columns (one dropped per AA family) so the
    CLR transform isn't operating on a rank-deficient vector.
    """
    if len(goi_rscu) < 3:
        return pd.DataFrame()

    cols = [c for c in rscu_cols_indep if c in goi_rscu.columns and c in bg_rscu.columns]
    if len(cols) < 5:
        return pd.DataFrame()

    goi_lengths = goi_rscu["length"].values if "length" in goi_rscu.columns else None
    bg_lengths = bg_rscu["length"].values if "length" in bg_rscu.columns else None
    bg_X = bg_rscu[cols].values
    goi_mean = goi_rscu[cols].mean().values

    use_lm = length_matched and goi_lengths is not None and bg_lengths is not None

    rows = []
    for label, ref_vec in references.items():
        if ref_vec is None:
            continue
        ref_arr = np.array([ref_vec.get(c, np.nan) for c in cols])
        if np.any(np.isnan(ref_arr)):
            logger.warning(
                "Aitchison test: reference '%s' missing %d/%d codon values; skipping",
                label, int(np.isnan(ref_arr).sum()), len(ref_arr),
            )
            continue
        obs = float(np.linalg.norm(_clr_transform(goi_mean) - _clr_transform(ref_arr)))

        perm_d = np.empty(n_perm)
        n_goi = len(goi_rscu)
        for i in range(n_perm):
            if use_lm:
                idx = _length_matched_indices(bg_lengths, goi_lengths, rng)
            else:
                idx = rng.choice(len(bg_X), size=n_goi, replace=False)
            sampled_mean = bg_X[idx].mean(axis=0)
            perm_d[i] = float(np.linalg.norm(_clr_transform(sampled_mean) - _clr_transform(ref_arr)))

        # One-sided (right-tail): GOI more divergent from reference than random
        p_more = (np.sum(perm_d >= obs) + 1) / (n_perm + 1)
        # One-sided (left-tail): GOI closer to reference than random (also
        # biologically meaningful — e.g. GOI clustering near the RP centroid
        # is evidence that the set is translationally optimized).
        p_less = (np.sum(perm_d <= obs) + 1) / (n_perm + 1)
        # Two-sided: divergence from the perm distribution in *either* direction.
        p_two_sided = float(min(1.0, 2 * min(p_more, p_less)))
        rows.append({
            "reference": label,
            "n_goi": n_goi,
            "n_codons": len(cols),
            "obs_aitchison": round(obs, 4),
            "perm_mean": round(float(perm_d.mean()), 4),
            "perm_median": round(float(np.median(perm_d)), 4),
            "perm_std": round(float(perm_d.std(ddof=1)), 4),
            "p_value": float(p_two_sided),
            "p_more_divergent": float(p_more),
            "p_more_similar": float(p_less),
            "n_perm": n_perm,
            "null_set": "length_matched" if use_lm else "uniform_random",
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# HGT-flag enrichment (one-sided permutation test)
# ──────────────────────────────────────────────────────────────────────────────


def _hgt_flag_enrichment(
    goi_df: pd.DataFrame,
    bg_df: pd.DataFrame,
    flag_col: str,
    rng: np.random.Generator,
    n_perm: int = 999,
    length_matched: bool = True,
) -> dict | None:
    """One-sided permutation test on the GOI's flag rate vs length-matched controls."""
    if flag_col not in goi_df.columns or flag_col not in bg_df.columns:
        return None

    use_lm = length_matched and "length" in goi_df.columns and "length" in bg_df.columns
    # Drop NaN jointly across (length, flag) so the lengths array stays aligned
    # with the flag array — otherwise length-matched sampling indexes a length
    # vector that has different rows from the flag vector.
    if use_lm:
        goi_clean = goi_df[["length", flag_col]].dropna()
        bg_clean = bg_df[["length", flag_col]].dropna()
        goi_flags = goi_clean[flag_col].values.astype(bool)
        bg_flags = bg_clean[flag_col].values.astype(bool)
        goi_lengths = goi_clean["length"].values
        bg_lengths = bg_clean["length"].values
    else:
        goi_flags = goi_df[flag_col].dropna().values.astype(bool)
        bg_flags = bg_df[flag_col].dropna().values.astype(bool)
        goi_lengths = bg_lengths = None

    if len(goi_flags) < 3 or len(bg_flags) < 30:
        return None
    obs_rate = float(goi_flags.mean())
    bg_rate = float(bg_flags.mean())

    perm_rates = np.empty(n_perm)
    for i in range(n_perm):
        if use_lm:
            idx = _length_matched_indices(bg_lengths, goi_lengths, rng)
        else:
            idx = rng.choice(len(bg_flags), size=len(goi_flags), replace=False)
        perm_rates[i] = bg_flags[idx].mean()
    p_value = (np.sum(perm_rates >= obs_rate) + 1) / (n_perm + 1)
    return {
        "flag": flag_col,
        "n_goi": int(len(goi_flags)),
        "n_background": int(len(bg_flags)),
        "obs_rate_goi": round(obs_rate, 4),
        "rate_background": round(bg_rate, 4),
        "perm_mean_rate": round(float(perm_rates.mean()), 4),
        "p_value_one_sided": float(p_value),
        "n_perm": n_perm,
        "null_set": "length_matched" if use_lm else "uniform_random",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Three-way genome partition: Mahal cluster vs bulk vs outliers
# ──────────────────────────────────────────────────────────────────────────────

# Partition category names (kept here so figure code and tests can reference them)
PARTITION_MAHAL_CLUSTER = "mahal_cluster"
PARTITION_OUTLIER = "outlier"
PARTITION_BULK = "bulk"
PARTITION_UNKNOWN = "unknown"
_PARTITION_CATEGORIES = (PARTITION_MAHAL_CLUSTER, PARTITION_BULK, PARTITION_OUTLIER)


def assign_gene_partition(
    base_df: pd.DataFrame,
    cluster_flag_col: str = "in_optimized_set",
    outlier_flag_col: str = "hgt_flag_combined",
) -> pd.Series:
    """Assign each gene to one of {mahal_cluster, bulk, outlier, unknown}.

    Priority order (each gene falls into exactly one category):
      1. Mahalanobis cluster member  → 'mahal_cluster'
      2. Otherwise, if outlier-flagged → 'outlier'
      3. Otherwise → 'bulk'

    A gene that is neither flagged nor known to be in the cluster (e.g.
    because the HGT detector skipped it for length reasons, or the Mahal
    clustering wasn't run) is assigned to 'bulk' if cluster information is
    available, or 'unknown' if neither flag is present in the input.

    Cluster membership takes precedence over outlier status because the
    Mahalanobis cluster is defined by closeness to the genome's optimized
    centroid (a direct codon-usage readout). HGT flags are a separate
    statistical claim about deviation from the genome bulk — a gene flagged
    as both is by construction inside a tight cluster of similarly-deviant
    genes, which is more naturally read as "translationally adapted but
    located outside the bulk distribution" than as HGT. The priority can be
    inverted at the call site by reordering the inputs.

    Args:
        base_df: DataFrame containing the cluster and/or outlier flag columns.
            Missing columns degrade gracefully; if both are missing every gene
            gets 'unknown'.
        cluster_flag_col: column name for the Mahal-cluster boolean.
        outlier_flag_col: column name for the outlier boolean.
    """
    n = len(base_df)
    has_cluster = cluster_flag_col in base_df.columns
    has_outlier = outlier_flag_col in base_df.columns

    if not has_cluster and not has_outlier:
        return pd.Series([PARTITION_UNKNOWN] * n, index=base_df.index)

    # Coerce to numeric first so .fillna(False).astype(bool) doesn't trigger
    # the pandas object-dtype downcasting FutureWarning on mixed/None inputs.
    if has_cluster:
        cluster = pd.Series(
            base_df[cluster_flag_col].astype("boolean").fillna(False).astype(bool).values,
            index=base_df.index,
        )
    else:
        cluster = pd.Series(False, index=base_df.index)
    if has_outlier:
        outlier = pd.Series(
            base_df[outlier_flag_col].astype("boolean").fillna(False).astype(bool).values,
            index=base_df.index,
        )
    else:
        outlier = pd.Series(False, index=base_df.index)

    out = pd.Series(PARTITION_BULK, index=base_df.index, dtype=object)
    out[outlier & ~cluster] = PARTITION_OUTLIER
    out[cluster] = PARTITION_MAHAL_CLUSTER
    return out


def _partition_enrichment_test(
    partition: pd.Series,
    in_goi: pd.Series,
) -> pd.DataFrame:
    """Cross-tab GOI × partition, then chi-squared + per-category Fisher tests.

    Returns one row per category with:
        category, n_goi, n_background, frac_goi, frac_background,
        odds_ratio, fisher_p, p_adjusted (BH across categories),
        plus an 'OMNIBUS' row reporting the chi-squared test across the
        full 2×3 (GOI × category) contingency table.
    """
    from scipy.stats import chi2_contingency, fisher_exact

    df = pd.DataFrame({"partition": partition, "in_goi": in_goi.astype(bool)})
    df = df.dropna(subset=["partition"])
    df = df[df["partition"] != PARTITION_UNKNOWN]
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    cats = [c for c in _PARTITION_CATEGORIES if (df["partition"] == c).any()]
    n_goi = int(df["in_goi"].sum())
    n_bg = int((~df["in_goi"]).sum())
    if n_goi < 1 or n_bg < 1:
        return pd.DataFrame()

    # Per-category Fisher's exact: 2×2 of (in_category, in_GOI)
    for cat in cats:
        in_cat = df["partition"] == cat
        a = int((in_cat & df["in_goi"]).sum())   # GOI in cat
        b = int((in_cat & ~df["in_goi"]).sum())  # background in cat
        c = int((~in_cat & df["in_goi"]).sum())  # GOI outside cat
        d = int((~in_cat & ~df["in_goi"]).sum()) # background outside cat
        # Two-sided Fisher's exact; use odds ratio with Haldane correction
        try:
            odds, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        except ValueError:
            odds, p = float("nan"), float("nan")
        rows.append({
            "category": cat,
            "n_goi": a,
            "n_background": b,
            "frac_goi": a / n_goi if n_goi else float("nan"),
            "frac_background": b / n_bg if n_bg else float("nan"),
            "odds_ratio": float(odds) if not np.isnan(odds) else float("nan"),
            "fisher_p": float(p) if not np.isnan(p) else float("nan"),
        })

    # BH FDR across the per-category Fisher tests
    if rows:
        adj = benjamini_hochberg(np.array([r["fisher_p"] for r in rows]))
        for i, r in enumerate(rows):
            r["p_adjusted"] = float(adj[i])

    # Omnibus chi-squared across the full 2 × len(cats) table
    omnibus_row = None
    if len(cats) >= 2:
        ct = pd.crosstab(df["partition"], df["in_goi"]).reindex(
            index=cats, columns=[False, True], fill_value=0,
        )
        try:
            chi2, p, dof, _ = chi2_contingency(ct.values)
            n_total = int(ct.values.sum())
            cramers_v = float(np.sqrt(chi2 / (n_total * (min(ct.shape) - 1)))) if n_total else float("nan")
        except Exception:
            chi2, p, dof, cramers_v = float("nan"), float("nan"), float("nan"), float("nan")
        omnibus_row = {
            "category": "OMNIBUS",
            "n_goi": n_goi,
            "n_background": n_bg,
            "frac_goi": 1.0,
            "frac_background": 1.0,
            "odds_ratio": cramers_v,  # Cramer's V repurposes this column
            "fisher_p": float(p),
            "p_adjusted": float(p),  # one-test row; no adjustment beyond itself
        }
    out = pd.DataFrame(rows)
    if omnibus_row is not None:
        out = pd.concat([pd.DataFrame([omnibus_row]), out], ignore_index=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def analyze_gene_set(
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    goi_ids: set[str],
    output_dir: Path,
    sample_id: str,
    *,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    milc_df: pd.DataFrame | None = None,
    hgt_df: pd.DataFrame | None = None,
    mahal_cluster_df: pd.DataFrame | None = None,
    cbi_rp_df: pd.DataFrame | None = None,
    cbi_mahal_df: pd.DataFrame | None = None,
    rscu_genome: dict[str, float] | None = None,
    rscu_rp: dict[str, float] | None = None,
    rscu_mahal_cluster: dict[str, float] | None = None,
    n_permutations: int = 999,
    length_matched: bool = True,
    rng_seed: int = 42,
    make_figure: bool = True,
) -> dict[str, Path]:
    """Run the full GOI-vs-genome comparison and write all output files.

    Args:
        rscu_gene_df: Per-gene RSCU table with COL_GENE, length, and the 59
            RSCU columns. Produced by rscu.compute_rscu_per_gene.
        enc_df: Per-gene ENC + GC3, from rscu.compute_enc.
        goi_ids: Set of Prokka locus tags to treat as the gene set of
            interest. Tags absent from rscu_gene_df are dropped with a warning.
        output_dir: Directory for output files (created if missing).
        sample_id: Used for output filename prefixing.
        expr_df: Combined expression table (MELP/CAI/Fop columns), optional.
        encprime_df: ENCprime per gene with columns gene + ENCprime, optional.
        milc_df: MILC per gene with columns gene + MILC, optional.
        hgt_df: HGT detector output with mahalanobis_dist + flag columns,
            optional. When provided, HGT enrichment tests run. The
            mahalanobis_dist here is from the *genome centroid*.
        mahal_cluster_df: Mahalanobis-cluster per-gene table (gmm_clusters.tsv)
            with mahal_cluster_distance + in_optimized_set + membership_score.
            When provided, the orchestrator adds GOI Mahal-cluster
            membership-rate permutation test and per-gene distances to the
            optimized cluster centroid (a different question from
            hgt_df.mahalanobis_dist).
        cbi_rp_df: Per-gene CBI keyed against ribosomal-protein-derived
            optimal codons. Columns: gene, cbi_rp.
        cbi_mahal_df: Per-gene CBI keyed against Mahal-cluster-derived
            optimal codons. Columns: gene, cbi_mahal.
        rscu_genome: Genome-mean (or median) RSCU as {column_name: value}.
        rscu_rp: Ribosomal-protein concatenated RSCU.
        rscu_mahal_cluster: Mahalanobis-cluster RSCU.
        n_permutations: Permutation count for Aitchison and HGT-flag tests.
        length_matched: Whether to length-match the permutation null.
        rng_seed: NumPy RNG seed for reproducibility.
        make_figure: Whether to render the six-panel summary figure.

    Returns:
        Dict mapping output kind → file path. Always includes 'summary',
        'distribution_tests', 'codon_tests'. May include 'aitchison',
        'hgt_enrichment', 'figure_png', 'figure_svg'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # ── Resolve GOI membership ─────────────────────────────────────────────
    all_genes = set(rscu_gene_df[COL_GENE]) if not rscu_gene_df.empty else set()
    matched = goi_ids & all_genes
    missing = goi_ids - all_genes
    if missing:
        logger.warning(
            "%d/%d GOI locus tags not found in rscu_gene_df (e.g. %s); "
            "dropping them from the analysis.",
            len(missing), len(goi_ids), sorted(missing)[:5],
        )
    if len(matched) < 3:
        raise ValueError(
            f"After filtering to genes present in the RSCU table, only "
            f"{len(matched)} GOI gene(s) remain — need at least 3 for "
            f"meaningful comparisons."
        )

    rscu_gene_df = rscu_gene_df.copy()
    enc_df = enc_df.copy() if enc_df is not None else None
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    rscu_cols_indep = _drop_redundant_codon_per_family(rscu_cols)

    # ── 1. Per-GOI summary table ───────────────────────────────────────────
    summary_df = _build_summary_table(
        matched, rscu_gene_df, enc_df, expr_df, encprime_df, milc_df, hgt_df,
        mahal_cluster_df=mahal_cluster_df,
        cbi_rp_df=cbi_rp_df, cbi_mahal_df=cbi_mahal_df,
    )
    out: dict[str, Path] = {}
    summary_path = output_dir / f"{sample_id}_goi_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    out["summary"] = summary_path
    logger.info("Wrote per-GOI summary: %s (n=%d genes)", summary_path, len(summary_df))

    # ── 2. Scalar-metric distribution tests (GOI vs rest) ─────────────────
    base = rscu_gene_df[[COL_GENE, "length"]].copy()
    if enc_df is not None and not enc_df.empty:
        enc_cols = [c for c in (COL_GENE, "ENC", "GC3") if c in enc_df.columns]
        base = base.merge(enc_df[enc_cols], on=COL_GENE, how="left")
    for df, score_cols in (
        (expr_df, ["MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop"]),
        (encprime_df, ["ENCprime"]),
        (milc_df, ["MILC"]),
        (hgt_df, ["mahalanobis_dist", "gc3_deviation"]),
        (mahal_cluster_df, ["mahal_cluster_distance", "membership_score"]),
        (cbi_rp_df, ["cbi_rp"]),
        (cbi_mahal_df, ["cbi_mahal"]),
    ):
        if df is None or df.empty:
            continue
        cols = [COL_GENE] + [c for c in score_cols if c in df.columns]
        if len(cols) > 1:
            base = base.merge(df[cols], on=COL_GENE, how="left")
    base["in_goi"] = base[COL_GENE].isin(matched)

    metric_candidates = [
        "length", "ENC", "GC3", "ENCprime", "MILC",
        "MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop",
        "mahalanobis_dist", "gc3_deviation",
        "mahal_cluster_distance", "membership_score",
        "cbi_rp", "cbi_mahal",
    ]
    goi_block = base[base["in_goi"]]
    bg_block = base[~base["in_goi"]]
    scalar_tests = _scalar_metric_tests(
        goi_block, bg_block, metric_candidates, rng=rng, n_boot_ci=500,
    )
    if not scalar_tests.empty:
        scalar_tests["p_adjusted"] = benjamini_hochberg(scalar_tests["p_value"].values)
        scalar_tests["significant"] = scalar_tests["p_adjusted"] < 0.05
    dist_path = output_dir / f"{sample_id}_goi_distribution_tests.tsv"
    scalar_tests.to_csv(dist_path, sep="\t", index=False)
    out["distribution_tests"] = dist_path
    logger.info("Wrote scalar distribution tests: %s (n=%d metrics)",
                dist_path, len(scalar_tests))

    # ── 3. Per-codon RSCU comparison vs each available reference ──────────
    references = {
        "genome": rscu_genome,
        "ribosomal_proteins": rscu_rp,
        "mahal_cluster": rscu_mahal_cluster,
    }
    references = {k: v for k, v in references.items() if v is not None}
    goi_rscu = rscu_gene_df[rscu_gene_df[COL_GENE].isin(matched)]
    bg_rscu = rscu_gene_df[~rscu_gene_df[COL_GENE].isin(matched)]
    codon_tests = _codon_tests_all_references(goi_rscu, bg_rscu, references, rscu_cols)
    codon_path = output_dir / f"{sample_id}_goi_rscu_comparison.tsv"
    codon_tests.to_csv(codon_path, sep="\t", index=False)
    out["codon_tests"] = codon_path
    logger.info("Wrote per-codon RSCU comparison: %s (rows=%d)",
                codon_path, len(codon_tests))

    # ── 4. Aitchison distance permutation test ─────────────────────────────
    if references:
        aitch = _aitchison_perm_test(
            goi_rscu, bg_rscu, references, rscu_cols_indep,
            rng=rng, n_perm=n_permutations, length_matched=length_matched,
        )
        if not aitch.empty:
            aitch["p_adjusted"] = benjamini_hochberg(aitch["p_value"].values)
            aitch["significant"] = aitch["p_adjusted"] < 0.05
        aitch_path = output_dir / f"{sample_id}_goi_aitchison_perm.tsv"
        aitch.to_csv(aitch_path, sep="\t", index=False)
        out["aitchison"] = aitch_path
        logger.info("Wrote Aitchison permutation test: %s", aitch_path)

    # ── 5. HGT-flag enrichment (one-sided, length-matched) ────────────────
    if hgt_df is not None and not hgt_df.empty:
        # The flag enrichment needs gene+length+flag columns together; build a
        # dedicated frame so it's not dependent on whether 'base' merged the
        # flag columns (it doesn't, since flags aren't continuous metrics).
        hgt_flag_cols = [c for c in
                         ("hgt_flag_combined", "hgt_flag_fdr", "gc3_outlier")
                         if c in hgt_df.columns]
        if hgt_flag_cols:
            flag_block = rscu_gene_df[[COL_GENE, "length"]].merge(
                hgt_df[[COL_GENE] + hgt_flag_cols], on=COL_GENE, how="left",
            )
            flag_block["in_goi"] = flag_block[COL_GENE].isin(matched)
            goi_flags = flag_block[flag_block["in_goi"]]
            bg_flags = flag_block[~flag_block["in_goi"]]

            hgt_rows = []
            for flag in hgt_flag_cols:
                r = _hgt_flag_enrichment(
                    goi_flags, bg_flags, flag,
                    rng=rng, n_perm=n_permutations, length_matched=length_matched,
                )
                if r is not None:
                    hgt_rows.append(r)
            if hgt_rows:
                hgt_enr = pd.DataFrame(hgt_rows)
                hgt_enr["p_adjusted"] = benjamini_hochberg(hgt_enr["p_value_one_sided"].values)
                hgt_enr["significant"] = hgt_enr["p_adjusted"] < 0.05
                hgt_path = output_dir / f"{sample_id}_goi_hgt_enrichment.tsv"
                hgt_enr.to_csv(hgt_path, sep="\t", index=False)
                out["hgt_enrichment"] = hgt_path
                logger.info("Wrote HGT-flag enrichment: %s", hgt_path)

    # ── 6. Three-way partition: Mahal cluster vs bulk vs outliers ─────────
    # Build a per-gene partition from whichever flag columns are available.
    # base already merged in_optimized_set + the HGT flag columns earlier;
    # gracefully no-op when those flags are absent.
    flag_block = rscu_gene_df[[COL_GENE]].copy()
    if mahal_cluster_df is not None and "in_optimized_set" in mahal_cluster_df.columns:
        flag_block = flag_block.merge(
            mahal_cluster_df[[COL_GENE, "in_optimized_set"]], on=COL_GENE, how="left",
        )
    if hgt_df is not None and "hgt_flag_combined" in hgt_df.columns:
        flag_block = flag_block.merge(
            hgt_df[[COL_GENE, "hgt_flag_combined"]], on=COL_GENE, how="left",
        )
    flag_block["partition"] = assign_gene_partition(flag_block)
    flag_block["in_goi"] = flag_block[COL_GENE].isin(matched)

    partition_table = _partition_enrichment_test(
        flag_block["partition"], flag_block["in_goi"],
    )
    if not partition_table.empty:
        part_path = output_dir / f"{sample_id}_goi_partition.tsv"
        partition_table.to_csv(part_path, sep="\t", index=False)
        out["partition"] = part_path
        # Surface a concise summary in the log so users see the headline result
        omnibus = partition_table[partition_table["category"] == "OMNIBUS"]
        if not omnibus.empty:
            row = omnibus.iloc[0]
            logger.info(
                "GOI partition (Mahal/bulk/outlier): omnibus chi-sq p=%.4g (Cramer's V=%.3f); "
                "see %s for per-category breakdown",
                row["fisher_p"], row["odds_ratio"], part_path,
            )

    # Add the per-gene partition to the per-GOI summary table so users see
    # which category each of their genes belongs to without rejoining files.
    if "partition" not in summary_df.columns:
        summary_df = summary_df.merge(
            flag_block[[COL_GENE, "partition"]], on=COL_GENE, how="left",
        )
        # Re-write summary with the new column
        summary_df.to_csv(summary_path, sep="\t", index=False)

    # ── 7. Mahal-cluster membership two-sided permutation test ────────────
    if mahal_cluster_df is not None and "in_optimized_set" in mahal_cluster_df.columns:
        mahal_block = rscu_gene_df[[COL_GENE, "length"]].merge(
            mahal_cluster_df[[COL_GENE, "in_optimized_set"]],
            on=COL_GENE, how="left",
        )
        mahal_block["in_goi"] = mahal_block[COL_GENE].isin(matched)
        goi_m = mahal_block[mahal_block["in_goi"]]
        bg_m = mahal_block[~mahal_block["in_goi"]]
        # Joint dropna on length + flag so the length array stays aligned
        # with the membership flag for length-matched sampling.
        gj = goi_m[["length", "in_optimized_set"]].dropna()
        bj = bg_m[["length", "in_optimized_set"]].dropna()
        if len(gj) >= 3 and len(bj) >= 30:
            goi_in = gj["in_optimized_set"].astype(bool).values
            bg_in = bj["in_optimized_set"].astype(bool).values
            obs_rate = float(goi_in.mean())
            bg_rate = float(bg_in.mean())
            use_lm = length_matched
            perm_rates = np.empty(n_permutations)
            if use_lm:
                bg_lengths = bj["length"].values
                goi_lengths = gj["length"].values
                for i in range(n_permutations):
                    idx = _length_matched_indices(bg_lengths, goi_lengths, rng)
                    perm_rates[i] = bg_in[idx].mean()
            else:
                for i in range(n_permutations):
                    idx = rng.choice(len(bg_in), size=len(goi_in), replace=False)
                    perm_rates[i] = bg_in[idx].mean()
            p_more = (np.sum(perm_rates >= obs_rate) + 1) / (n_permutations + 1)
            p_less = (np.sum(perm_rates <= obs_rate) + 1) / (n_permutations + 1)
            p_two = float(min(1.0, 2 * min(p_more, p_less)))
            mahal_membership_row = pd.DataFrame([{
                "test": "in_mahal_cluster_rate",
                "n_goi": int(len(goi_in)),
                "n_background": int(len(bg_in)),
                "obs_rate_goi": round(obs_rate, 4),
                "rate_background": round(bg_rate, 4),
                "perm_mean_rate": round(float(perm_rates.mean()), 4),
                "p_value_two_sided": p_two,
                "p_more_in_cluster": float(p_more),
                "p_less_in_cluster": float(p_less),
                "n_perm": n_permutations,
                "null_set": "length_matched" if use_lm else "uniform_random",
            }])
            mahal_path = output_dir / f"{sample_id}_goi_mahal_membership.tsv"
            mahal_membership_row.to_csv(mahal_path, sep="\t", index=False)
            out["mahal_membership"] = mahal_path
            logger.info(
                "Wrote Mahal-cluster membership test: %s (GOI in-cluster rate %.3f vs background %.3f, p=%.4f)",
                mahal_path, obs_rate, bg_rate, p_two,
            )

    # ── 8. Summary figure ──────────────────────────────────────────────────
    if make_figure:
        try:
            from codonpipe.modules._gene_set_figure import render_summary_figure
            png_path, svg_path = render_summary_figure(
                output_dir=output_dir,
                sample_id=sample_id,
                rscu_gene_df=rscu_gene_df,
                enc_df=enc_df,
                expr_df=expr_df,
                hgt_df=hgt_df,
                summary_df=summary_df,
                scalar_tests=scalar_tests,
                codon_tests=codon_tests,
                goi_ids=matched,
                rscu_genome=rscu_genome,
                rscu_rp=rscu_rp,
                rscu_mahal_cluster=rscu_mahal_cluster,
                mahal_cluster_df=mahal_cluster_df,
                partition_table=partition_table if 'partition_table' in dir() else None,
                partition_per_gene=flag_block[[COL_GENE, "partition"]] if 'flag_block' in dir() else None,
            )
            if png_path is not None:
                out["figure_png"] = png_path
                out["figure_svg"] = svg_path
        except Exception as e:
            logger.warning("Figure rendering failed (%s); continuing without figure", e)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# File-system loader (per-sample output directory layout)
# ──────────────────────────────────────────────────────────────────────────────


def _read_score_tsv(path: Path, score_name: str) -> pd.DataFrame | None:
    """Read a coRdon-flavoured TSV with 'self'/'gene'/'width' columns and rename score."""
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    # Rename the first numeric column that isn't gene/width to *score_name*
    meta = {COL_GENE, "width", "length", "sample_id"}
    for c in df.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df = df.rename(columns={c: score_name})
            break
    return df


def _read_single_row_rscu(path: Path) -> dict[str, float] | None:
    """Read a per-genome single-row RSCU TSV (median or ribosomal) into a codon dict."""
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        c: float(row[c])
        for c in df.columns
        if c in RSCU_COLUMN_NAMES and pd.notna(row[c])
    }


def _read_codon_keyed_rscu(path: Path) -> dict[str, float] | None:
    """Read a 'codon\\tRSCU' TSV (e.g. Mahalanobis-cluster file)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", header=0)
    # File has empty header on first column; make it 'codon'
    if df.columns[0] != "codon":
        df = df.rename(columns={df.columns[0]: "codon"})
    if "RSCU" not in df.columns:
        # Some variants name it differently; pick first numeric column
        for c in df.columns[1:]:
            if pd.api.types.is_numeric_dtype(df[c]):
                df = df.rename(columns={c: "RSCU"})
                break
    return {row["codon"]: float(row["RSCU"]) for _, row in df.iterrows() if pd.notna(row["RSCU"])}


def load_sample_outputs(sample_dir: Path, sample_id: str) -> dict:
    """Load the standard per-sample output files into a dict ready for analyze_gene_set.

    Missing files are silently skipped with a warning. The keys correspond
    one-to-one to the keyword arguments of analyze_gene_set.
    """
    sample_dir = Path(sample_dir)
    rscu_dir = sample_dir / "rscu"
    cu_dir = sample_dir / "cu_statistics"
    expr_dir = sample_dir / "expression"
    bio_dir = sample_dir / "bio_ecology"
    gmm_dir = sample_dir / "gmm_clustering"

    rscu_gene_path = rscu_dir / f"{sample_id}_rscu_all_genes.tsv"
    enc_path = rscu_dir / f"{sample_id}_enc.tsv"
    if not rscu_gene_path.exists() or not enc_path.exists():
        raise FileNotFoundError(
            f"Required files not found under {sample_dir}: "
            f"need rscu/{sample_id}_rscu_all_genes.tsv and rscu/{sample_id}_enc.tsv"
        )

    rscu_gene_df = pd.read_csv(rscu_gene_path, sep="\t")
    enc_df = pd.read_csv(enc_path, sep="\t")

    expr_path = expr_dir / f"{sample_id}_expression.tsv"
    expr_df = pd.read_csv(expr_path, sep="\t") if expr_path.exists() else None

    encprime_df = _read_score_tsv(cu_dir / f"{sample_id}_encprime.tsv", "ENCprime")
    milc_df = _read_score_tsv(cu_dir / f"{sample_id}_milc.tsv", "MILC")

    hgt_path = bio_dir / f"{sample_id}_hgt_candidates.tsv"
    hgt_df = pd.read_csv(hgt_path, sep="\t") if hgt_path.exists() else None

    rscu_genome = _read_single_row_rscu(rscu_dir / f"{sample_id}_rscu_median.tsv")
    rscu_rp = _read_single_row_rscu(rscu_dir / f"{sample_id}_rscu_ribosomal.tsv")
    rscu_mahal = _read_codon_keyed_rscu(gmm_dir / f"{sample_id}_gmm_cluster_rscu.tsv")
    if rscu_mahal is not None:
        # The codon-keyed file uses bare codons; remap to RSCU column names.
        from codonpipe.utils.codon_tables import codon_to_col_name, CODON_TABLE_11
        remapped = {}
        for codon, val in rscu_mahal.items():
            aa = CODON_TABLE_11.get(codon)
            if aa is None or aa in ("*", "Met", "Trp"):
                continue
            col = codon_to_col_name(codon, aa)
            remapped[col] = val
        rscu_mahal = remapped if remapped else rscu_mahal

    # Per-gene Mahal-cluster table: distance-to-cluster + in_optimized_set flag.
    mahal_clusters_path = gmm_dir / f"{sample_id}_gmm_clusters.tsv"
    mahal_cluster_df = None
    if mahal_clusters_path.exists():
        mahal_cluster_df = pd.read_csv(mahal_clusters_path, sep="\t")
        # Rename so it doesn't collide with hgt_df.mahalanobis_dist (which is
        # genome-centroid distance, not cluster-centroid).
        mahal_cluster_df = mahal_cluster_df.rename(columns={
            "mahalanobis_distance": "mahal_cluster_distance",
        })

    # CBI tables: one keyed against RP optimal codons, one against Mahal-cluster.
    cbi_dir = sample_dir / "codon_tables"
    cbi_rp_df = None
    cbi_rp_path = cbi_dir / f"{sample_id}_cbi.tsv"
    if cbi_rp_path.exists():
        cbi_rp_df = pd.read_csv(cbi_rp_path, sep="\t")[[COL_GENE, "cbi"]].rename(
            columns={"cbi": "cbi_rp"}
        )

    cbi_mahal_df = None
    cbi_mahal_path = cbi_dir / f"{sample_id}_gmm_cluster_cbi.tsv"
    if cbi_mahal_path.exists():
        cbi_mahal_df = pd.read_csv(cbi_mahal_path, sep="\t")[[COL_GENE, "cbi"]].rename(
            columns={"cbi": "cbi_mahal"}
        )

    return dict(
        rscu_gene_df=rscu_gene_df,
        enc_df=enc_df,
        expr_df=expr_df,
        encprime_df=encprime_df,
        milc_df=milc_df,
        hgt_df=hgt_df,
        mahal_cluster_df=mahal_cluster_df,
        cbi_rp_df=cbi_rp_df,
        cbi_mahal_df=cbi_mahal_df,
        rscu_genome=rscu_genome,
        rscu_rp=rscu_rp,
        rscu_mahal_cluster=rscu_mahal,
    )


def read_goi_file(path: Path) -> set[str]:
    """Read a GOI list file (one locus tag per line; '#' comments allowed)."""
    ids: set[str] = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on whitespace and take the first token to allow trailing
            # comments after the locus tag.
            ids.add(line.split()[0])
    return ids
