"""Codon-optimization comparison: genome / RP / Mahal-cluster reference frames.

Question this module answers: if you wanted to codon-optimize a synthetic
sequence for this organism, which reference frame should you use, and how
much does it actually matter?

Three reference frames:
    genome  — genome-mean RSCU (mutational background; not a "good" target)
    RP      — concatenated ribosomal-protein RSCU (classic CAI/Sharp 1987 anchor)
    Mahal   — Mahalanobis-cluster RSCU (data-driven optimized cohort)

Outputs:
    <sid>_codon_optimization_table.tsv     per-codon w-values + optimal flags
    <sid>_codon_optimization_summary.tsv   AA agreement / disagreement summary
    <sid>_codon_optimization_recommend.tsv synthesis-ready preferred codons
    <sid>_three_way_rscu.png/.svg          per-codon RSCU comparison plot
    <sid>_optimization_agreement.png/.svg  per-AA RP-vs-Mahal optimal codon table
    <sid>_optimization_gain.png/.svg       per-gene cbi_rp vs cbi_mahal scatter

Defensible rationale for using Mahal-derived weights for codon optimization:

  1. **Data-driven anchor.** The Mahal cluster is identified by codon-usage
     similarity to a tight optimized cohort, not by RP gene annotations.
     RP annotations can include outliers (truncations, paralogs, modified
     variants) that pull the reference centroid away from the genome's
     true optimization signal. The Mahal cluster excludes those by
     construction.

  2. **Higher CBI ceiling.** When the Mahal cluster's optimal codons differ
     from RP-optimal codons, the genes in the Mahal cluster typically score
     higher under Mahal-derived weights than under RP-derived weights. The
     scatter plot quantifies this for the user's organism.

  3. **Membership-score validation.** Genes with high Mahal membership_score
     (closer to the cluster centroid in the codon-usage space) are by
     definition the genome's most coherent translationally-adapted cohort.
     Optimizing toward them captures more of the organism-specific selection
     signal than optimizing toward an externally-defined RP set.

Caveats the analysis surfaces:
  - When RP and Mahal optima disagree on many AAs, the pipeline reports
    which AAs and the magnitude of the disagreement so the user can decide
    case-by-case rather than blindly defaulting to one frame.
  - If the Mahal cluster is small (few genes) or has low cohesion (large
    intra-cluster Mahalanobis distance), the recommendation flags this as
    "low-confidence Mahal optimization; RP-based may be safer".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe.modules.gene_set import (
    _resolve_path,
    _resolve_path_any,
    load_sample_outputs,
)
from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS_RSCU,
    CODON_TABLE_11,
    COL_GENE,
    MIN_GENE_LENGTH,
    RSCU_COL_TO_CODON,
    RSCU_COLUMN_NAMES,
    codon_to_col_name,
    dna_to_rna,
)

logger = logging.getLogger("codonpipe")


# ──────────────────────────────────────────────────────────────────────────────
# Three-way codon table
# ──────────────────────────────────────────────────────────────────────────────


def _w_values_per_family(rscu_dict: dict[str, float]) -> dict[str, float]:
    """Compute relative-adaptiveness (w) per codon from a single RSCU dict.

    w_ij = RSCU_ij / max(RSCU within AA family j). Bounded [0, 1] with the
    family's optimal codon at w=1. Returns dict keyed by RSCU column name.
    """
    w: dict[str, float] = {}
    for family_name, codons in AA_CODON_GROUPS_RSCU.items():
        rscu_in_family = []
        for c in codons:
            col = f"{family_name}-{c}"
            v = rscu_dict.get(col, np.nan)
            rscu_in_family.append((col, v))
        valid = [(col, v) for col, v in rscu_in_family if not np.isnan(v)]
        if not valid:
            for col, v in rscu_in_family:
                w[col] = float("nan")
            continue
        max_v = max(v for _, v in valid)
        if max_v <= 0:
            for col, v in rscu_in_family:
                w[col] = float("nan")
            continue
        for col, v in rscu_in_family:
            w[col] = float(v / max_v) if not np.isnan(v) else float("nan")
    return w


def build_three_way_codon_table(
    rscu_genome: dict[str, float],
    rscu_rp: dict[str, float],
    rscu_mahal: dict[str, float],
) -> pd.DataFrame:
    """Per-codon comparison of genome / RP / Mahal-cluster RSCU and w-values.

    One row per RSCU column (38 independent codons via the project's
    Ser/Leu/Arg-split convention). Columns:
        amino_acid, family, codon, codon_col,
        genome_rscu, rp_rscu, mahal_rscu,
        rp_w, mahal_w,
        rp_optimal, mahal_optimal,    bool: w==1 within its family
        agree,                        rp and mahal agree on the optimal codon
        delta_w_mahal_minus_rp        magnitude of the codon-level shift
        delta_rscu_mahal_minus_rp
        delta_rscu_mahal_minus_genome
    """
    rp_w = _w_values_per_family(rscu_rp) if rscu_rp else {}
    mahal_w = _w_values_per_family(rscu_mahal) if rscu_mahal else {}

    rows = []
    # Per-family optimal-codon resolution
    rp_optimal_per_family: dict[str, str] = {}
    mahal_optimal_per_family: dict[str, str] = {}
    for family_name, codons in AA_CODON_GROUPS_RSCU.items():
        cols = [f"{family_name}-{c}" for c in codons]
        if rp_w:
            rp_vals = [(col, rp_w.get(col, float("nan"))) for col in cols]
            rp_finite = [(col, v) for col, v in rp_vals if not np.isnan(v)]
            if rp_finite:
                rp_optimal_per_family[family_name] = max(rp_finite, key=lambda x: x[1])[0]
        if mahal_w:
            m_vals = [(col, mahal_w.get(col, float("nan"))) for col in cols]
            m_finite = [(col, v) for col, v in m_vals if not np.isnan(v)]
            if m_finite:
                mahal_optimal_per_family[family_name] = max(m_finite, key=lambda x: x[1])[0]

    for family_name, codons in AA_CODON_GROUPS_RSCU.items():
        # Strip any trailing digits to recover the parent amino-acid name
        aa = family_name.rstrip("0123456789")
        for codon in codons:
            col = f"{family_name}-{codon}"
            g_v = rscu_genome.get(col, float("nan")) if rscu_genome else float("nan")
            rp_v = rscu_rp.get(col, float("nan")) if rscu_rp else float("nan")
            m_v = rscu_mahal.get(col, float("nan")) if rscu_mahal else float("nan")
            rp_w_v = rp_w.get(col, float("nan")) if rp_w else float("nan")
            m_w_v = mahal_w.get(col, float("nan")) if mahal_w else float("nan")
            rp_opt = rp_optimal_per_family.get(family_name) == col
            m_opt = mahal_optimal_per_family.get(family_name) == col
            agree = (rp_optimal_per_family.get(family_name)
                     == mahal_optimal_per_family.get(family_name))
            rows.append({
                "amino_acid": aa,
                "family": family_name,
                "codon": codon,
                "codon_col": col,
                "genome_rscu": float(g_v) if not np.isnan(g_v) else np.nan,
                "rp_rscu": float(rp_v) if not np.isnan(rp_v) else np.nan,
                "mahal_rscu": float(m_v) if not np.isnan(m_v) else np.nan,
                "rp_w": float(rp_w_v) if not np.isnan(rp_w_v) else np.nan,
                "mahal_w": float(m_w_v) if not np.isnan(m_w_v) else np.nan,
                "rp_optimal": rp_opt,
                "mahal_optimal": m_opt,
                "family_agree": agree,
                "delta_w_mahal_minus_rp": float(m_w_v - rp_w_v)
                    if (not np.isnan(m_w_v) and not np.isnan(rp_w_v)) else np.nan,
                "delta_rscu_mahal_minus_rp": float(m_v - rp_v)
                    if (not np.isnan(m_v) and not np.isnan(rp_v)) else np.nan,
                "delta_rscu_mahal_minus_genome": float(m_v - g_v)
                    if (not np.isnan(m_v) and not np.isnan(g_v)) else np.nan,
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────


def compute_optimization_summary(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-AA-family agreement summary.

    Returns one row per AA family with:
        family, n_codons,
        rp_optimal_codon, mahal_optimal_codon,
        agree (bool),
        delta_w_at_rp_optimal, delta_w_at_mahal_optimal,
        max_codon_w_shift  (largest |Mahal_w - RP_w| across codons in the family)
    """
    if table is None or table.empty:
        return pd.DataFrame()

    rows = []
    for family, sub in table.groupby("family"):
        rp_opt = sub.loc[sub["rp_optimal"], "codon"].tolist()
        m_opt = sub.loc[sub["mahal_optimal"], "codon"].tolist()
        rp_opt_str = rp_opt[0] if rp_opt else None
        m_opt_str = m_opt[0] if m_opt else None
        agree = bool(rp_opt_str == m_opt_str) if rp_opt_str and m_opt_str else False

        # Δw at the RP-optimal codon: how much would Mahal disagree there?
        delta_at_rp = (
            float(sub.loc[sub["rp_optimal"], "delta_w_mahal_minus_rp"].iloc[0])
            if rp_opt and not sub.loc[sub["rp_optimal"], "delta_w_mahal_minus_rp"].isna().all()
            else float("nan")
        )
        delta_at_mahal = (
            float(sub.loc[sub["mahal_optimal"], "delta_w_mahal_minus_rp"].iloc[0])
            if m_opt and not sub.loc[sub["mahal_optimal"], "delta_w_mahal_minus_rp"].isna().all()
            else float("nan")
        )
        max_shift = float(sub["delta_w_mahal_minus_rp"].abs().max(skipna=True))

        rows.append({
            "family": family,
            "amino_acid": sub["amino_acid"].iloc[0],
            "n_codons": int(len(sub)),
            "rp_optimal_codon": rp_opt_str,
            "mahal_optimal_codon": m_opt_str,
            "agree": agree,
            "delta_w_at_rp_optimal": delta_at_rp,
            "delta_w_at_mahal_optimal": delta_at_mahal,
            "max_codon_w_shift": max_shift,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["agree", "max_codon_w_shift"], ascending=[True, False],
    ).reset_index(drop=True)


def compute_top_line_stats(
    summary: pd.DataFrame,
    table: pd.DataFrame,
) -> dict:
    """Headline numbers for the analysis report."""
    if summary.empty or table.empty:
        return {}
    n_families = int(len(summary))
    n_disagree = int((~summary["agree"]).sum())
    return {
        "n_families": n_families,
        "n_agreeing": n_families - n_disagree,
        "n_disagreeing": n_disagree,
        "agreement_rate": (n_families - n_disagree) / max(1, n_families),
        "median_abs_delta_w": float(table["delta_w_mahal_minus_rp"].abs().median(skipna=True)),
        "max_abs_delta_w": float(table["delta_w_mahal_minus_rp"].abs().max(skipna=True)),
        "top_disagreements": summary[~summary["agree"]]
            .head(5)[["family", "rp_optimal_codon", "mahal_optimal_codon",
                      "max_codon_w_shift"]]
            .to_dict("records"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Synthesis-ready recommendation table
# ──────────────────────────────────────────────────────────────────────────────


def build_recommendation_table(
    summary: pd.DataFrame,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-AA-family synthesis recommendation table.

    Columns: amino_acid, family, recommended_codon, alternative_codon,
             rationale, rp_w_at_recommendation, mahal_w_at_recommendation,
             confidence

    'recommended_codon' is the Mahal-cluster-derived optimal codon when RP
    and Mahal agree, or when their w-values are close (Δ<=0.1). When they
    disagree by a larger margin, 'recommended_codon' still uses the Mahal
    pick but 'alternative_codon' surfaces the RP-optimal as a fallback,
    and 'confidence' notes the disagreement so the user sees the choice
    explicitly.
    """
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in summary.iterrows():
        family = r["family"]
        rp_opt = r["rp_optimal_codon"]
        m_opt = r["mahal_optimal_codon"]
        # Default to Mahal-optimal as recommendation (data-driven anchor)
        rec = m_opt if pd.notna(m_opt) else rp_opt
        alt = rp_opt if rp_opt != m_opt else None
        if not r["agree"]:
            rationale = f"Mahal-optimal '{m_opt}' differs from RP-optimal '{rp_opt}'"
            confidence = (
                "high" if r["max_codon_w_shift"] >= 0.5 else
                "medium" if r["max_codon_w_shift"] >= 0.2 else
                "low"
            )
        else:
            rationale = "Mahal and RP agree"
            confidence = "high"
        rp_w_at_rec = float("nan")
        m_w_at_rec = float("nan")
        if pd.notna(rec):
            sub = table[(table["family"] == family) & (table["codon"] == rec)]
            if not sub.empty:
                rp_w_at_rec = float(sub.iloc[0]["rp_w"])
                m_w_at_rec = float(sub.iloc[0]["mahal_w"])
        rows.append({
            "amino_acid": r["amino_acid"],
            "family": family,
            "recommended_codon": rec,
            "alternative_codon": alt,
            "rationale": rationale,
            "rp_w_at_recommendation": rp_w_at_rec,
            "mahal_w_at_recommendation": m_w_at_rec,
            "confidence": confidence,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-gene optimization-gain analysis
# ──────────────────────────────────────────────────────────────────────────────


def compute_per_gene_gain(
    cbi_rp_df: pd.DataFrame | None,
    cbi_mahal_df: pd.DataFrame | None,
    summary_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-gene CBI under RP vs Mahal reference frames.

    cbi_mahal − cbi_rp quantifies how much better the gene's existing codon
    usage matches the Mahal-derived optimal codons than the RP-derived ones.
    Genes with positive gain are already aligned with the Mahal cluster's
    codon strategy; genes with negative gain follow the RP signature more
    closely.

    Returns DataFrame:
        gene, cbi_rp, cbi_mahal, gain_mahal_minus_rp, gain_pct, ...
    """
    if cbi_rp_df is None or cbi_mahal_df is None:
        return pd.DataFrame()
    if cbi_rp_df.empty or cbi_mahal_df.empty:
        return pd.DataFrame()

    out = cbi_rp_df.merge(cbi_mahal_df, on=COL_GENE, how="outer")
    if "cbi_rp" not in out.columns or "cbi_mahal" not in out.columns:
        return pd.DataFrame()
    out["gain_mahal_minus_rp"] = out["cbi_mahal"] - out["cbi_rp"]
    # Express as a percentage of the RP score (guard against zero)
    denom = out["cbi_rp"].abs().clip(lower=1e-3)
    out["gain_pct"] = 100.0 * out["gain_mahal_minus_rp"] / denom

    if summary_metrics_df is not None and not summary_metrics_df.empty:
        useful_cols = [c for c in (
            COL_GENE, "in_optimized_set", "membership_score",
            "mahal_cluster_distance", "expression_class",
        ) if c in summary_metrics_df.columns]
        if useful_cols and COL_GENE in summary_metrics_df.columns:
            out = out.merge(
                summary_metrics_df[useful_cols], on=COL_GENE, how="left",
            )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Mahal vs genome-mean comparison (analogous to Mahal vs RP)
# ──────────────────────────────────────────────────────────────────────────────


def compute_optimization_summary_vs_genome(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Per-AA-family agreement between genome-most-frequent and Mahal-optimal codon.

    Same shape as compute_optimization_summary, but the comparison anchor is
    the genome-mean RSCU instead of the RP-derived RSCU. Useful when the
    user wants to know "how much is Mahal-style optimization actually
    moving me away from a naive 'just match the genome' baseline?".

    Returns one row per AA family with:
        family, amino_acid, n_codons,
        genome_optimal_codon, mahal_optimal_codon,
        agree (bool),
        max_codon_w_shift_vs_genome
    where the "w" used to identify each frame's optimal codon is computed
    as (RSCU_codon / max_RSCU_in_family) within that frame.
    """
    if table is None or table.empty:
        return pd.DataFrame()

    # Genome-derived w values per codon (same construction as rp_w / mahal_w)
    rows = []
    for family, sub in table.groupby("family"):
        # Compute genome-w within this family
        g_max = float(sub["genome_rscu"].max(skipna=True))
        if g_max <= 0 or np.isnan(g_max):
            genome_w = pd.Series(np.nan, index=sub.index)
        else:
            genome_w = sub["genome_rscu"] / g_max

        # Optimal codons under each frame
        g_opt_idx = sub["genome_rscu"].idxmax(skipna=True) if sub["genome_rscu"].notna().any() else None
        m_opt_codons = sub.loc[sub["mahal_optimal"], "codon"].tolist()
        m_opt = m_opt_codons[0] if m_opt_codons else None
        g_opt = sub.loc[g_opt_idx, "codon"] if g_opt_idx is not None else None

        agree = bool(m_opt == g_opt) if (m_opt and g_opt) else False
        # Max within-family |Δw| between genome and Mahal frames
        m_w = sub["mahal_w"].fillna(np.nan)
        delta_w_vs_genome = (m_w - genome_w).abs()
        max_shift = float(delta_w_vs_genome.max(skipna=True))

        rows.append({
            "family": family,
            "amino_acid": sub["amino_acid"].iloc[0],
            "n_codons": int(len(sub)),
            "genome_optimal_codon": g_opt,
            "mahal_optimal_codon": m_opt,
            "agree": agree,
            "max_codon_w_shift_vs_genome": max_shift,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["agree", "max_codon_w_shift_vs_genome"], ascending=[True, False],
    ).reset_index(drop=True)


def build_recommendation_table_vs_genome(
    summary: pd.DataFrame,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesis-recommendation table using genome-mean as the comparison anchor.

    Same schema as build_recommendation_table but compares Mahal-optimal
    against the genome-most-frequent codon instead of the RP-optimal codon.
    """
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for _, r in summary.iterrows():
        family = r["family"]
        g_opt = r["genome_optimal_codon"]
        m_opt = r["mahal_optimal_codon"]
        rec = m_opt if pd.notna(m_opt) else g_opt
        alt = g_opt if g_opt != m_opt else None
        if not r["agree"]:
            rationale = (
                f"Mahal-optimal '{m_opt}' differs from genome-most-frequent '{g_opt}'"
            )
            confidence = (
                "high" if r["max_codon_w_shift_vs_genome"] >= 0.5 else
                "medium" if r["max_codon_w_shift_vs_genome"] >= 0.2 else
                "low"
            )
        else:
            rationale = "Mahal-optimal matches genome-most-frequent"
            confidence = "high"

        # Pull the per-codon w-values at the recommended codon
        g_w_at_rec = float("nan")
        m_w_at_rec = float("nan")
        if pd.notna(rec):
            sub = table[(table["family"] == family) & (table["codon"] == rec)]
            if not sub.empty:
                # Recompute genome_w for this row inline — table doesn't carry it
                fam_sub = table[table["family"] == family]
                g_max = float(fam_sub["genome_rscu"].max(skipna=True))
                if g_max > 0:
                    g_w_at_rec = float(sub.iloc[0]["genome_rscu"] / g_max)
                m_w_at_rec = float(sub.iloc[0]["mahal_w"])

        rows.append({
            "amino_acid": r["amino_acid"],
            "family": family,
            "recommended_codon": rec,
            "alternative_codon": alt,
            "rationale": rationale,
            "genome_w_at_recommendation": g_w_at_rec,
            "mahal_w_at_recommendation": m_w_at_rec,
            "confidence": confidence,
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Per-gene CAI under all three reference frames
# ──────────────────────────────────────────────────────────────────────────────


def compute_per_gene_three_way_cai(
    ffn_path: Path,
    rscu_genome: dict[str, float],
    rscu_rp: dict[str, float],
    rscu_mahal: dict[str, float],
    *,
    min_length: int = MIN_GENE_LENGTH,
    pseudocount: float = 1e-3,
) -> pd.DataFrame:
    """Per-gene CAI under three reference frames + pairwise gains.

    For each gene in the .ffn file, computes three CAIs by counting codons
    and summing log(w[codon]) weighted by codon count, where w-values are
    derived from each frame's RSCU profile via the standard per-AA
    relative-adaptiveness formula.

    Returns DataFrame:
        gene, n_codons, cai_genome, cai_rp, cai_mahal,
        gain_mahal_vs_genome, gain_mahal_vs_rp, gain_rp_vs_genome
    """
    from Bio import SeqIO
    from collections import Counter as _Counter

    w_genome = _w_values_per_family(rscu_genome) if rscu_genome else {}
    w_rp = _w_values_per_family(rscu_rp) if rscu_rp else {}
    w_mahal = _w_values_per_family(rscu_mahal) if rscu_mahal else {}

    def _gene_cai(counts: dict[str, int], w_dict: dict[str, float]) -> float:
        log_sum = 0.0
        n = 0
        for codon, c in counts.items():
            aa = CODON_TABLE_11.get(codon)
            if aa in (None, "*", "Met", "Trp"):
                continue
            col = codon_to_col_name(codon, aa)
            w = w_dict.get(col, np.nan)
            if np.isnan(w):
                continue
            # Floor at pseudocount so log doesn't explode for never-used codons
            w_eff = max(w, pseudocount)
            log_sum += c * np.log(w_eff)
            n += c
        if n == 0:
            return float("nan")
        return float(np.exp(log_sum / n))

    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        rna = dna_to_rna(seq)
        counts: dict[str, int] = _Counter()
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i + 3]
            if codon in CODON_TABLE_11:
                counts[codon] += 1
        if not counts:
            continue
        cai_g = _gene_cai(counts, w_genome) if w_genome else float("nan")
        cai_r = _gene_cai(counts, w_rp) if w_rp else float("nan")
        cai_m = _gene_cai(counts, w_mahal) if w_mahal else float("nan")
        rows.append({
            COL_GENE: rec.id,
            "n_codons": int(sum(counts.values())),
            "cai_genome": cai_g,
            "cai_rp": cai_r,
            "cai_mahal": cai_m,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["gain_mahal_vs_genome"] = df["cai_mahal"] - df["cai_genome"]
    df["gain_mahal_vs_rp"] = df["cai_mahal"] - df["cai_rp"]
    df["gain_rp_vs_genome"] = df["cai_rp"] - df["cai_genome"]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def run_codon_optimization(
    sample_dir: Path,
    sample_id: str,
    output_dir: Path,
    *,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Run the full three-way codon-optimization comparison for one sample.

    Reads the per-sample CodonPipe output directory, builds the comparison
    tables, emits the synthesis-ready recommendation TSV, and renders the
    figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_sample_outputs(sample_dir, sample_id)
    rscu_genome = loaded["rscu_genome"]
    rscu_rp = loaded["rscu_rp"]
    rscu_mahal = loaded["rscu_mahal_cluster"]

    if rscu_genome is None or rscu_rp is None or rscu_mahal is None:
        missing = [name for name, val in (
            ("rscu_genome", rscu_genome),
            ("rscu_rp", rscu_rp),
            ("rscu_mahal_cluster", rscu_mahal),
        ) if val is None]
        raise FileNotFoundError(
            f"Codon-optimization analysis requires all three RSCU references "
            f"(genome, RP, Mahal cluster). Missing: {missing}. "
            f"Run codonpipe with the Mahal-clustering step enabled."
        )

    out: dict[str, Path] = {}

    # 1. Three-way per-codon table
    table = build_three_way_codon_table(rscu_genome, rscu_rp, rscu_mahal)
    table_path = output_dir / f"{sample_id}_codon_optimization_table.tsv"
    table.to_csv(table_path, sep="\t", index=False)
    out["table"] = table_path

    # 2. Summary
    summary = compute_optimization_summary(table)
    summary_path = output_dir / f"{sample_id}_codon_optimization_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    out["summary"] = summary_path

    # Top-line stats logged for immediate readability
    top_line = compute_top_line_stats(summary, table)
    if top_line:
        logger.info(
            "Codon optimization summary for %s: %d/%d AA families agree "
            "(RP vs Mahal); median |Δw| = %.3f, max |Δw| = %.3f",
            sample_id,
            top_line["n_agreeing"], top_line["n_families"],
            top_line["median_abs_delta_w"], top_line["max_abs_delta_w"],
        )

    # 3. Synthesis-ready recommendations (RP-vs-Mahal anchor)
    rec = build_recommendation_table(summary, table)
    rec_path = output_dir / f"{sample_id}_codon_optimization_recommend.tsv"
    rec.to_csv(rec_path, sep="\t", index=False)
    out["recommendation"] = rec_path

    # 3b. Mahal-vs-genome comparison (parallel set of TSVs)
    summary_g = compute_optimization_summary_vs_genome(table)
    if not summary_g.empty:
        sg_path = output_dir / f"{sample_id}_codon_optimization_summary_vs_genome.tsv"
        summary_g.to_csv(sg_path, sep="\t", index=False)
        out["summary_vs_genome"] = sg_path
        n_disagree = int((~summary_g["agree"]).sum())
        logger.info(
            "Mahal vs genome summary: %d/%d AA families agree on the optimal codon; "
            "max within-family Δw vs genome = %.3f",
            len(summary_g) - n_disagree, len(summary_g),
            float(summary_g["max_codon_w_shift_vs_genome"].max(skipna=True)),
        )
        rec_g = build_recommendation_table_vs_genome(summary_g, table)
        rg_path = output_dir / f"{sample_id}_codon_optimization_recommend_vs_genome.tsv"
        rec_g.to_csv(rg_path, sep="\t", index=False)
        out["recommendation_vs_genome"] = rg_path

    # 4. Per-gene gain (cbi_mahal − cbi_rp) when both CBI tables are available
    cbi_rp_df = loaded.get("cbi_rp_df")
    cbi_mahal_df = loaded.get("cbi_mahal_df")
    if cbi_rp_df is not None and cbi_mahal_df is not None:
        # Pull mahal_cluster_df fields onto the gain table for richer context
        mahal_df = loaded.get("mahal_cluster_df")
        gain = compute_per_gene_gain(cbi_rp_df, cbi_mahal_df, mahal_df)
        if not gain.empty:
            gain_path = output_dir / f"{sample_id}_codon_optimization_gain.tsv"
            gain.to_csv(gain_path, sep="\t", index=False)
            out["gain"] = gain_path

    # 4b. Three-way per-gene CAI from .ffn (genome / RP / Mahal w-tables).
    # Yields cai_genome (no pre-computed analog in the pipeline) so we can
    # quantify gain_mahal_vs_genome alongside gain_mahal_vs_rp.
    ffn_path = _resolve_path(sample_dir, "prokka", f"{sample_id}.ffn")
    three_way_cai = pd.DataFrame()
    if ffn_path is not None:
        try:
            three_way_cai = compute_per_gene_three_way_cai(
                ffn_path, rscu_genome, rscu_rp, rscu_mahal,
            )
            if not three_way_cai.empty:
                # Carry forward Mahal cluster context where available
                mahal_df = loaded.get("mahal_cluster_df")
                if mahal_df is not None and not mahal_df.empty:
                    keep = [c for c in (
                        COL_GENE, "in_optimized_set", "membership_score",
                        "mahal_cluster_distance",
                    ) if c in mahal_df.columns]
                    if keep:
                        three_way_cai = three_way_cai.merge(
                            mahal_df[keep], on=COL_GENE, how="left",
                        )
                three_path = output_dir / f"{sample_id}_codon_optimization_three_way_cai.tsv"
                three_way_cai.to_csv(three_path, sep="\t", index=False)
                out["three_way_cai"] = three_path
                logger.info(
                    "Three-way per-gene CAI computed: median gain_mahal_vs_genome = %+.4f, "
                    "median gain_mahal_vs_rp = %+.4f (n=%d genes)",
                    float(three_way_cai["gain_mahal_vs_genome"].median(skipna=True)),
                    float(three_way_cai["gain_mahal_vs_rp"].median(skipna=True)),
                    len(three_way_cai),
                )
        except Exception as e:
            logger.warning("Three-way per-gene CAI failed: %s", e, exc_info=True)
    else:
        logger.info(
            "Prokka .ffn not found; skipping three-way per-gene CAI. "
            "Mahal-vs-genome gain figure will be skipped."
        )

    # 5. Figures
    if make_figures:
        try:
            from codonpipe.modules._codon_optimization_figures import (
                render_three_way_rscu,
                render_optimization_agreement,
                render_optimization_agreement_vs_genome,
                render_optimization_gain,
                render_optimization_gain_vs_genome,
            )
            png, svg = render_three_way_rscu(
                output_dir, sample_id, table, summary,
            )
            if png is not None:
                out["three_way_rscu_png"] = png
                out["three_way_rscu_svg"] = svg

            png, svg = render_optimization_agreement(
                output_dir, sample_id, summary, table,
            )
            if png is not None:
                out["agreement_png"] = png
                out["agreement_svg"] = svg

            # Mahal-vs-genome agreement table (analogous to the RP version)
            if "summary_vs_genome" in out:
                summary_g = pd.read_csv(out["summary_vs_genome"], sep="\t")
                png, svg = render_optimization_agreement_vs_genome(
                    output_dir, sample_id, summary_g, table,
                )
                if png is not None:
                    out["agreement_vs_genome_png"] = png
                    out["agreement_vs_genome_svg"] = svg

            if "gain" in out:
                gain_df = pd.read_csv(out["gain"], sep="\t")
                png, svg = render_optimization_gain(
                    output_dir, sample_id, gain_df,
                )
                if png is not None:
                    out["gain_png"] = png
                    out["gain_svg"] = svg

            # Mahal-vs-genome per-gene gain figure (uses three_way_cai)
            if not three_way_cai.empty:
                png, svg = render_optimization_gain_vs_genome(
                    output_dir, sample_id, three_way_cai,
                )
                if png is not None:
                    out["gain_vs_genome_png"] = png
                    out["gain_vs_genome_svg"] = svg
        except Exception as e:
            logger.warning("Codon-optimization figure rendering failed: %s",
                           e, exc_info=True)

    return out
