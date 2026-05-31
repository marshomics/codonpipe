"""Codon pair bias (CPB) analysis.

Codon pair bias is the over- or under-representation of specific *adjacent*
codon pairs (dicodons) relative to what is expected from a genome's
single-codon usage AND its amino-acid-pair (dipeptide) usage. By construction
it is independent of single-codon usage, so it captures a layer of
coding-sequence constraint that RSCU / CAI / ENC / Fop cannot. It is directly
relevant to genetic engineering: codon-pair de-optimization attenuates viruses
(Coleman et al. 2008) and codon-pair optimization is used to tune heterologous
expression, so a host's codon-pair score (CPS) table is the reference an
optimizer needs.

Definitions
-----------
Following Coleman et al. (2008, Science 320:1784-1787, PMID 18583614; building
on Gutman & Hatfield 1989, PNAS 86:3699 and Buchan et al. 2006, NAR 34:1015),
for a codon pair (A, B) that encodes the amino-acid pair (X, Y):

    CPS(A, B) = ln( N(A, B) / E(A, B) )

with the expected dicodon count

    E(A, B) = ( N(A) * N(B) / ( N(X) * N(Y) ) ) * N(X, Y)

where, over the reference coding sequences,
    N(A), N(B)   = single-codon counts,
    N(X), N(Y)   = single-amino-acid counts,
    N(X, Y)      = amino-acid-pair (dipeptide) count,
    N(A, B)      = codon-pair (dicodon) count.

E(A, B) is the count expected if, *within each dipeptide class (X, Y)*, the two
synonymous codons were chosen independently in proportion to their genome-wide
usage. CPS > 0 means the pair is used more than that null predicts; CPS < 0,
less. A gene's codon pair bias is the mean CPS over its consecutive
sense-codon pairs:

    CPB(gene) = mean_i CPS(codon_i, codon_{i+1}).

Conventions adopted here (for reproducibility / defensibility)
--------------------------------------------------------------
* Reference is genome-wide (all CDS), matching Coleman et al. Single-codon and
  single-amino-acid counts are taken over all sense codons; codon-pair and
  dipeptide counts over consecutive sense-codon pairs. Pairs in which either
  member is a stop or a non-standard codon are excluded, so no pair spans a
  stop codon.
* NCBI translation table 11 (the pipeline-wide genetic code).
* Minimum gene length = MIN_GENE_LENGTH (240 nt), matching the rest of the
  pipeline.
* CPS is defined only for codon pairs observed in the reference (N(A,B) > 0,
  which guarantees every term in E is > 0, so there is no log(0) on the
  in-genome path). For scoring a *foreign* gene against this host table,
  ``score_sequence_cpb`` reports the fraction of pairs found in the host table
  and, optionally, applies a caller-supplied penalty to host-unobserved pairs.

INTERPRETATION CAVEAT -- read before attributing CPB to translational selection
-------------------------------------------------------------------------------
The *existence* of CPB is uncontroversial and reproducible; its *cause* is
debated. Proposed mechanisms include tRNA re-loading / ribosome pausing at
specific dicodons and internal Shine-Dalgarno-like sequences (Li, Oh &
Weissman 2012, Nature 484:538), but a prominent line of evidence holds that
much of the apparent CPB is a direct consequence of dinucleotide composition
at the codon-codon junction -- the "bridge" dinucleotide -- in particular the
selective avoidance of CpG and UpA, for reasons largely unrelated to
translation (Tulloch et al. 2014, eLife 3:e04531, PMID 25341032; Kunec &
Osterrieder 2016, Cell Rep 14:55, PMID 27278157, whose title is literally
"Codon pair bias is a direct consequence of dinucleotide bias"). Because the
bridge dinucleotide is the leading alternative explanation, this module
computes bridge-dinucleotide observed/expected ratios alongside CPB so the user
can judge whether a genome's CPB is carried by junction dinucleotide
composition rather than by pair-level selection. CPB is therefore emitted as a
descriptive statistic, not as direct evidence of translational selection.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from codonpipe.utils.codon_tables import (
    CODON_TABLE_11,
    COL_GENE,
    MIN_GENE_LENGTH,
    dna_to_rna,
)
from codonpipe.utils.io import get_output_subdir

logger = logging.getLogger("codonpipe")

_STOP = "*"
# Number of possible sense-sense codon pairs: 59 sense codons (61 non-stop
# minus... actually 64 - 3 stops = 61 amino-acid-encoding codons incl. Met/Trp).
# We count pairs over codons that encode an amino acid (non-stop), i.e. 61.
_N_SENSE_CODONS = sum(1 for aa in CODON_TABLE_11.values() if aa != _STOP)
_N_POSSIBLE_PAIRS = _N_SENSE_CODONS * _N_SENSE_CODONS


# ──────────────────────────────────────────────────────────────────────────────
# Counting
# ──────────────────────────────────────────────────────────────────────────────


def _ordered_codons(seq: str) -> list[str]:
    """Return the in-frame codons of a sequence in order (RNA), trailing
    partial codon dropped. Includes stop / non-standard codons so the caller
    can break pairs at them."""
    rna = dna_to_rna(seq)
    return [rna[i:i + 3] for i in range(0, len(rna) - 2, 3)]


def count_codon_pairs(ffn_path: Path, min_length: int = MIN_GENE_LENGTH) -> dict:
    """Accumulate the counts needed for codon-pair scores over a CDS FASTA.

    Single-codon and single-amino-acid counts are taken over all sense codons;
    codon-pair and dipeptide counts over consecutive sense-codon pairs (a pair
    in which either member is a stop or non-standard codon is skipped, so no
    pair spans a stop).

    Returns a dict with: codon_counts, aa_counts, pair_counts, aa_pair_counts
    (all Counters), n_genes, n_pairs.
    """
    codon_counts: Counter = Counter()
    aa_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    aa_pair_counts: Counter = Counter()
    n_genes = 0
    n_pairs = 0

    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        codons = _ordered_codons(seq)
        used = False
        # Single-codon / single-aa counts over sense codons.
        for c in codons:
            aa = CODON_TABLE_11.get(c)
            if aa is None or aa == _STOP:
                continue
            codon_counts[c] += 1
            aa_counts[aa] += 1
            used = True
        # Consecutive sense-codon pairs.
        for i in range(len(codons) - 1):
            c1, c2 = codons[i], codons[i + 1]
            aa1 = CODON_TABLE_11.get(c1)
            aa2 = CODON_TABLE_11.get(c2)
            if aa1 is None or aa1 == _STOP or aa2 is None or aa2 == _STOP:
                continue
            pair_counts[(c1, c2)] += 1
            aa_pair_counts[(aa1, aa2)] += 1
            n_pairs += 1
        if used:
            n_genes += 1

    return {
        "codon_counts": codon_counts,
        "aa_counts": aa_counts,
        "pair_counts": pair_counts,
        "aa_pair_counts": aa_pair_counts,
        "n_genes": n_genes,
        "n_pairs": n_pairs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Codon pair score (CPS) reference table
# ──────────────────────────────────────────────────────────────────────────────


def compute_cps_table(counts: dict) -> tuple[dict[tuple[str, str], float], pd.DataFrame]:
    """Compute the Coleman (2008) codon-pair-score table from accumulated counts.

    Returns (cps_dict, cps_df) where cps_dict maps (codon1, codon2) -> CPS and
    cps_df is the tidy table (codon1, codon2, aa1, aa2, observed, expected, cps).
    CPS is defined only for observed pairs (observed > 0), so every term in the
    expected count is strictly positive and the log is finite.
    """
    cc = counts["codon_counts"]
    ac = counts["aa_counts"]
    pc = counts["pair_counts"]
    apc = counts["aa_pair_counts"]

    cps: dict[tuple[str, str], float] = {}
    rows = []
    for (c1, c2), n_ab in pc.items():
        aa1 = CODON_TABLE_11[c1]
        aa2 = CODON_TABLE_11[c2]
        # observed > 0 guarantees N(c1), N(c2), N(aa1), N(aa2), N(aa1,aa2) > 0.
        expected = (cc[c1] * cc[c2] / (ac[aa1] * ac[aa2])) * apc[(aa1, aa2)]
        score = math.log(n_ab / expected)
        cps[(c1, c2)] = score
        rows.append({
            "codon1": c1,
            "codon2": c2,
            "aa1": aa1,
            "aa2": aa2,
            "observed": int(n_ab),
            "expected": round(float(expected), 4),
            "cps": round(float(score), 6),
        })

    if not rows:
        cps_df = pd.DataFrame(
            columns=["codon1", "codon2", "aa1", "aa2", "observed", "expected", "cps"]
        )
    else:
        cps_df = pd.DataFrame(rows).sort_values("cps", ascending=False).reset_index(drop=True)
    return cps, cps_df


# ──────────────────────────────────────────────────────────────────────────────
# Per-gene codon pair bias (CPB)
# ──────────────────────────────────────────────────────────────────────────────


def compute_gene_cpb(
    ffn_path: Path,
    cps_table: dict[tuple[str, str], float],
    min_length: int = MIN_GENE_LENGTH,
) -> pd.DataFrame:
    """Per-gene codon pair bias = mean CPS over a gene's consecutive sense pairs.

    On the in-genome path every pair is present in *cps_table* by construction
    (the table was built from the same CDS set), so ``n_scored == n_pairs`` and
    ``frac_scored == 1``. Those columns matter only when scoring sequences
    against a *foreign* host table (see ``score_sequence_cpb``).

    Returns DataFrame: gene, cpb, n_pairs, n_scored, frac_scored, length.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq)
        if len(seq) < min_length:
            continue
        codons = _ordered_codons(seq)
        scores = []
        n_pairs = 0
        for i in range(len(codons) - 1):
            c1, c2 = codons[i], codons[i + 1]
            aa1 = CODON_TABLE_11.get(c1)
            aa2 = CODON_TABLE_11.get(c2)
            if aa1 is None or aa1 == _STOP or aa2 is None or aa2 == _STOP:
                continue
            n_pairs += 1
            s = cps_table.get((c1, c2))
            if s is not None:
                scores.append(s)
        n_scored = len(scores)
        cpb = float(np.mean(scores)) if scores else np.nan
        rows.append({
            COL_GENE: rec.id,
            "cpb": round(cpb, 6) if not np.isnan(cpb) else np.nan,
            "n_pairs": n_pairs,
            "n_scored": n_scored,
            "frac_scored": round(n_scored / n_pairs, 4) if n_pairs else np.nan,
            "length": len(seq),
        })
    if not rows:
        return pd.DataFrame(
            columns=[COL_GENE, "cpb", "n_pairs", "n_scored", "frac_scored", "length"]
        )
    return pd.DataFrame(rows)


def score_sequence_cpb(
    seq: str,
    cps_table: dict[tuple[str, str], float],
    *,
    unobserved_penalty: float | None = None,
) -> dict:
    """Score a single (possibly foreign) sequence against a host CPS table.

    For genetic engineering: given a candidate CDS and the host genome's CPS
    table, returns the mean CPS (the sequence's CPB in the host's frame) plus a
    ``frac_scored`` diagnostic. Codon pairs absent from the host table are, by
    default, skipped and counted in ``n_unobserved`` (a high ``n_unobserved``
    means the candidate uses dicodons the host avoids -- exactly the pairs an
    optimizer should reconsider). Pass ``unobserved_penalty`` (a negative CPS,
    e.g. the host table's minimum) to score them instead of skipping.

    Returns: cpb, n_pairs, n_scored, n_unobserved, frac_scored.
    """
    codons = _ordered_codons(seq)
    scores = []
    n_pairs = 0
    n_unobserved = 0
    for i in range(len(codons) - 1):
        c1, c2 = codons[i], codons[i + 1]
        aa1 = CODON_TABLE_11.get(c1)
        aa2 = CODON_TABLE_11.get(c2)
        if aa1 is None or aa1 == _STOP or aa2 is None or aa2 == _STOP:
            continue
        n_pairs += 1
        s = cps_table.get((c1, c2))
        if s is not None:
            scores.append(s)
        else:
            n_unobserved += 1
            if unobserved_penalty is not None:
                scores.append(float(unobserved_penalty))
    cpb = float(np.mean(scores)) if scores else float("nan")
    return {
        "cpb": cpb,
        "n_pairs": n_pairs,
        "n_scored": len(scores),
        "n_unobserved": n_unobserved,
        "frac_scored": (len(scores) / n_pairs) if n_pairs else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bridge dinucleotide bias (the leading CPB confounder)
# ──────────────────────────────────────────────────────────────────────────────


def compute_bridge_dinucleotide_bias(pair_counts: Counter) -> pd.DataFrame:
    """Observed/expected ratios for the 16 codon-junction ("bridge") dinucleotides.

    The bridge dinucleotide of a codon pair (c1, c2) is (c1[2], c2[0]) -- the
    3' base of the first codon and the 5' base of the second. Karlin's rho
    (Karlin & Burge 1995, TIG 11:283) restricted to the junction:

        rho(XY) = f(XY) / ( f(X) * f(Y) )

    where f(XY) is the junction-dinucleotide frequency and f(X), f(Y) are the
    marginal base frequencies at the two junction positions. rho << 1 means the
    junction dinucleotide is under-represented (e.g. CpG, UpA in many genomes);
    rho >> 1 over-represented. This is reported alongside CPB because junction
    dinucleotide composition (esp. CpG/UpA) is the leading non-translational
    explanation for codon pair bias (Tulloch 2014; Kunec & Osterrieder 2016).
    Computed from the same junction set used for CPB, so the two are directly
    comparable.

    Returns DataFrame: dinucleotide, observed, obs_freq, exp_freq, rho,
    is_CpG, is_UpA.
    """
    bridge: Counter = Counter()
    pos1: Counter = Counter()
    pos2: Counter = Counter()
    total = 0
    for (c1, c2), n in pair_counts.items():
        x, y = c1[2], c2[0]
        bridge[(x, y)] += n
        pos1[x] += n
        pos2[y] += n
        total += n

    rows = []
    if total > 0:
        for x in "ACGU":
            for y in "ACGU":
                obs = bridge.get((x, y), 0)
                f_obs = obs / total
                f_exp = (pos1.get(x, 0) / total) * (pos2.get(y, 0) / total)
                rho = (f_obs / f_exp) if f_exp > 0 else np.nan
                rows.append({
                    "dinucleotide": x + y,
                    "observed": int(obs),
                    "obs_freq": round(f_obs, 6),
                    "exp_freq": round(f_exp, 6),
                    "rho": round(float(rho), 4) if not np.isnan(rho) else np.nan,
                    "is_CpG": (x == "C" and y == "G"),
                    "is_UpA": (x == "U" and y == "A"),
                })
    return pd.DataFrame(
        rows,
        columns=["dinucleotide", "observed", "obs_freq", "exp_freq", "rho",
                 "is_CpG", "is_UpA"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Small local statistics helpers (kept local to avoid cross-module coupling;
# identical in definition to the pipeline-wide versions)
# ──────────────────────────────────────────────────────────────────────────────


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta = (#(x>y) - #(x<y)) / (nx*ny), range [-1, 1]."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.sum() / (nx * ny))


def _expression_class_column(expr_df: pd.DataFrame) -> str | None:
    for col in ("expression_class", "CAI_class", "MELP_class", "Fop_class"):
        if col in expr_df.columns:
            return col
    return None


def compare_cpb_by_expression_tier(
    gene_cpb: pd.DataFrame,
    expr_df: pd.DataFrame,
    min_per_group: int = 5,
) -> pd.DataFrame:
    """High- vs low-expression CPB comparison (Mann-Whitney U + Cliff's delta).

    Tests the translational-selection hypothesis: if CPB is shaped by
    translational selection, highly expressed genes should carry a higher mean
    CPB than lowly expressed ones. Reported with the same MWU + Cliff's-delta
    machinery as the rest of the pipeline and with the standing CPB caveat: a
    positive result is consistent with, but not proof of, translational
    selection (see module docstring and the bridge-dinucleotide table).

    Returns a one-row DataFrame (or empty if the tiers are too small).
    """
    from scipy import stats as sp_stats

    cls_col = _expression_class_column(expr_df)
    if cls_col is None or COL_GENE not in expr_df.columns:
        return pd.DataFrame()
    merged = gene_cpb.merge(expr_df[[COL_GENE, cls_col]], on=COL_GENE, how="inner")
    merged = merged.dropna(subset=["cpb"])
    high = merged.loc[merged[cls_col] == "high", "cpb"].values
    low = merged.loc[merged[cls_col] == "low", "cpb"].values
    if len(high) < min_per_group or len(low) < min_per_group:
        return pd.DataFrame()
    try:
        u_stat, p_val = sp_stats.mannwhitneyu(high, low, alternative="two-sided")
    except ValueError:
        return pd.DataFrame()
    delta = _cliffs_delta(high, low)
    return pd.DataFrame([{
        "tier_column": cls_col,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "median_cpb_high": round(float(np.median(high)), 6),
        "median_cpb_low": round(float(np.median(low)), 6),
        "U_statistic": round(float(u_stat), 2),
        "p_value": float(p_val),
        "cliffs_delta": round(float(delta), 4),
        "direction": "high>low" if np.median(high) > np.median(low) else "high<=low",
        "caveat": "descriptive; CPB cause is debated (see bridge-dinucleotide table)",
    }])


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def run_codon_pair_analysis(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    *,
    expr_df: pd.DataFrame | None = None,
    min_length: int = MIN_GENE_LENGTH,
    make_figures: bool = True,
) -> dict[str, Path]:
    """Run the full codon-pair-bias analysis for one genome.

    Emits to ``<output_dir>/codon_usage/codon_pair/``:
      <sid>_cps_table.tsv             reference codon-pair-score table (engineering input)
      <sid>_gene_cpb.tsv              per-gene codon pair bias
      <sid>_bridge_dinucleotide.tsv   junction-dinucleotide O/E (CPB confounder)
      <sid>_codon_pair_summary.tsv    genome-level scalars
      <sid>_cpb_expression_tiers.tsv  high-vs-low CPB test (only if expr tiers given)
      <sid>_codon_pair.png/.svg       summary figure (if make_figures)
    """
    out_dir = get_output_subdir(output_dir, "codon_usage", "codon_pair")
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    logger.info("Codon pair bias: counting dicodons for %s", sample_id)
    counts = count_codon_pairs(ffn_path, min_length=min_length)
    if counts["n_pairs"] == 0:
        logger.warning("Codon pair bias: no scorable codon pairs for %s; skipping", sample_id)
        return outputs

    cps_table, cps_df = compute_cps_table(counts)
    cps_path = out_dir / f"{sample_id}_cps_table.tsv"
    cps_df.to_csv(cps_path, sep="\t", index=False)
    outputs["cps_table"] = cps_path

    gene_cpb = compute_gene_cpb(ffn_path, cps_table, min_length=min_length)
    gene_path = out_dir / f"{sample_id}_gene_cpb.tsv"
    gene_cpb.to_csv(gene_path, sep="\t", index=False)
    outputs["gene_cpb"] = gene_path

    bridge = compute_bridge_dinucleotide_bias(counts["pair_counts"])
    bridge_path = out_dir / f"{sample_id}_bridge_dinucleotide.tsv"
    bridge.to_csv(bridge_path, sep="\t", index=False)
    outputs["bridge_dinucleotide"] = bridge_path

    # Genome-level summary scalars.
    # Frequency-weighted overall CPS = sum(N(AB)*CPS) / sum(N(AB)).
    weighted_num = sum(counts["pair_counts"][k] * cps_table[k] for k in cps_table)
    weighted_cpb = weighted_num / counts["n_pairs"] if counts["n_pairs"] else np.nan
    gene_cpb_valid = gene_cpb["cpb"].dropna()
    cpg_rho = bridge.loc[bridge["is_CpG"], "rho"]
    upa_rho = bridge.loc[bridge["is_UpA"], "rho"]
    summary = {
        "sample_id": sample_id,
        "n_genes": counts["n_genes"],
        "n_codon_pairs_total": counts["n_pairs"],
        "n_distinct_pairs_observed": len(cps_table),
        "n_possible_sense_pairs": _N_POSSIBLE_PAIRS,
        "frac_pairs_observed": round(len(cps_table) / _N_POSSIBLE_PAIRS, 4),
        "mean_gene_cpb": round(float(gene_cpb_valid.mean()), 6) if not gene_cpb_valid.empty else np.nan,
        "median_gene_cpb": round(float(gene_cpb_valid.median()), 6) if not gene_cpb_valid.empty else np.nan,
        "weighted_overall_cps": round(float(weighted_cpb), 6) if not np.isnan(weighted_cpb) else np.nan,
        "bridge_CpG_rho": round(float(cpg_rho.iloc[0]), 4) if not cpg_rho.empty and not np.isnan(cpg_rho.iloc[0]) else np.nan,
        "bridge_UpA_rho": round(float(upa_rho.iloc[0]), 4) if not upa_rho.empty and not np.isnan(upa_rho.iloc[0]) else np.nan,
    }
    summary_path = out_dir / f"{sample_id}_codon_pair_summary.tsv"
    pd.DataFrame([summary]).to_csv(summary_path, sep="\t", index=False)
    outputs["summary"] = summary_path
    logger.info(
        "Codon pair bias for %s: mean gene CPB=%.4f, %d/%d distinct pairs observed, "
        "bridge CpG rho=%s, UpA rho=%s",
        sample_id, summary["mean_gene_cpb"] if summary["mean_gene_cpb"] == summary["mean_gene_cpb"] else float("nan"),
        summary["n_distinct_pairs_observed"], _N_POSSIBLE_PAIRS,
        summary["bridge_CpG_rho"], summary["bridge_UpA_rho"],
    )

    # Optional expression-tier comparison.
    if expr_df is not None and not expr_df.empty:
        tiers = compare_cpb_by_expression_tier(gene_cpb, expr_df)
        if not tiers.empty:
            tiers_path = out_dir / f"{sample_id}_cpb_expression_tiers.tsv"
            tiers.to_csv(tiers_path, sep="\t", index=False)
            outputs["expression_tiers"] = tiers_path
            row = tiers.iloc[0]
            logger.info(
                "CPB vs expression (%s): high n=%d (median %.4f) vs low n=%d (median %.4f); "
                "Cliff's delta=%.3f, p=%.2e [descriptive]",
                row["tier_column"], row["n_high"], row["median_cpb_high"],
                row["n_low"], row["median_cpb_low"], row["cliffs_delta"], row["p_value"],
            )

    if make_figures:
        try:
            png, svg = _render_codon_pair_figure(
                out_dir, sample_id, gene_cpb, cps_df, bridge, expr_df,
            )
            if png is not None:
                outputs["figure_png"] = png
                outputs["figure_svg"] = svg
        except Exception as e:  # pragma: no cover - plotting is best-effort
            logger.warning("Codon pair figure failed: %s", e)

    return outputs


def _render_codon_pair_figure(
    out_dir: Path,
    sample_id: str,
    gene_cpb: pd.DataFrame,
    cps_df: pd.DataFrame,
    bridge: pd.DataFrame,
    expr_df: pd.DataFrame | None,
) -> tuple[Path | None, Path | None]:
    """Three-panel summary: per-gene CPB distribution, top |CPS| pairs, bridge rho."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cpb_vals = gene_cpb["cpb"].dropna().values
    if len(cpb_vals) < 5 or cps_df.empty or bridge.empty:
        return (None, None)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel A: per-gene CPB distribution, split by expression tier if available.
    axA = axes[0]
    plotted_tier = False
    if expr_df is not None and not expr_df.empty:
        cls_col = _expression_class_column(expr_df)
        if cls_col is not None and COL_GENE in expr_df.columns:
            m = gene_cpb.merge(expr_df[[COL_GENE, cls_col]], on=COL_GENE, how="inner").dropna(subset=["cpb"])
            for tier, color in (("low", "#4C72B0"), ("high", "#C44E52")):
                v = m.loc[m[cls_col] == tier, "cpb"].values
                if len(v) >= 5:
                    axA.hist(v, bins=30, alpha=0.55, label=f"{tier} (n={len(v)})", color=color)
                    plotted_tier = True
    if not plotted_tier:
        axA.hist(cpb_vals, bins=30, color="#55A868", alpha=0.8)
    axA.axvline(0.0, color="black", lw=0.8, ls="--")
    axA.set_xlabel("Per-gene codon pair bias (mean CPS)")
    axA.set_ylabel("Genes")
    axA.set_title("A. CPB distribution")
    if plotted_tier:
        axA.legend(fontsize=8)

    # Panel B: most over- and under-represented codon pairs by CPS.
    axB = axes[1]
    top = cps_df.nlargest(10, "cps")
    bot = cps_df.nsmallest(10, "cps")
    show = pd.concat([bot, top])
    labels = [f"{r.codon1}-{r.codon2}" for r in show.itertuples()]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in show["cps"]]
    ypos = np.arange(len(show))
    axB.barh(ypos, show["cps"].values, color=colors)
    axB.set_yticks(ypos)
    axB.set_yticklabels(labels, fontsize=6)
    axB.axvline(0.0, color="black", lw=0.8)
    axB.set_xlabel("Codon pair score (ln obs/exp)")
    axB.set_title("B. Most biased codon pairs")

    # Panel C: bridge dinucleotide rho, CpG/UpA highlighted.
    axC = axes[2]
    b = bridge.dropna(subset=["rho"]).copy()
    bar_colors = []
    for r in b.itertuples():
        if r.is_CpG:
            bar_colors.append("#C44E52")
        elif r.is_UpA:
            bar_colors.append("#DD8452")
        else:
            bar_colors.append("#8C8C8C")
    xpos = np.arange(len(b))
    axC.bar(xpos, b["rho"].values, color=bar_colors)
    axC.axhline(1.0, color="black", lw=0.8, ls="--")
    axC.set_xticks(xpos)
    axC.set_xticklabels(b["dinucleotide"].values, fontsize=7, rotation=90)
    axC.set_ylabel("Bridge dinucleotide rho (obs/exp)")
    axC.set_title("C. Junction dinucleotide bias\n(CPB confounder; CpG red, UpA orange)")

    fig.suptitle(f"Codon pair bias — {sample_id}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png = out_dir / f"{sample_id}_codon_pair.png"
    svg = out_dir / f"{sample_id}_codon_pair.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return (png, svg)
