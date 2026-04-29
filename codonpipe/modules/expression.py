"""Module for predicting gene expression levels using codon adaptation metrics.

Uses the R package coRdon (via subprocess) for MELP, CAI, and Fop calculations,
with ribosomal proteins (or, downstream, the Mahalanobis cluster) as the
reference set for highly expressed genes.

What each metric measures, and what to read into a 0
====================================================

  * **CAI** (Sharp & Li 1987): geometric mean of relative adaptiveness
    ``w_i = RSCU_i / RSCU_max`` over the codons in a gene, where
    ``RSCU_max`` is taken from the reference set. Bounded in (0, 1].
    A neutral gene has CAI ~ 0.5; the reference set itself sits near 1.

  * **Fop** (Ikemura 1981): fraction of "optimal" codons in a gene, where
    "optimal" = most-used codon for each amino acid in the reference.
    Bounded in [0, 1]. Sensitive only to the most-frequent codon per AA;
    insensitive to graded preferences.

  * **MELP** (Supek & Smuc 2010): MILC ratio of self vs reference,
    interpreted as expression-likeness above the genome baseline.
    coRdon's implementation is **non-negative by construction**: genes
    whose codon usage is closer to the genome than to the reference get
    MELP = 0 exactly. This is the published convention, not a bug. In a
    typical bacterial genome ~30-50% of genes will land at MELP = 0.
    Treat MELP_class "low" as "below genome baseline", not as "actively
    avoided codon usage".

When the three look identical
-----------------------------

If MELP, CAI, and Fop agree on every gene, look at the reference set,
not the metrics. Each metric measures a different mathematical aspect of
codon usage difference (chi-squared / geometric mean / mode count), so
they only collapse into a single signal when the reference is too uniform
relative to the genome's variation. Common causes:

  * Mahalanobis cluster dominated by RPs → mahal-referenced ≈ rp-referenced
  * Strong translational selection saturating all three measures
  * Small reference set (n_RP < 30) producing noisy w_i estimates that
    propagate identically through CAI, Fop, and MELP

The pipeline emits a Pearson-correlation diagnostic in the log when any
pair of metrics within the same reference frame exceeds r = 0.95 across
genes. See :func:`pipeline._log_metric_correlations`.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe.utils.codon_tables import (
    MIN_GENE_LENGTH,
    COL_GENE, COL_WIDTH, COL_SAMPLE_ID,
    COL_MELP, COL_CAI, COL_FOP, EXPRESSION_METRICS,
    COL_MELP_CLASS, COL_CAI_CLASS, COL_FOP_CLASS, COL_EXPRESSION_CLASS,
)
from codonpipe.utils.io import check_tool, get_output_subdir, run_cmd

logger = logging.getLogger("codonpipe")


_EXPRESSION_R_TEMPLATE = r"""
library(coRdon)
library(Biostrings)
library(IRanges)

args <- commandArgs(trailingOnly = TRUE)
fasta_file <- args[1]
rp_ids_file <- args[2]
output_file <- args[3]

# Read ribosomal protein IDs (or Mahal-cluster IDs — depends on the caller).
# These are the gene IDs that anchor the "highly expressed" reference set
# for the metric.
rp_ids <- readLines(rp_ids_file)
rp_ids <- rp_ids[nchar(rp_ids) > 0]
rp <- list(rp = rp_ids)

tryCatch({
    fasta <- readSet(file = fasta_file)
    # Strip FASTA headers to first word so IDs match rp_ids
    names(fasta) <- sub(" .*", "", names(fasta))
    # Build the codon table with NCBI translation table 11 (bacterial /
    # archaeal / plastid standard code). Previously id_or_name2 was passed
    # to MELP/CAI/Fop — those functions accept it via ... but the codon
    # table itself was being built with the default universal genetic
    # code, which silently mis-handled bacterial start codons in some
    # corner cases. Setting it on codonTable() keeps the codon decoding
    # consistent throughout the pipeline.
    codons <- codonTable(fasta)
    # codonTable does not always honour the genetic code argument across
    # coRdon versions. Re-set the genetic code slot explicitly when the
    # object exposes it; otherwise rely on codonTable's default and let
    # the metric calls pass id_or_name2 = "11" via ... .
    if ("genetic.code" %in% slotNames(codons)) {
        codons@genetic.code <- "11"
    }
    codons@KO <- codons@ID

    # Pass id_or_name2 = "11" defensively — coRdon's metric implementations
    # also re-derive the codon-AA map internally and use this argument when
    # the cTobject's genetic.code slot is unset.
    scores <- __METRIC__(codons, filtering = "none", subsets = rp,
                        id_or_name2 = "11")
    scores_df <- as.data.frame(scores)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(scores_df, names_df, width_df)
    result <- subset(result, width > __MIN_LEN__)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("__METRIC__ analysis complete:", nrow(result), "genes\n")
}, error = function(e) {
    message("ERROR: ", e$message)
    quit(status = 1)
})
"""


def _expression_r_script(metric_func: str) -> str:
    """Build an R script for a coRdon expression metric.

    The template is shared across MELP, CAI, and Fop; only the function
    call differs.  MIN_GENE_LENGTH is injected so the R-side filter stays
    in sync with the Python constant.

    Args:
        metric_func: coRdon function name — "MELP", "CAI", or "Fop".
    """
    return (
        _EXPRESSION_R_TEMPLATE
        .replace("__METRIC__", metric_func)
        .replace("__MIN_LEN__", str(MIN_GENE_LENGTH))
    )


def run_expression_analysis(
    ffn_path: Path,
    rp_ids_file: Path,
    output_dir: Path,
    sample_id: str,
    force: bool = False,
) -> tuple[dict[str, Path], pd.DataFrame | None]:
    """Run MELP, CAI, and Fop expression level prediction.

    Args:
        ffn_path: Path to all CDS nucleotide sequences.
        rp_ids_file: Path to text file with ribosomal protein IDs (one per line).
        output_dir: Base output directory.
        sample_id: Sample identifier.
        force: Accepted for API compatibility; ignored. Outputs are written
            to a fresh temporary directory on every call, so there is never
            a prior result to skip.

    Returns:
        Tuple of (outputs_dict, combined_dataframe) where outputs_dict contains
        file paths and combined_dataframe is the combined expression table
        (or None if skipped/failed).
    """
    check_tool("Rscript")
    expr_dir = get_output_subdir(output_dir, "expression", "scores")
    expr_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Check that ribosomal protein IDs file is non-empty
    rp_ids = [l.strip() for l in rp_ids_file.read_text().splitlines() if l.strip()]
    if not rp_ids:
        logger.warning(
            "No ribosomal protein IDs for %s; skipping expression analysis "
            "(MELP, CAI, Fop will be absent from output). "
            "This genome may lack ribosomal protein annotations.", sample_id,
        )
        outputs["_skipped"] = "no_rp_ids"
        return outputs

    # Use temporary directory for per-metric coRdon outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # MELP, CAI, Fop are independent R subprocesses with the same
        # input FASTA and reference file but disjoint output paths.
        # Each Rscript call takes ~the same time, so launching them in
        # parallel cuts the wall clock to roughly the time of a single
        # call. Threading is the right primitive here — the work is in
        # Rscript (an external OS process) and Python is just waiting
        # on subprocess.run; threads avoid the pickling cost of process
        # workers and there's nothing for them to fight over (each
        # writes to its own file in the tempdir).
        outputs_dict: dict[str, Path] = {}
        # ``force`` is retained for API compatibility but has no effect
        # here — every invocation writes into a fresh TemporaryDirectory,
        # so there is never a prior output to skip.
        _ = force
        metric_jobs = [("MELP", "melp"), ("CAI", "cai"), ("Fop", "fop")]
        for metric, outname in metric_jobs:
            outputs_dict[outname] = tmpdir / f"{sample_id}_{outname}.tsv"

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=len(metric_jobs)) as pool:
            futures = {
                pool.submit(
                    _run_r_expression,
                    _expression_r_script(metric),
                    ffn_path, rp_ids_file,
                    outputs_dict[outname], metric, sample_id,
                ): (metric, outname)
                for metric, outname in metric_jobs
            }
            logger.info(
                "Computing %s expression scores for %s in parallel",
                ", ".join(m for m, _ in metric_jobs), sample_id,
            )
            for future in as_completed(futures):
                metric, _outname = futures[future]
                # Surface worker exceptions on the main thread.
                future.result()
                logger.info("Completed %s for %s", metric, sample_id)

        melp_out = outputs_dict.get("melp", tmpdir / f"{sample_id}_melp.tsv")
        cai_out = outputs_dict.get("cai", tmpdir / f"{sample_id}_cai.tsv")
        fop_out = outputs_dict.get("fop", tmpdir / f"{sample_id}_fop.tsv")

        # Combine and classify expression levels
        combined_df = None
        if melp_out.exists() and cai_out.exists():
            combined = _combine_expression(melp_out, cai_out, fop_out, sample_id)
            combined_out = expr_dir / f"{sample_id}_expression.tsv"
            combined.to_csv(combined_out, sep="\t", index=False)
            outputs["expression_combined"] = combined_out
            combined_df = combined
            logger.info(
                "Expression analysis: %d genes classified for %s",
                len(combined), sample_id,
            )

            # Diagnostic: check for the MELP-floor and metric-collinearity
            # pathologies. The pipeline-level _log_metric_correlations is
            # the canonical caller, but we also log here so the warnings
            # are visible even when run_expression_analysis is invoked
            # directly (e.g. from tests).
            _log_local_diagnostics(combined, sample_id)

    return (outputs, combined_df)


def _run_r_expression(
    script_template: str,
    ffn_path: Path,
    rp_ids_file: Path,
    output_file: Path,
    method_name: str,
    sample_id: str,
) -> None:
    """Execute an R expression analysis script."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as tmp:
        tmp.write(script_template)
        r_script_path = Path(tmp.name)

    try:
        result = run_cmd(
            ["Rscript", str(r_script_path), str(ffn_path), str(rp_ids_file), str(output_file)],
            description=f"Running {method_name} for {sample_id}",
            capture=True,
            timeout=3600,
        )
        if not output_file.exists():
            raise RuntimeError(
                f"{method_name} R script did not produce output for {sample_id}. "
                f"stderr: {result.stderr[:500] if result.stderr else '(none)'}"
            )
    finally:
        r_script_path.unlink(missing_ok=True)


def _log_local_diagnostics(combined: pd.DataFrame, sample_id: str) -> None:
    """Emit warnings for the two well-known failure modes:
    (1) >40% of MELP values clipped to 0 (Supek & Smuc 2010 floor),
    (2) MELP/CAI/Fop carrying near-identical signal across genes
        (Pearson r > 0.95 within a single reference frame).

    Duplicates the diagnostic in :func:`pipeline._log_metric_correlations`
    so the warning fires even when this module is invoked outside the
    full pipeline.
    """
    if combined.empty:
        return

    # MELP floor diagnostic
    if COL_MELP in combined.columns:
        melp_vals = pd.to_numeric(combined[COL_MELP], errors="coerce").dropna()
        if len(melp_vals) >= 20:
            frac_zero = float((melp_vals == 0).mean())
            if frac_zero > 0.4:
                logger.warning(
                    "[%s] %.1f%% of MELP values are exactly 0 (coRdon's "
                    "non-negative MELP convention; Supek & Smuc 2010). "
                    "MELP_class 'low' will be ~the bottom %.0f%% rather "
                    "than the bottom 10%%; the 10/90 percentile bins "
                    "lose meaning when most of the lower tail is "
                    "saturated. Use CAI_class or Fop_class for "
                    "low-expression discrimination.",
                    sample_id, 100 * frac_zero, 100 * frac_zero,
                )

    # Pairwise correlation diagnostic
    metrics = [c for c in (COL_MELP, COL_CAI, COL_FOP) if c in combined.columns]
    if len(metrics) >= 2:
        sub = combined[metrics].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) >= 10:
            high = []
            for i, m1 in enumerate(metrics):
                for m2 in metrics[i + 1:]:
                    if sub[m1].std() == 0 or sub[m2].std() == 0:
                        continue
                    r = float(sub[m1].corr(sub[m2]))
                    if r >= 0.95:
                        high.append((m1, m2, r))
            if high:
                logger.warning(
                    "[%s] MELP/CAI/Fop are nearly redundant in this "
                    "reference frame: %s. The three metrics are giving "
                    "essentially the same gene ranking — treat as one "
                    "signal. This typically means the reference set "
                    "(RP or Mahal cluster) is too uniform relative to "
                    "the genome's codon-usage variation.",
                    sample_id,
                    ", ".join(f"{a}~{b} r={r:.3f}" for a, b, r in high),
                )


def _classify_by_percentile(
    series: pd.Series,
    low_pctl: float = 10.0,
    high_pctl: float = 90.0,
) -> pd.Series:
    """Classify a numeric series into high/medium/low using quantile thresholds.

    Default cutoffs are the 10th and 90th percentiles, so each tail captures
    ~10% of genes — a working compromise between statistical power for
    downstream pathway enrichment (hypergeometric tests need enough genes per
    tier) and biological stringency (genes in the tail should be plausibly
    high- or low-expression). Pass low_pctl=5.0, high_pctl=95.0 for a stricter
    Sharp-style split.

    Quantile-based cutoffs avoid the problem where mean ± 1 SD produces wildly
    different tier sizes depending on score distribution shape (bimodal CAI,
    skewed MELP, etc.).

    Args:
        series: Score values (NaN-safe).
        low_pctl: Percentile for the low threshold (default 10).
        high_pctl: Percentile for the high threshold (default 90).

    Returns:
        Series of 'high', 'medium', 'low', or 'unknown' labels.
    """
    if series.notna().sum() == 0:
        return pd.Series("unknown", index=series.index)

    vals = series.dropna()
    hi_thresh = np.nanpercentile(vals, high_pctl)
    lo_thresh = np.nanpercentile(vals, low_pctl)

    # Build labels row-by-row so that NaN scores stay "unknown" instead of
    # being collapsed into "medium" (NaN >= x and NaN <= x are both False,
    # so a naive np.where chain misclassifies missing values).
    result = pd.Series("medium", index=series.index, dtype=object)
    result[series.isna()] = "unknown"
    result[series >= hi_thresh] = "high"
    result[series <= lo_thresh] = "low"

    # Degenerate-bin guard. coRdon's MELP saturates at 0 for genes below
    # the genome baseline (Supek & Smuc 2010 convention), so MELP commonly
    # has a large mass at exactly 0. When that mass exceeds 10% of the
    # distribution, np.percentile(vals, 10) is also 0, and the rule
    # ``series <= 0 → "low"`` swallows EVERY zero gene into "low" — a tier
    # that should be ~10% of the genome instead becomes ~50%. Detect this
    # and tighten the rule so "low" only includes genes strictly below the
    # next non-floor value, while still capturing the bottom 10%.
    if lo_thresh == hi_thresh:
        # Distribution is degenerate (all values equal). Everything is
        # "medium" — give up on tiers rather than report meaningless ones.
        result[~series.isna()] = "medium"
        logger.warning(
            "Percentile classification: low and high thresholds are "
            "identical (%.4f). All non-null values labelled 'medium'.",
            float(lo_thresh),
        )
    else:
        floor_mass = float((vals == lo_thresh).mean())
        if floor_mass > 0.10 and lo_thresh == np.nanmin(vals):
            # The "low" boundary sits on a saturated floor (very common for
            # MELP's clipped-at-0 distribution). Reassign genes at the
            # floor to "low" only up to the original 10% target by ranking
            # within the floor mass; the remainder go to "medium".
            # Concretely: keep the strict-equality "low" assignment for
            # genes at the floor up to the original low_pctl quota; mark
            # the overflow as a separate "low_floor" sub-tier so
            # downstream consumers can choose how to handle them.
            target_low_count = int(np.ceil(len(vals) * (low_pctl / 100.0)))
            floor_mask = (series == lo_thresh) & series.notna()
            n_floor = int(floor_mask.sum())
            if n_floor > target_low_count:
                # Keep `target_low_count` of the floor genes labelled
                # "low"; relabel the overflow with a new "low_floor"
                # tier so users can see the saturation explicitly.
                # Choose deterministically by index order to avoid
                # randomness biasing downstream tests.
                floor_indices = list(series.index[floor_mask])
                low_keep = set(floor_indices[:target_low_count])
                for idx in floor_indices:
                    if idx not in low_keep:
                        result.at[idx] = "low_floor"
                logger.warning(
                    "Percentile classification: %.1f%% of values sit at "
                    "the distribution floor (%.4f); %d genes assigned to "
                    "'low' and %d to 'low_floor' to preserve the 10%% "
                    "low-tier target while flagging saturation. Treat "
                    "'low_floor' the same as 'low' for most purposes; "
                    "the distinct label preserves the diagnostic.",
                    100 * floor_mass, float(lo_thresh),
                    target_low_count, n_floor - target_low_count,
                )
    return result


def _rename_score_column(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Rename the single numeric score column in a coRdon output DataFrame.

    coRdon produces a TSV with columns: a numeric score, 'gene', and 'width'.
    This function finds the first numeric column that isn't 'gene' or 'width'
    and renames it to *target_name*.  If no suitable column is found, the
    DataFrame is returned unchanged and a warning is logged.
    """
    meta_cols = {COL_GENE, COL_WIDTH, COL_SAMPLE_ID, "KO"}
    for col in df.columns:
        if col in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return df.rename(columns={col: target_name})
    logger.warning(
        "No numeric score column found in coRdon output (columns: %s); "
        "expected a column to rename to '%s'",
        list(df.columns), target_name,
    )
    return df


def _combine_expression(
    melp_path: Path, cai_path: Path, fop_path: Path, sample_id: str,
) -> pd.DataFrame:
    """Combine MELP, CAI, and Fop results and classify expression levels.

    Each metric gets its own classification column (MELP_class, CAI_class,
    Fop_class) using the same percentile scheme:
        - high:   >= 90th percentile of that metric
        - low:    <= 10th percentile of that metric
        - medium: everything else
    See _classify_by_percentile for the rationale behind 10/90.

    The ``expression_class`` column defaults to MELP_class (MELP outperforms
    CAI in high-GC organisms). Falls back to CAI_class if MELP is unavailable.
    """
    melp_df = pd.read_csv(melp_path, sep="\t")
    cai_df = pd.read_csv(cai_path, sep="\t")

    # Rename the score column.  coRdon outputs a single numeric column
    # alongside 'gene' and 'width'; identify it by excluding those two.
    melp_df = _rename_score_column(melp_df, COL_MELP)
    cai_df = _rename_score_column(cai_df, COL_CAI)

    # Merge MELP and CAI on gene
    combined = melp_df[[COL_GENE, COL_WIDTH, COL_MELP]].merge(
        cai_df[[COL_GENE, COL_CAI]], on=COL_GENE, how="outer"
    )

    # Merge Fop if available
    if fop_path.exists():
        fop_df = pd.read_csv(fop_path, sep="\t")
        fop_df = _rename_score_column(fop_df, COL_FOP)
        if COL_FOP in fop_df.columns:
            combined = combined.merge(fop_df[[COL_GENE, COL_FOP]], on=COL_GENE, how="outer")
        else:
            logger.warning("Fop output has no usable score column; skipping Fop merge")

    # Fill width from whichever metric provided it (outer merge can leave NaN).
    # Drop genes with missing width rather than substituting 0 (which is
    # biologically nonsensical and causes downstream division errors).
    if COL_WIDTH in combined.columns:
        n_missing = combined[COL_WIDTH].isna().sum()
        if n_missing > 0:
            logger.warning(
                "%d genes have missing width after merge; dropping them", n_missing
            )
            combined = combined.dropna(subset=[COL_WIDTH])
        combined[COL_WIDTH] = combined[COL_WIDTH].astype(int)

    # Per-metric classification
    for metric in EXPRESSION_METRICS:
        if metric in combined.columns:
            combined[f"{metric}_class"] = _classify_by_percentile(combined[metric])

    # Primary expression class: prefer MELP (more robust in high-GC organisms),
    # fall back to CAI if MELP unavailable
    if COL_MELP_CLASS in combined.columns:
        combined[COL_EXPRESSION_CLASS] = combined[COL_MELP_CLASS]
    elif COL_CAI_CLASS in combined.columns:
        combined[COL_EXPRESSION_CLASS] = combined[COL_CAI_CLASS]
    else:
        combined[COL_EXPRESSION_CLASS] = "unknown"

    # Outer merges can leave NaN in class columns for genes present in one
    # metric but not another; fill with "unknown" to avoid downstream surprises.
    for col in [c for c in combined.columns if c.endswith("_class")]:
        combined[col] = combined[col].fillna("unknown")

    combined[COL_SAMPLE_ID] = sample_id
    return combined
