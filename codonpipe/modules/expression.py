"""Module for predicting gene expression levels using codon adaptation metrics.

Uses the R package coRdon (via subprocess) for MELP, CAI, and Fop calculations,
with ribosomal proteins as the reference set for highly expressed genes.
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
from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")


_EXPRESSION_R_TEMPLATE = r"""
library(coRdon)
library(Biostrings)
library(IRanges)

args <- commandArgs(trailingOnly = TRUE)
fasta_file <- args[1]
rp_ids_file <- args[2]
output_file <- args[3]

# Read ribosomal protein IDs
rp_ids <- readLines(rp_ids_file)
rp_ids <- rp_ids[nchar(rp_ids) > 0]
rp <- list(rp = rp_ids)

tryCatch({
    fasta <- readSet(file = fasta_file)
    # Strip FASTA headers to first word so IDs match rp_ids
    names(fasta) <- sub(" .*", "", names(fasta))
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    scores <- __METRIC__(codons, filtering = "none", subsets = rp, id_or_name2 = "11")
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
) -> dict[str, Path]:
    """Run MELP, CAI, and Fop expression level prediction.

    Args:
        ffn_path: Path to all CDS nucleotide sequences.
        rp_ids_file: Path to text file with ribosomal protein IDs (one per line).
        output_dir: Base output directory.
        sample_id: Sample identifier.
        force: Rerun even if output exists.

    Returns:
        Dict of output file paths (melp, cai, fop, expression_combined).
    """
    check_tool("Rscript")
    expr_dir = output_dir / "expression"
    expr_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Check that ribosomal protein IDs file is non-empty
    rp_ids = [l.strip() for l in rp_ids_file.read_text().splitlines() if l.strip()]
    if not rp_ids:
        logger.warning(
            "No ribosomal protein IDs for %s; skipping expression analysis. "
            "This genome may lack ribosomal protein annotations.", sample_id,
        )
        return outputs

    # Run each metric using the shared R script template
    for metric, outname in [("MELP", "melp"), ("CAI", "cai"), ("Fop", "fop")]:
        out_path = expr_dir / f"{sample_id}_{outname}.tsv"
        if not out_path.exists() or force:
            logger.info("Computing %s expression scores for %s", metric, sample_id)
            _run_r_expression(
                _expression_r_script(metric),
                ffn_path, rp_ids_file, out_path, metric, sample_id,
            )
        outputs[outname] = out_path

    melp_out = outputs.get("melp", expr_dir / f"{sample_id}_melp.tsv")
    cai_out = outputs.get("cai", expr_dir / f"{sample_id}_cai.tsv")
    fop_out = outputs.get("fop", expr_dir / f"{sample_id}_fop.tsv")

    # Combine and classify expression levels
    combined_out = expr_dir / f"{sample_id}_expression.tsv"
    if melp_out.exists() and cai_out.exists():
        combined = _combine_expression(melp_out, cai_out, fop_out, sample_id)
        combined.to_csv(combined_out, sep="\t", index=False)
        outputs["expression_combined"] = combined_out
        logger.info(
            "Expression analysis: %d genes classified for %s",
            len(combined), sample_id,
        )

    return outputs


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


def _classify_by_percentile(
    series: pd.Series,
    low_pctl: float = 10.0,
    high_pctl: float = 90.0,
) -> pd.Series:
    """Classify a numeric series into high/medium/low using quantile thresholds.

    Uses fixed quantile cutoffs (default top/bottom 10%) for stable,
    distribution-independent tier boundaries. This avoids the problem where
    mean ± 1 SD produces wildly different tier sizes depending on score
    distribution shape (bimodal CAI, skewed MELP, etc.).

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

    return pd.Series(
        np.where(series >= hi_thresh, "high",
                 np.where(series <= lo_thresh, "low", "medium")),
        index=series.index,
    )


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
        - high: >= 90th percentile of that metric
        - low:  <= 10th percentile of that metric
        - medium: everything else

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

    # Fill width from whichever metric provided it (outer merge can leave NaN)
    if COL_WIDTH in combined.columns:
        combined[COL_WIDTH] = combined[COL_WIDTH].fillna(0).astype(int)

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
