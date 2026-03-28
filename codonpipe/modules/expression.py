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

from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")

# R script template for MELP calculation
_MELP_R_SCRIPT = r"""
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

tryCatch({{
    fasta <- readSet(file = fasta_file)
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    melp <- MELP(codons, filtering = "none", subsets = rp, id_or_name2 = "11")
    melp_df <- as.data.frame(melp)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(melp_df, names_df, width_df)
    result <- subset(result, width > 240)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("MELP analysis complete:", nrow(result), "genes\n")
}}, error = function(e) {{
    message("ERROR: ", e$message)
    quit(status = 1)
}})
"""

# R script template for CAI calculation
_CAI_R_SCRIPT = r"""
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

tryCatch({{
    fasta <- readSet(file = fasta_file)
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    cai <- CAI(codons, filtering = "none", subsets = rp, id_or_name2 = "11")
    cai_df <- as.data.frame(cai)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(cai_df, names_df, width_df)
    result <- subset(result, width > 240)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("CAI analysis complete:", nrow(result), "genes\n")
}}, error = function(e) {{
    message("ERROR: ", e$message)
    quit(status = 1)
}})
"""

# R script template for Fop (Frequency of optimal codons) calculation
_FOP_R_SCRIPT = r"""
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

tryCatch({{
    fasta <- readSet(file = fasta_file)
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    fop <- Fop(codons, filtering = "none", subsets = rp, id_or_name2 = "11")
    fop_df <- as.data.frame(fop)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(fop_df, names_df, width_df)
    result <- subset(result, width > 240)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("Fop analysis complete:", nrow(result), "genes\n")
}}, error = function(e) {{
    message("ERROR: ", e$message)
    quit(status = 1)
}})
"""


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

    # MELP
    melp_out = expr_dir / f"{sample_id}_melp.tsv"
    if not melp_out.exists() or force:
        logger.info("Computing MELP expression scores for %s", sample_id)
        _run_r_expression(_MELP_R_SCRIPT, ffn_path, rp_ids_file, melp_out, "MELP", sample_id)
    outputs["melp"] = melp_out

    # CAI
    cai_out = expr_dir / f"{sample_id}_cai.tsv"
    if not cai_out.exists() or force:
        logger.info("Computing CAI expression scores for %s", sample_id)
        _run_r_expression(_CAI_R_SCRIPT, ffn_path, rp_ids_file, cai_out, "CAI", sample_id)
    outputs["cai"] = cai_out

    # Fop (Frequency of optimal codons)
    fop_out = expr_dir / f"{sample_id}_fop.tsv"
    if not fop_out.exists() or force:
        logger.info("Computing Fop (frequency of optimal codons) for %s", sample_id)
        _run_r_expression(_FOP_R_SCRIPT, ffn_path, rp_ids_file, fop_out, "Fop", sample_id)
    outputs["fop"] = fop_out

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
    low_pctl: float = 5.0,
    high_pctl: float = 95.0,
) -> pd.Series:
    """Classify a numeric series into high/medium/low by percentile thresholds.

    Args:
        series: Score values (NaN-safe).
        low_pctl: Percentile below which genes are 'low'.
        high_pctl: Percentile above which genes are 'high'.

    Returns:
        Series of 'high', 'medium', 'low', or 'unknown' labels.
    """
    if series.notna().sum() == 0:
        return pd.Series("unknown", index=series.index)
    p_lo = np.nanpercentile(series, low_pctl)
    p_hi = np.nanpercentile(series, high_pctl)
    return pd.Series(
        np.where(series >= p_hi, "high",
                 np.where(series <= p_lo, "low", "medium")),
        index=series.index,
    )


def _combine_expression(
    melp_path: Path, cai_path: Path, fop_path: Path, sample_id: str,
) -> pd.DataFrame:
    """Combine MELP, CAI, and Fop results and classify expression levels.

    Each metric gets its own classification column (MELP_class, CAI_class,
    Fop_class) using the same percentile scheme:
        - high: >= 95th percentile of that metric
        - low:  <= 5th percentile of that metric
        - medium: everything else

    The legacy ``expression_class`` column is retained (identical to CAI_class)
    for backward compatibility.
    """
    melp_df = pd.read_csv(melp_path, sep="\t")
    cai_df = pd.read_csv(cai_path, sep="\t")

    # Rename the score columns
    melp_score_col = [c for c in melp_df.columns if c not in ("gene", "width")][0]
    cai_score_col = [c for c in cai_df.columns if c not in ("gene", "width")][0]

    melp_df = melp_df.rename(columns={melp_score_col: "MELP"})
    cai_df = cai_df.rename(columns={cai_score_col: "CAI"})

    # Merge MELP and CAI on gene
    combined = melp_df[["gene", "width", "MELP"]].merge(
        cai_df[["gene", "CAI"]], on="gene", how="outer"
    )

    # Merge Fop if available
    if fop_path.exists():
        fop_df = pd.read_csv(fop_path, sep="\t")
        fop_score_col = [c for c in fop_df.columns if c not in ("gene", "width")][0]
        fop_df = fop_df.rename(columns={fop_score_col: "Fop"})
        combined = combined.merge(fop_df[["gene", "Fop"]], on="gene", how="outer")

    # Per-metric classification
    for metric in ["MELP", "CAI", "Fop"]:
        if metric in combined.columns:
            combined[f"{metric}_class"] = _classify_by_percentile(combined[metric])

    # Legacy column (matches CAI_class)
    combined["expression_class"] = combined.get("CAI_class", "unknown")

    combined["sample_id"] = sample_id
    return combined
