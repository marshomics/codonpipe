"""Module for R-based codon usage bias statistics via coRdon.

Computes per-gene CU bias measures that complement the native ENC implementation:
    - ENCprime (Novembre 2002): ENC corrected for background GC composition
    - MILC (Supek & Vlahovicek 2005): Measure Independent of Length and Composition
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

from codonpipe.utils.codon_tables import MIN_GENE_LENGTH
from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")


_CU_STATISTIC_R_TEMPLATE = r"""
library(coRdon)
library(Biostrings)
library(IRanges)

args <- commandArgs(trailingOnly = TRUE)
fasta_file <- args[1]
output_file <- args[2]
min_len <- as.integer(args[3])
if (is.na(min_len)) {
    message("ERROR: min_length argument is not a valid integer: ", args[3])
    quit(status = 1)
}

tryCatch({
    fasta <- readSet(file = fasta_file)
    # Strip FASTA headers to first word for consistent gene IDs
    names(fasta) <- sub(" .*", "", names(fasta))
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    scores <- __METRIC__(codons, id_or_name2 = "11")
    scores_df <- as.data.frame(scores)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(scores_df, names_df, width_df)
    result <- subset(result, width > min_len)

    if (nrow(result) == 0) {
        message("ERROR: No genes passed the minimum length filter (min_len=", min_len, ")")
        quit(status = 1)
    }

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("__METRIC__ analysis complete:", nrow(result), "genes\n")
}, error = function(e) {
    message("ERROR: ", e$message)
    quit(status = 1)
})
"""


def _cu_statistic_r_script(metric_func: str) -> str:
    """Build an R script for a coRdon CU bias statistic.

    The template is shared between ENCprime and MILC; only the function
    call differs.  Accepts a min_len argument from the command line.

    Args:
        metric_func: coRdon function name — "ENCprime" or "MILC".
    """
    _VALID_METRICS = {"ENCprime", "MILC"}
    if metric_func not in _VALID_METRICS:
        raise ValueError(f"Unknown CU metric {metric_func!r}; expected one of {_VALID_METRICS}")
    return _CU_STATISTIC_R_TEMPLATE.replace("__METRIC__", metric_func)


# ── Public API ───────────────────────────────────────────────────────────────

def run_cu_statistics(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    force: bool = False,
    min_length: int = MIN_GENE_LENGTH,
) -> dict[str, Path]:
    """Compute R-based codon usage bias statistics (ENCprime, MILC).

    These are genome-wide per-gene measures that do not require a reference set
    of highly expressed genes. They complement the native Python ENC calculation
    by correcting for sequence length and/or GC composition.

    Args:
        ffn_path: Path to nucleotide CDS FASTA (Prokka .ffn).
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        force: Rerun even if output exists.
        min_length: Minimum gene length in nucleotides to include (default 240).

    Returns:
        Dict of output file paths keyed by statistic name.
    """
    check_tool("Rscript")
    stats_dir = output_dir / "cu_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    for metric, outname in [("ENCprime", "encprime"), ("MILC", "milc")]:
        out_path = stats_dir / f"{sample_id}_{outname}.tsv"
        if not out_path.exists() or force:
            logger.info("Computing %s for %s", metric, sample_id)
            _run_r_statistic(
                _cu_statistic_r_script(metric),
                ffn_path, out_path, metric, sample_id,
                min_length=min_length,
            )
        outputs[outname] = out_path

    return outputs


def _run_r_statistic(
    script_template: str,
    ffn_path: Path,
    output_file: Path,
    method_name: str,
    sample_id: str,
    min_length: int = MIN_GENE_LENGTH,
) -> None:
    """Execute an R CU statistics script (no reference set needed)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as tmp:
        tmp.write(script_template)
        r_script_path = Path(tmp.name)

    try:
        result = run_cmd(
            ["Rscript", str(r_script_path), str(ffn_path), str(output_file), str(min_length)],
            description=f"Running {method_name} for {sample_id}",
            capture=True,
            timeout=3600,
        )
        if not output_file.exists():
            raise RuntimeError(
                f"{method_name} R script did not produce output for {sample_id}. "
                f"stderr: {result.stderr[:2000] if result.stderr else '(none)'}"
            )
    finally:
        r_script_path.unlink(missing_ok=True)
