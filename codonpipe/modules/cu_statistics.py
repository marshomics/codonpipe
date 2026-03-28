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

from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")


# ── R script templates ───────────────────────────────────────────────────────

_ENCPRIME_R_SCRIPT = r"""
library(coRdon)
library(Biostrings)
library(IRanges)

args <- commandArgs(trailingOnly = TRUE)
fasta_file <- args[1]
output_file <- args[2]

tryCatch({{
    fasta <- readSet(file = fasta_file)
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    enc_prime <- ENCprime(codons, id_or_name2 = "11")
    enc_prime_df <- as.data.frame(enc_prime)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(enc_prime_df, names_df, width_df)
    result <- subset(result, width > 240)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("ENCprime analysis complete:", nrow(result), "genes\n")
}}, error = function(e) {{
    cat("ERROR:", e$message, "\n")
    quit(status = 1)
}})
"""

_MILC_R_SCRIPT = r"""
library(coRdon)
library(Biostrings)
library(IRanges)

args <- commandArgs(trailingOnly = TRUE)
fasta_file <- args[1]
output_file <- args[2]

tryCatch({{
    fasta <- readSet(file = fasta_file)
    codons <- codonTable(fasta)
    codons@KO <- codons@ID

    milc <- MILC(codons, id_or_name2 = "11")
    milc_df <- as.data.frame(milc)

    names_df <- data.frame(gene = names(fasta))
    width_df <- data.frame(width = width(fasta))

    result <- cbind(milc_df, names_df, width_df)
    result <- subset(result, width > 240)

    write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
    cat("MILC analysis complete:", nrow(result), "genes\n")
}}, error = function(e) {{
    cat("ERROR:", e$message, "\n")
    quit(status = 1)
}})
"""


# ── Public API ───────────────────────────────────────────────────────────────

def run_cu_statistics(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    force: bool = False,
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

    Returns:
        Dict of output file paths keyed by statistic name.
    """
    check_tool("Rscript")
    stats_dir = output_dir / "cu_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # ENCprime
    encprime_out = stats_dir / f"{sample_id}_encprime.tsv"
    if not encprime_out.exists() or force:
        logger.info("Computing ENCprime (GC-corrected ENC) for %s", sample_id)
        _run_r_statistic(
            _ENCPRIME_R_SCRIPT, ffn_path, encprime_out, "ENCprime", sample_id,
        )
    outputs["encprime"] = encprime_out

    # MILC
    milc_out = stats_dir / f"{sample_id}_milc.tsv"
    if not milc_out.exists() or force:
        logger.info("Computing MILC for %s", sample_id)
        _run_r_statistic(
            _MILC_R_SCRIPT, ffn_path, milc_out, "MILC", sample_id,
        )
    outputs["milc"] = milc_out

    return outputs


def _run_r_statistic(
    script_template: str,
    ffn_path: Path,
    output_file: Path,
    method_name: str,
    sample_id: str,
) -> None:
    """Execute an R CU statistics script (no reference set needed)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as tmp:
        tmp.write(script_template)
        r_script_path = Path(tmp.name)

    try:
        result = run_cmd(
            ["Rscript", str(r_script_path), str(ffn_path), str(output_file)],
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
