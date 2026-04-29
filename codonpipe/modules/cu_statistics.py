"""Module for R-based codon usage bias statistics via coRdon.

Computes per-gene CU bias measures that complement the native ENC implementation:
    - ENCprime (Novembre 2002): ENC corrected for background GC composition
    - MILC (Supek & Vlahovicek 2005): Measure Independent of Length and Composition

Reference set / background notes (important for reproducibility):

    The coRdon library is invoked without an explicit ``subsets=`` argument
    in this wrapper, which uses the *whole-genome* codon usage as the
    expected/background distribution for both metrics. This is coRdon's
    default behaviour and matches the original definitions:

        * ENCprime expects a per-gene null derived from genome-wide GC
          composition (Novembre 2002 §2). Passing no subset means coRdon
          computes E_ij from the full ORFeome's codon counts.
        * MILC's M_a chi-squared term is taken against the genome-mean
          synonymous frequency for each amino acid (Supek & Vlahovicek
          2005 eq. 4) when no reference set is provided.

    If you need a different reference (e.g. ribosomal-protein-only baseline
    for ENCprime, as in some downstream comparisons), call coRdon directly
    with ``subsets = list(rp = rp_ids)``. This wrapper deliberately keeps
    the genome-wide background for two reasons: (1) the README and CodonPipe
    output schema treat ENCprime/MILC as gene-intrinsic measures so they
    must not depend on the RP set being annotated; (2) the RP-conditioned
    measures live in the expression module (MELP, CAI, Fop) which already
    pass ``subsets = list(rp = rp_ids)`` explicitly.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe.utils.codon_tables import MIN_GENE_LENGTH
from codonpipe.utils.io import check_tool, get_output_subdir, run_cmd

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
    stats_dir = get_output_subdir(output_dir, "codon_usage", "cu_statistics")
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
        # Sanity-check the coRdon output. ENCprime should sit roughly in
        # [20, 61] (Wright-style bounds; small over-shoots are possible
        # under shrinkage). MILC is positive; values much larger than ~5
        # indicate either a numerical pathology or an unexpected reference
        # set. We log warnings — not errors — so a single off-bound gene
        # doesn't kill the run, but reproducibility issues are visible.
        _validate_cu_statistic_output(out_path, metric)

    return outputs


_VALID_RANGES = {
    "ENCprime": (20.0, 61.5),  # small over-bound tolerance for shrinkage
    "MILC": (0.0, 5.0),        # MILC is unbounded above but >5 is unusual
}


def _validate_cu_statistic_output(out_path: Path, metric: str) -> None:
    """Range-check the CU statistic table emitted by coRdon.

    Logs warnings for NaN/Inf and out-of-range values. Does not raise,
    so users can still inspect the table even when coRdon misbehaves
    (e.g. version drift). Pin coRdon in environment.yml to keep results
    stable across machines.
    """
    try:
        df = pd.read_csv(out_path, sep="\t")
    except Exception as exc:
        logger.warning("Could not read %s output for validation (%s)", metric, exc)
        return

    # The first non-(gene,width) column holds the metric values.
    candidate_cols = [c for c in df.columns if c not in ("gene", "width")]
    if not candidate_cols:
        logger.warning("%s output at %s has no metric column to validate.",
                       metric, out_path)
        return
    metric_col = candidate_cols[0]
    values = pd.to_numeric(df[metric_col], errors="coerce")

    n_nan = int(values.isna().sum())
    n_inf = int(np.isinf(values.fillna(0).values).sum())
    if n_nan or n_inf:
        logger.warning(
            "%s output: %d NaN and %d Inf values out of %d rows in column %r. "
            "Check coRdon installation and gene-length filter.",
            metric, n_nan, n_inf, len(values), metric_col,
        )

    lo, hi = _VALID_RANGES.get(metric, (None, None))
    if lo is not None and hi is not None:
        finite = values.replace([float("inf"), float("-inf")], pd.NA).dropna()
        out_of_range = ((finite < lo) | (finite > hi)).sum()
        if out_of_range:
            logger.warning(
                "%s output: %d/%d genes outside expected range [%.2f, %.2f]. "
                "If many genes are out of range, suspect a coRdon version "
                "mismatch or an unexpected background set.",
                metric, int(out_of_range), len(finite), lo, hi,
            )


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
