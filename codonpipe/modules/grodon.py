"""gRodon2 growth rate prediction wrapper.

Calls the gRodon2 R package (Weissman et al. 2021) via subprocess to predict
minimum doubling time from codon usage bias of highly expressed genes.

gRodon2 uses three codon usage metrics:
  - CUBHE:  Codon Usage Bias in Highly Expressed genes (MILC-based)
  - ConsistencyHE:  Consistency of CUB across highly expressed genes
  - CPB:  Genome-wide Codon Pair Bias

These are fed into a Box-Cox-transformed linear regression trained on the
Madin et al. database of experimentally measured doubling times.

Requirements:
  - R >= 4.0
  - R packages: gRodon (v2+), Biostrings, coRdon, matrixStats, dplyr

If R or gRodon2 is not installed the module logs a warning and returns None.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger("codonpipe")

# ── Availability detection ──────────────────────────────────────────────

_GRODON_AVAILABLE: bool | None = None  # cached after first probe


def is_grodon_available() -> bool:
    """Check whether R and gRodon2 are installed and loadable."""
    global _GRODON_AVAILABLE
    if _GRODON_AVAILABLE is not None:
        return _GRODON_AVAILABLE

    rscript = shutil.which("Rscript")
    if rscript is None:
        logger.info("gRodon2: Rscript not found on PATH; gRodon2 predictions will be skipped")
        _GRODON_AVAILABLE = False
        return False

    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", "library(gRodon); cat(packageVersion('gRodon')[[1]], sep='.')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info("gRodon2: found version %s", version)
            _GRODON_AVAILABLE = True
        else:
            logger.info("gRodon2: R package not loadable; gRodon2 predictions will be skipped")
            logger.debug("gRodon2 probe stderr: %s", result.stderr.strip())
            _GRODON_AVAILABLE = False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.info("gRodon2: probe failed (%s); gRodon2 predictions will be skipped", e)
        _GRODON_AVAILABLE = False

    return _GRODON_AVAILABLE


# ── R script template ───────────────────────────────────────────────────

_R_SCRIPT = r"""
# gRodon2 wrapper called by CodonPipe
# Arguments: <cds_fasta> <output_json>
suppressPackageStartupMessages({
  library(gRodon)
  library(Biostrings)
})

args <- commandArgs(trailingOnly = TRUE)
cds_fasta  <- args[1]
output_json <- args[2]

genes <- readDNAStringSet(cds_fasta)

# Identify ribosomal proteins from FASTA headers (same regex as gRodon's own getStatistics)
highly_expressed <- grepl(
  "^(?!.*(methyl|hydroxy)).*0S ribosomal protein",
  names(genes),
  ignore.case = TRUE,
  perl = TRUE
)

n_he <- sum(highly_expressed)
if (n_he < 1) {
  # Write a JSON indicating no HE genes
  result <- list(
    status = "no_highly_expressed_genes",
    n_highly_expressed = 0
  )
  writeLines(jsonlite::toJSON(result, auto_unbox = TRUE), output_json)
  quit(save = "no", status = 0)
}

# Run gRodon2 in full mode with default (Madin) training set
tryCatch({
  pred <- predictGrowth(genes, highly_expressed, mode = "full", training_set = "madin")

  result <- list(
    status = "success",
    d = pred$d,
    lower_ci = pred$LowerCI,
    upper_ci = pred$UpperCI,
    CUBHE = pred$CUBHE,
    ConsistencyHE = pred$ConsistencyHE,
    CPB = pred$CPB,
    GC = pred$GC,
    n_highly_expressed = n_he,
    filtered_sequences = pred$FilteredSequences
  )

  writeLines(jsonlite::toJSON(result, auto_unbox = TRUE), output_json)
}, error = function(e) {
  result <- list(
    status = "error",
    message = conditionMessage(e)
  )
  writeLines(jsonlite::toJSON(result, auto_unbox = TRUE), output_json)
})
"""


# ── Public API ──────────────────────────────────────────────────────────

def run_grodon(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
) -> dict | None:
    """Run gRodon2 growth rate prediction on a CDS FASTA file.

    Args:
        ffn_path: Path to CDS nucleotide FASTA (in-frame coding sequences).
        output_dir: Directory to save the result TSV.
        sample_id: Sample identifier.

    Returns:
        Dict with gRodon2 results (doubling time, CI, codon stats),
        or None if gRodon2 is unavailable or the run fails.
    """
    if not is_grodon_available():
        return None

    if not ffn_path.exists():
        logger.warning("gRodon2: CDS FASTA not found: %s", ffn_path)
        return None

    eco_dir = output_dir / "bio_ecology"
    eco_dir.mkdir(parents=True, exist_ok=True)

    # Write R script to temp file and run it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as r_file:
        r_file.write(_R_SCRIPT)
        r_script_path = r_file.name

    json_out = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json_out_path = json_out.name
    json_out.close()

    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", r_script_path, str(ffn_path), json_out_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min should be plenty for a single genome
        )

        if result.returncode != 0:
            logger.warning("gRodon2: R script failed (exit %d): %s", result.returncode, result.stderr.strip()[:500])
            return None

        # Parse JSON output
        with open(json_out_path) as f:
            raw = json.load(f)

        if raw.get("status") == "no_highly_expressed_genes":
            logger.warning("gRodon2: no ribosomal proteins identified in FASTA headers")
            return None

        if raw.get("status") == "error":
            logger.warning("gRodon2: R error: %s", raw.get("message", "unknown"))
            return None

        # Build result dict
        d = raw["d"]
        lower_ci = raw["lower_ci"]
        upper_ci = raw["upper_ci"]

        # Growth class (same thresholds as gRodon2's own warnings)
        if d is None or d != d:  # NaN check
            growth_class = "very_slow"
            caveat = "gRodon2 returned NaN; doubling time too long to estimate reliably."
        elif d > 5.0:
            growth_class = "slow"
            caveat = "CUB signal saturates above ~5 h; gRodon2 may underestimate true doubling time."
        elif d > 2.0:
            growth_class = "moderate"
            caveat = ""
        else:
            growth_class = "fast"
            caveat = ""

        grodon_result = {
            "predicted_doubling_time_hours": float(d) if d is not None else None,
            "lower_ci_hours": float(lower_ci) if lower_ci is not None else None,
            "upper_ci_hours": float(upper_ci) if upper_ci is not None else None,
            "CUBHE": float(raw["CUBHE"]),
            "ConsistencyHE": float(raw["ConsistencyHE"]),
            "CPB": float(raw["CPB"]) if raw.get("CPB") is not None else None,
            "GC": float(raw["GC"]),
            "n_highly_expressed": int(raw["n_highly_expressed"]),
            "filtered_sequences": int(raw["filtered_sequences"]),
            "growth_class": growth_class,
            "model": "gRodon2_full_madin",
            "caveat": caveat,
        }

        # Save TSV
        out_path = eco_dir / f"{sample_id}_grodon2_prediction.tsv"
        pd.DataFrame([grodon_result]).to_csv(out_path, sep="\t", index=False)
        grodon_result["path"] = out_path

        logger.info(
            "gRodon2: predicted doubling time = %.2f h [%.2f, %.2f] (class: %s, %d HE genes)",
            d, lower_ci, upper_ci, growth_class, raw["n_highly_expressed"],
        )
        return grodon_result

    except subprocess.TimeoutExpired:
        logger.warning("gRodon2: R script timed out after 300 s")
        return None
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("gRodon2: failed to parse output: %s", e)
        return None
    finally:
        Path(r_script_path).unlink(missing_ok=True)
        Path(json_out_path).unlink(missing_ok=True)
