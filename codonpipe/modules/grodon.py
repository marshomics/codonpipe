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
Run ``codonpipe install-grodon`` to install gRodon2 and its dependencies.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("codonpipe")

# ── Availability detection ──────────────────────────────────────────────

_GRODON_AVAILABLE: bool | None = None  # cached after first probe

# R script that installs gRodon2 and its Bioconductor dependencies.
# Invoked by ``codonpipe install-grodon`` (not at runtime).
_INSTALL_R_SCRIPT = r"""
# Install gRodon2 and dependencies for CodonPipe
# Run via:  codonpipe install-grodon
#      or:  Rscript --no-save --no-restore <this_file>

local({
  # ── Ensure a writable user library exists ──────────────────────
  # In non-interactive mode R will NOT prompt to create a personal
  # library, so install.packages() may fail silently or install into
  # a session-temp path that the next Rscript invocation cannot see.
  user_lib <- Sys.getenv("R_LIBS_USER", unset = "")
  if (nchar(user_lib) == 0 || grepl("%", user_lib)) {
    user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
  }
  if (!dir.exists(user_lib)) {
    dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
    cat(sprintf("Created user R library: %s\n", user_lib))
  }
  .libPaths(c(user_lib, .libPaths()))
  cat(sprintf("R library paths: %s\n", paste(.libPaths(), collapse = "; ")))

  # 1. BiocManager (needed to install Bioconductor packages)
  if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "https://cloud.r-project.org",
                     lib = user_lib)

  # 2. Bioconductor dependencies
  bioc_pkgs <- c("Biostrings", "coRdon")
  for (pkg in bioc_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing Bioconductor package: %s\n", pkg))
      BiocManager::install(pkg, ask = FALSE, update = FALSE, lib = user_lib)
    }
  }

  # 3. CRAN dependencies
  cran_pkgs <- c("matrixStats", "dplyr", "jsonlite", "remotes")
  for (pkg in cran_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing CRAN package: %s\n", pkg))
      install.packages(pkg, repos = "https://cloud.r-project.org",
                       lib = user_lib)
    }
  }

  # 4. gRodon2 from GitHub
  if (!requireNamespace("gRodon", quietly = TRUE)) {
    cat("Installing gRodon2 from GitHub (jlw-ecoevo/gRodon2)\n")
    remotes::install_github("jlw-ecoevo/gRodon2", upgrade = "never",
                            lib = user_lib)
  }

  # Verify
  library(gRodon)
  cat(sprintf("gRodon2 version %s installed successfully\n",
              paste(packageVersion("gRodon"), collapse = ".")))
  cat(sprintf("INSTALL_LIB=%s\n", user_lib))
})
"""


def _r_user_lib(rscript: str) -> str | None:
    """Return R's resolved user library path.

    Asks R to expand ``R_LIBS_USER`` (which may contain ``%p`` and ``%v``
    placeholders for platform and version) into an actual directory path.
    Falls back to ``~/R/library`` if the environment variable is unset.
    """
    # R expands the placeholders internally via path.expand + sub;
    # we ask R itself to do it so we get the same path it would use.
    r_code = (
        "p <- Sys.getenv('R_LIBS_USER', unset='');"
        "if (nchar(p) == 0) p <- file.path(Sys.getenv('HOME'), 'R', 'library');"
        "p <- gsub('%p', R.version$platform, p, fixed=TRUE);"
        "p <- gsub('%v', paste(R.version$major, R.version$minor, sep='.'), p, fixed=TRUE);"
        "p <- gsub('%V', paste(R.version$major, substr(R.version$minor, 1, 1), sep='.'), p, fixed=TRUE);"
        "cat(p)"
    )
    try:
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", "-e", r_code],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def _probe_grodon(
    rscript: str,
    extra_lib: str | None = None,
    verbose: bool = False,
) -> str | None:
    """Try to load gRodon2 in R. Returns the version string or None.

    Uses ``--no-save --no-restore`` instead of ``--vanilla`` so that
    ``.Renviron`` and ``.Rprofile`` are still read — this is critical
    for finding packages in user library paths (R_LIBS_USER).

    If *extra_lib* is given it is prepended to ``.libPaths()`` before
    the ``library()`` call, which handles the case where the user lib
    directory was created during install but isn't on the default path.

    When *verbose* is True the R stderr is logged at INFO level (used
    during install verification to surface the actual error).
    """
    # Use a sentinel marker so we can extract the version from stdout
    # even when R prints warnings or other messages before it.
    marker = "GRODON_VER="
    preamble = ""
    if extra_lib:
        preamble = f".libPaths(c('{extra_lib}', .libPaths())); "
    r_code = (
        preamble
        + f"library(gRodon); cat('{marker}', "
        + "paste(packageVersion('gRodon')[[1]], collapse='.'), '\\n', sep='')"
    )
    try:
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", "-e", r_code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith(marker):
                    ver = line[len(marker):].strip()
                    if ver:
                        return ver
            # Marker not found — library loaded but version output garbled
            logger.debug("gRodon2 probe stdout (no marker):\n%s", result.stdout[:500])
        else:
            log = logger.info if verbose else logger.debug
            log("gRodon2 probe failed (exit %d):\n%s",
                result.returncode, result.stderr[:2000])
            if verbose:
                logger.info("gRodon2 probe .libPaths used: %s",
                            extra_lib or "(default)")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return None


def install_grodon(timeout: int = 600) -> bool:
    """Install gRodon2 and its R dependencies.

    Requires R (Rscript) to be on PATH. Installs BiocManager,
    Biostrings, coRdon, matrixStats, dplyr, jsonlite, remotes,
    and gRodon2 from GitHub (jlw-ecoevo/gRodon2).

    Called by ``codonpipe install-grodon``. Not called at runtime.

    Args:
        timeout: Maximum seconds to allow for the install process.

    Returns:
        True if gRodon2 is loadable after installation, False otherwise.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        logger.error("Rscript not found on PATH. Install R (>= 4.0) first.")
        return False

    logger.info("Installing gRodon2 and R dependencies...")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(_INSTALL_R_SCRIPT)
        script_path = f.name

    try:
        # Use --no-save --no-restore (not --vanilla) so .Renviron is read
        # and R_LIBS_USER is respected for both install and later loading.
        result = subprocess.run(
            [rscript, "--no-save", "--no-restore", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("gRodon2 installation succeeded")
            logger.debug("gRodon2 install stdout:\n%s", result.stdout)

            # Extract the library path the install script used so the
            # probe can look in the same place even if .Renviron is absent.
            install_lib = None
            for line in result.stdout.splitlines():
                if line.startswith("INSTALL_LIB="):
                    install_lib = line.split("=", 1)[1].strip()
                    break

            # Verify the install is loadable — try plain probe first,
            # then with the explicit library path from the install.
            # Use verbose=True so any R error is logged at INFO level.
            version = _probe_grodon(rscript, verbose=True)
            if version is None and install_lib:
                logger.info(
                    "Plain probe failed; retrying with lib=%s", install_lib
                )
                version = _probe_grodon(rscript, extra_lib=install_lib,
                                        verbose=True)
            if version is not None:
                logger.info("gRodon2 version %s is now available", version)
                return True
            else:
                logger.error(
                    "gRodon2 install script succeeded but the package "
                    "could not be loaded. Check R library paths.\n"
                    "Install lib: %s",
                    install_lib or "(unknown)",
                )
                return False
        else:
            logger.error(
                "gRodon2 installation failed (exit %d).\nstderr:\n%s",
                result.returncode,
                result.stderr[:3000],
            )
            return False
    except subprocess.TimeoutExpired:
        logger.error("gRodon2 installation timed out after %d s", timeout)
        return False
    except (FileNotFoundError, OSError) as e:
        logger.error("gRodon2 installation failed: %s", e)
        return False
    finally:
        Path(script_path).unlink(missing_ok=True)


def is_grodon_available() -> bool:
    """Check whether R and gRodon2 are installed and loadable.

    Does NOT attempt installation — run ``codonpipe install-grodon``
    to set up gRodon2 before using the pipeline.
    """
    global _GRODON_AVAILABLE
    if _GRODON_AVAILABLE is not None:
        return _GRODON_AVAILABLE

    rscript = shutil.which("Rscript")
    if rscript is None:
        logger.info("gRodon2: Rscript not found on PATH; gRodon2 predictions will be skipped")
        _GRODON_AVAILABLE = False
        return False

    version = _probe_grodon(rscript)
    if version is None:
        # The user library may not be on R's default .libPaths() (e.g.
        # when .Renviron is absent or R_LIBS_USER has %V placeholders).
        # Try once more with the conventional ~/R/library path.
        user_lib = _r_user_lib(rscript)
        if user_lib:
            version = _probe_grodon(rscript, extra_lib=user_lib)

    if version is not None:
        logger.info("gRodon2: found version %s", version)
        _GRODON_AVAILABLE = True
    else:
        logger.info(
            "gRodon2: R package not loadable; predictions will be skipped. "
            "Run 'codonpipe install-grodon' to install it."
        )
        _GRODON_AVAILABLE = False

    return _GRODON_AVAILABLE


# ── R script template ───────────────────────────────────────────────────

_R_SCRIPT = r"""
# gRodon2 wrapper called by CodonPipe
# Arguments: <cds_fasta> <output_json> [<he_ids_file>] [<background_ids_file>]
#
# he_ids_file (optional): Gene IDs (one per line) to mark as highly expressed.
#   When provided these replace gRodon2's built-in RP regex. Typically the
#   high-MELP tier genes scored against the Mahalanobis-defined reference.
#
# background_ids_file (optional): Gene IDs (one per line) that define the
#   background gene set.  When provided the input FASTA is subsetted to only
#   these genes BEFORE computing CUBHE and CPB, so the "genome average" that
#   gRodon2 measures against is the Mahalanobis-defined optimised set rather
#   than the full CDS complement.  The he_ids must be a subset of these.

# Ensure user library is on the search path even when .Renviron is absent
local({
  user_lib <- Sys.getenv("R_LIBS_USER", unset = "")
  if (nchar(user_lib) == 0 || grepl("%%", user_lib)) {
    user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
  }
  if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
})

suppressPackageStartupMessages({
  library(gRodon)
  library(Biostrings)
})

args <- commandArgs(trailingOnly = TRUE)
cds_fasta       <- args[1]
output_json     <- args[2]
he_ids_file     <- if (length(args) >= 3 && nchar(args[3]) > 0) args[3] else NULL
bg_ids_file     <- if (length(args) >= 4 && nchar(args[4]) > 0) args[4] else NULL

genes <- readDNAStringSet(cds_fasta)

# Extract the gene ID (first whitespace-delimited token) from each FASTA header
gene_ids <- sub("\\s.*", "", names(genes))

# ── Optional: subset to background (Mahalanobis-defined) gene set ──
n_total <- length(genes)
if (!is.null(bg_ids_file) && file.exists(bg_ids_file)) {
  bg_ids <- readLines(bg_ids_file)
  bg_ids <- trimws(bg_ids[nchar(trimws(bg_ids)) > 0])
  keep <- gene_ids %in% bg_ids
  if (sum(keep) < 10) {
    cat(sprintf("WARNING: only %d/%d CDS matched background IDs; using full genome\n",
                sum(keep), n_total))
  } else {
    genes    <- genes[keep]
    gene_ids <- gene_ids[keep]
    cat(sprintf("Subsetting to %d/%d Mahalanobis-defined background genes\n",
                length(genes), n_total))
  }
}

# ── Mark highly expressed genes ──
if (!is.null(he_ids_file) && file.exists(he_ids_file)) {
  he_ids <- readLines(he_ids_file)
  he_ids <- trimws(he_ids[nchar(trimws(he_ids)) > 0])
  highly_expressed <- gene_ids %in% he_ids
  cat(sprintf("Using %d high-MELP gene IDs; matched %d/%d CDS\n",
              length(he_ids), sum(highly_expressed), length(genes)))
} else {
  # Fallback: gRodon2's own regex on FASTA headers
  highly_expressed <- grepl(
    "^(?!.*(methyl|hydroxy)).*0S ribosomal protein",
    names(genes),
    ignore.case = TRUE,
    perl = TRUE
  )
  cat(sprintf("Using gRodon2 header regex; matched %d/%d CDS\n",
              sum(highly_expressed), length(genes)))
}

n_he <- sum(highly_expressed)
if (n_he < 1) {
  # Write a JSON indicating no HE genes
  result <- list(
    status = "no_highly_expressed_genes",
    n_highly_expressed = 0,
    n_background = length(genes)
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
    n_background = length(genes),
    n_total_cds = n_total,
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
    rp_ids_file: Path | str | None = None,
    he_ids_file: Path | str | None = None,
    background_ids_file: Path | str | None = None,
) -> dict | None:
    """Run gRodon2 growth rate prediction on a CDS FASTA file.

    Args:
        ffn_path: Path to CDS nucleotide FASTA (in-frame coding sequences).
        output_dir: Directory to save the result TSV.
        sample_id: Sample identifier.
        rp_ids_file: Optional path to a file listing ribosomal protein gene
            IDs (one per line), as identified by COGclassifier.  Used as
            the highly-expressed set when *he_ids_file* is not provided.
        he_ids_file: Optional path to a file listing highly-expressed gene
            IDs (one per line).  When provided, takes precedence over
            *rp_ids_file* for marking the HE set.  Typically contains
            high-MELP-tier genes scored against the Mahalanobis reference.
        background_ids_file: Optional path to a file listing the background
            gene set (one gene ID per line).  When provided, the input FASTA
            is subsetted to only these genes before computing CUBHE and CPB,
            so the "genome average" is the Mahalanobis-defined optimised set
            rather than the full CDS complement.

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

    # Resolve which file supplies the HE gene IDs: prefer explicit
    # high-MELP IDs, fall back to RP IDs, then gRodon2's own regex.
    effective_he = None
    if he_ids_file is not None and Path(he_ids_file).exists():
        effective_he = str(he_ids_file)
    elif rp_ids_file is not None and Path(rp_ids_file).exists():
        effective_he = str(rp_ids_file)

    effective_bg = None
    if background_ids_file is not None and Path(background_ids_file).exists():
        effective_bg = str(background_ids_file)

    try:
        cmd = ["Rscript", "--no-save", "--no-restore", r_script_path,
               str(ffn_path), json_out_path,
               effective_he or "",
               effective_bg or ""]
        result = subprocess.run(
            cmd,
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
            if effective_he:
                logger.warning(
                    "gRodon2: none of the HE IDs from %s matched CDS in %s "
                    "(background: %s)",
                    effective_he, ffn_path,
                    effective_bg or "full genome",
                )
            else:
                logger.warning(
                    "gRodon2: no highly expressed genes identified "
                    "(no HE IDs file provided and header regex found no matches)"
                )
            return None

        if raw.get("status") == "error":
            logger.warning("gRodon2: R error: %s", raw.get("message", "unknown"))
            return None

        # Build result dict
        d = raw["d"]
        lower_ci = raw["lower_ci"]
        upper_ci = raw["upper_ci"]

        # Growth class (same thresholds as gRodon2's own warnings)
        if d is None or (d is not None and np.isnan(d)):
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

        # gRodon2 training range: Madin et al. (2020) dataset spans
        # ~0.2 h to ~72 h doubling time; predictions outside this range
        # are extrapolations and should be flagged.
        training_min_hours = 0.2
        training_max_hours = 72.0
        if d is not None and not np.isnan(d):
            in_training_range = training_min_hours <= d <= training_max_hours
        else:
            in_training_range = False

        # Determine the reference mode for provenance tracking
        if effective_bg and effective_he:
            ref_mode = "mahalanobis_melp"
        elif effective_he:
            ref_mode = "melp_he"
        elif rp_ids_file and Path(rp_ids_file).exists():
            ref_mode = "rp_ids"
        else:
            ref_mode = "header_regex"

        grodon_result = {
            "predicted_doubling_time_hours": float(d) if d is not None else None,
            "lower_ci_hours": float(lower_ci) if lower_ci is not None else None,
            "upper_ci_hours": float(upper_ci) if upper_ci is not None else None,
            "CUBHE": float(raw["CUBHE"]),
            "ConsistencyHE": float(raw["ConsistencyHE"]),
            "CPB": float(raw["CPB"]) if raw.get("CPB") is not None else None,
            "GC": float(raw["GC"]),
            "n_highly_expressed": int(raw["n_highly_expressed"]),
            "n_background": int(raw.get("n_background", raw["n_highly_expressed"])),
            "n_total_cds": int(raw.get("n_total_cds", 0)),
            "filtered_sequences": int(raw["filtered_sequences"]),
            "growth_class": growth_class,
            "model": "gRodon2_full_madin",
            "reference_mode": ref_mode,
            "in_training_range": in_training_range,
            "training_range_hours": f"{training_min_hours}-{training_max_hours}",
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
