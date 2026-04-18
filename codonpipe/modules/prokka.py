"""Module for running Prokka gene prediction on microbial genome assemblies."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path

from codonpipe.utils.io import get_output_subdir, check_tool, run_cmd

logger = logging.getLogger("codonpipe")

# Prokka with --compliant requires locus tags ≤ 37 characters total.
# The locus tag gets a "_NNNNN" suffix (6 chars), so the prefix must be ≤ 31.
_MAX_LOCUSTAG_LEN = 31

# GenBank format limits contig/sequence IDs to 37 characters.
_MAX_CONTIG_ID_LEN = 37


def _safe_locustag(sample_id: str) -> str:
    """Derive a GenBank-compliant locus tag from a sample ID.

    If the sample_id fits within the limit, it's used as-is (after replacing
    characters that Prokka rejects).  Otherwise, the tag is truncated and a
    short hash suffix is appended to preserve uniqueness.
    """
    # Prokka locus tags must be alphanumeric + underscores
    tag = "".join(c if c.isalnum() or c == "_" else "_" for c in sample_id)

    if len(tag) <= _MAX_LOCUSTAG_LEN:
        return tag

    # Truncate and append 6-char hash to avoid collisions
    digest = hashlib.md5(sample_id.encode()).hexdigest()[:6]
    max_prefix = _MAX_LOCUSTAG_LEN - len(digest) - 1  # -1 for underscore
    return f"{tag[:max_prefix]}_{digest}"


def _sanitize_contig_ids(input_fasta: Path, output_fasta: Path) -> dict[str, str]:
    """Rewrite FASTA headers to fit within the GenBank 37-character limit.

    Contigs that already fit are left unchanged.  Long IDs are replaced with
    ``contig_NNNN`` (sequential numbering), which is always ≤ 37 chars.

    A mapping file (``{output_fasta}.contig_map.tsv``) is written alongside
    the sanitized FASTA so the original names can be recovered.

    Args:
        input_fasta: Original genome FASTA.
        output_fasta: Where to write the sanitized FASTA.

    Returns:
        Dict mapping new contig ID → original contig ID.
    """
    needs_rename = False
    # Quick scan: check if any contig ID exceeds the limit
    with open(input_fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                cid = line[1:].split()[0]
                if len(cid) > _MAX_CONTIG_ID_LEN:
                    needs_rename = True
                    break

    if not needs_rename:
        # No renaming needed — just copy the file
        shutil.copy2(input_fasta, output_fasta)
        return {}

    contig_map: dict[str, str] = {}
    idx = 0
    with open(input_fasta) as fin, open(output_fasta, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                original_id = line[1:].split()[0]
                rest = line[1 + len(original_id):]  # preserve description
                if len(original_id) > _MAX_CONTIG_ID_LEN:
                    idx += 1
                    new_id = f"contig_{idx:04d}"
                    contig_map[new_id] = original_id
                    fout.write(f">{new_id}{rest}")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    # Save mapping for traceability
    map_path = Path(str(output_fasta) + ".contig_map.tsv")
    with open(map_path, "w") as mf:
        mf.write("new_contig_id\toriginal_contig_id\n")
        for new_id, orig_id in contig_map.items():
            mf.write(f"{new_id}\t{orig_id}\n")

    logger.info(
        "Renamed %d/%d contigs with IDs > %d chars for GenBank compliance",
        len(contig_map), idx, _MAX_CONTIG_ID_LEN,
    )
    return contig_map


def run_prokka(
    genome_fasta: Path,
    output_dir: Path,
    sample_id: str,
    kingdom: str = "Bacteria",
    cpus: int = 4,
    metagenome: bool = False,
    force: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Path]:
    """Run Prokka on a genome assembly.

    Args:
        genome_fasta: Path to input genome FASTA.
        output_dir: Directory for Prokka output.
        sample_id: Prefix/locus tag for output files.
        kingdom: Prokka --kingdom flag (Bacteria, Archaea, Viruses).
        cpus: Number of threads.
        metagenome: Use --metagenome mode.
        force: Overwrite existing output.
        extra_args: Additional Prokka arguments.

    Returns:
        Dict mapping output type to file path:
            faa, ffn, fna, gff, gbk, tsv, txt, log
    """
    check_tool("prokka")
    # Ensure the parent ``annotation/`` exists, but do NOT pre-create the
    # final ``prokka/`` subdirectory — Prokka refuses to write into an
    # existing folder unless ``--force`` is passed.
    annotation_dir = get_output_subdir(output_dir, "annotation")
    prokka_dir = annotation_dir / "prokka"

    # Check for existing output
    expected_faa = prokka_dir / f"{sample_id}.faa"
    locustag = _safe_locustag(sample_id)

    if expected_faa.exists() and not force:
        logger.info("Prokka output already exists for %s, skipping (use --force to rerun)", sample_id)
        outputs = _collect_outputs(prokka_dir, sample_id)
        outputs["locustag"] = locustag
        return outputs

    # If the directory exists but lacks the expected .faa, a prior run
    # crashed mid-way. Prokka would bail with "Folder already exists"; pass
    # --force so we can re-run cleanly instead of dead-ending.
    overwrite_stale = prokka_dir.exists() and not expected_faa.exists()
    if overwrite_stale and not force:
        logger.info(
            "Prokka output directory %s exists but is incomplete; "
            "overwriting with --force",
            prokka_dir,
        )

    if locustag != sample_id:
        logger.info(
            "Sample ID '%s' exceeds GenBank locus tag limit (%d chars); "
            "using truncated tag '%s'",
            sample_id, _MAX_LOCUSTAG_LEN, locustag,
        )

    cmd = [
        "prokka",
        "--outdir", str(prokka_dir),
        "--prefix", sample_id,
        "--kingdom", kingdom,
        "--cpus", str(cpus),
        "--locustag", locustag,
    ]
    if metagenome:
        cmd.append("--metagenome")
    if force or overwrite_stale:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(genome_fasta))

    run_cmd(cmd, description=f"Running Prokka on {sample_id}")

    outputs = _collect_outputs(prokka_dir, sample_id)
    outputs["locustag"] = locustag
    _validate_outputs(outputs, sample_id)
    return outputs


def _collect_outputs(prokka_dir: Path, sample_id: str) -> dict[str, Path]:
    """Collect expected Prokka output files."""
    extensions = ["faa", "ffn", "fna", "gff", "gbk", "tsv", "txt", "log"]
    outputs = {}
    for ext in extensions:
        p = prokka_dir / f"{sample_id}.{ext}"
        if p.exists():
            outputs[ext] = p
    return outputs


def _validate_outputs(outputs: dict[str, Path], sample_id: str) -> None:
    """Validate that critical Prokka outputs exist and are non-empty."""
    critical = ["faa", "ffn"]
    for ext in critical:
        if ext not in outputs:
            raise FileNotFoundError(
                f"Prokka failed to produce {ext} file for {sample_id}"
            )
        if outputs[ext].stat().st_size == 0:
            raise RuntimeError(
                f"Prokka produced empty {ext} file for {sample_id}. "
                "Check if the input genome contains valid sequences."
            )
    n_proteins = sum(1 for line in outputs["faa"].open() if line.startswith(">"))
    logger.info("Prokka predicted %d proteins for %s", n_proteins, sample_id)
