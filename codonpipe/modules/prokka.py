"""Module for running Prokka gene prediction on microbial genome assemblies."""

from __future__ import annotations

import logging
from pathlib import Path

from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")


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
    prokka_dir = output_dir / "prokka"

    # Check for existing output
    expected_faa = prokka_dir / f"{sample_id}.faa"
    if expected_faa.exists() and not force:
        logger.info("Prokka output already exists for %s, skipping (use --force to rerun)", sample_id)
        return _collect_outputs(prokka_dir, sample_id)

    cmd = [
        "prokka",
        "--outdir", str(prokka_dir),
        "--prefix", sample_id,
        "--kingdom", kingdom,
        "--cpus", str(cpus),
        "--locustag", sample_id,
        "--compliant",
    ]
    if metagenome:
        cmd.append("--metagenome")
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(genome_fasta))

    run_cmd(cmd, description=f"Running Prokka on {sample_id}")

    outputs = _collect_outputs(prokka_dir, sample_id)
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
