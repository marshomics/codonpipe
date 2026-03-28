"""I/O utilities for CodonPipe."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO


def check_tool(name: str) -> str:
    """Return full path of an executable, or raise if not found."""
    import shutil

    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Required tool '{name}' not found in PATH. "
            f"Install it or activate the codonpipe conda environment."
        )
    return path


def run_cmd(
    cmd: list[str],
    description: str = "",
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a shell command with logging and error handling.

    Args:
        cmd: Command as list of strings.
        description: Human-readable description for logging.
        cwd: Working directory.
        check: Raise on non-zero exit.
        capture: Capture stdout/stderr.
        timeout: Timeout in seconds.

    Returns:
        CompletedProcess instance.
    """
    import logging

    logger = logging.getLogger("codonpipe")
    logger.debug("Running: %s", " ".join(str(c) for c in cmd))
    if description:
        logger.info(description)

    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        stderr_msg = result.stderr[:2000] if result.stderr else "(no stderr)"
        stdout_msg = result.stdout[:2000] if result.stdout else ""
        error_parts = [
            f"Command failed (exit {result.returncode}): {' '.join(str(c) for c in cmd)}",
            f"stderr: {stderr_msg}",
        ]
        if stdout_msg:
            error_parts.append(f"stdout: {stdout_msg}")
        raise RuntimeError("\n".join(error_parts))
    return result


def read_fasta_ids(fasta_path: Path) -> list[str]:
    """Return sequence IDs from a FASTA file."""
    return [rec.id for rec in SeqIO.parse(str(fasta_path), "fasta")]


def load_batch_table(table_path: Path) -> pd.DataFrame:
    """Load a batch input table (TSV or CSV) with at minimum a 'genome_path' column.

    Accepted columns:
        - genome_path (required): path to genome FASTA
        - sample_id (optional): identifier; defaults to filename stem
        - prokka_faa (optional): path to pre-existing Prokka .faa file
        - prokka_ffn (optional): path to pre-existing Prokka .ffn file
        - prokka_gff (optional): path to pre-existing Prokka .gff file
        - gff_path (optional): path to a GFF3 file for tRNA extraction
        - kofam_results (optional): path to pre-computed KofamScan detail-tsv output
        - metadata columns (optional): any additional columns carried through

    When prokka_faa and prokka_ffn are both populated for a row, Prokka
    is skipped for that sample. Rows where either column is empty or
    missing will run Prokka normally. You can mix pre-run and fresh
    samples in the same table.

    Returns:
        DataFrame with validated paths.
    """
    import logging

    logger = logging.getLogger("codonpipe")

    sep = "\t" if table_path.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(table_path, sep=sep, dtype=str)

    if "genome_path" not in df.columns:
        raise ValueError(
            f"Batch table must have a 'genome_path' column. Found: {list(df.columns)}"
        )

    # Validate genome paths exist
    missing = []
    for idx, row in df.iterrows():
        p = Path(row["genome_path"])
        if not p.is_file():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} genome file(s) not found:\n" + "\n".join(missing[:10])
        )

    # Default sample_id to filename stem
    if "sample_id" not in df.columns:
        df["sample_id"] = df["genome_path"].apply(lambda x: Path(x).stem)

    # Validate pre-existing Prokka and GFF file paths if columns are present
    prokka_cols = {"prokka_faa", "prokka_ffn", "prokka_gff", "gff_path", "kofam_results"}
    present_prokka_cols = prokka_cols & set(df.columns)

    if present_prokka_cols:
        missing_prokka = []
        for idx, row in df.iterrows():
            sid = row.get("sample_id", f"row {idx}")
            for col in present_prokka_cols:
                val = str(row.get(col, "") or "").strip()
                if val:
                    p = Path(val)
                    if not p.is_file():
                        missing_prokka.append(f"  {sid}: {col} = {val}")

        if missing_prokka:
            raise FileNotFoundError(
                f"{len(missing_prokka)} pre-existing Prokka file(s) not found:\n"
                + "\n".join(missing_prokka[:10])
            )

        # Warn about incomplete pairs (faa without ffn or vice versa)
        if "prokka_faa" in df.columns and "prokka_ffn" in df.columns:
            for idx, row in df.iterrows():
                sid = row.get("sample_id", f"row {idx}")
                faa_val = str(row.get("prokka_faa", "") or "").strip()
                ffn_val = str(row.get("prokka_ffn", "") or "").strip()
                if bool(faa_val) != bool(ffn_val):
                    logger.warning(
                        "Sample '%s' has only one of prokka_faa/prokka_ffn set. "
                        "Both must be provided to skip Prokka. Prokka will run for this sample.",
                        sid,
                    )

    return df


def write_tsv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a DataFrame as TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, **kwargs)
