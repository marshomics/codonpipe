"""Module for identifying ribosomal proteins using COGclassifier."""

from __future__ import annotations

import logging
from pathlib import Path
from codonpipe.utils.io import get_output_subdir

import pandas as pd
from Bio import SeqIO

from codonpipe.utils.io import check_tool, find_gene_id_column, run_cmd

logger = logging.getLogger("codonpipe")

# Default ribosomal COG accessions bundled with the package
_RIBOSOMAL_COGS_FILE = Path(__file__).parent.parent / "data" / "ribosomal_cogs.txt"


def load_ribosomal_cogs(cogs_file: Path | None = None) -> set[str]:
    """Load the set of ribosomal protein COG accessions.

    Args:
        cogs_file: Path to a text file with one COG ID per line.
                   Defaults to the bundled ribosomal_cogs.txt.

    Returns:
        Set of COG IDs (e.g., {"COG0090", "COG0091", ...}).
    """
    path = cogs_file or _RIBOSOMAL_COGS_FILE
    cogs = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                cogs.add(line)
    logger.debug("Loaded %d ribosomal COG accessions from %s", len(cogs), path)
    return cogs


def run_cogclassifier(
    faa_path: Path,
    output_dir: Path,
    sample_id: str,
    cpus: int = 4,
    force: bool = False,
) -> Path:
    """Run COGclassifier on predicted proteins.

    Args:
        faa_path: Path to Prokka .faa output.
        output_dir: Base output directory.
        sample_id: Sample identifier.
        cpus: Number of threads for RPS-BLAST.
        force: Rerun even if output exists.

    Returns:
        Path to the COGclassifier result TSV.
    """
    check_tool("COGclassifier")
    cog_dir = get_output_subdir(output_dir, "annotation", "cogclassifier")

    # COGclassifier v1 writes result.tsv; v2 writes cog_classify.tsv
    result_candidates = [cog_dir / "result.tsv", cog_dir / "cog_classify.tsv"]

    if not force:
        for candidate in result_candidates:
            if candidate.exists():
                logger.info("COGclassifier output exists for %s, skipping", sample_id)
                return candidate

    cog_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "COGclassifier",
        "-i", str(faa_path),
        "-o", str(cog_dir),
        "-t", str(cpus),
    ]
    run_cmd(cmd, description=f"Running COGclassifier on {sample_id}")

    for candidate in result_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"COGclassifier did not produce result.tsv or cog_classify.tsv for {sample_id}. "
        f"Check {cog_dir} for error logs."
    )


def extract_ribosomal_proteins(
    cog_result_tsv: Path,
    faa_path: Path,
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    cogs_file: Path | None = None,
) -> dict[str, Path]:
    """Extract ribosomal protein sequences based on COG assignments.

    Args:
        cog_result_tsv: COGclassifier result.tsv file.
        faa_path: Prokka .faa file (amino acid sequences).
        ffn_path: Prokka .ffn file (nucleotide CDS sequences).
        output_dir: Base output directory.
        sample_id: Sample identifier.
        cogs_file: Optional custom ribosomal COGs file.

    Returns:
        Dict with paths:
            - rp_faa: ribosomal protein amino acid FASTA
            - rp_ffn: ribosomal protein nucleotide FASTA
            - non_rp_ffn: non-ribosomal protein nucleotide FASTA
            - rp_ids: text file listing ribosomal protein IDs
            - cog_assignments: filtered COG table for ribosomal proteins
    """
    ribosomal_cogs = load_ribosomal_cogs(cogs_file)

    # Parse COGclassifier results
    cog_df = pd.read_csv(cog_result_tsv, sep="\t")

    # Identify the COG ID column (COGclassifier output varies)
    cog_col = None
    for candidate in ["COG_ID", "COG", "cog_id", "COG_id", "best_hit_cog"]:
        if candidate in cog_df.columns:
            cog_col = candidate
            break
    if cog_col is None:
        # Try to find any column containing COG-like IDs
        for col in cog_df.columns:
            sample_vals = cog_df[col].dropna().astype(str).head(10)
            if sample_vals.str.match(r"^COG\d+$").any():
                cog_col = col
                break
    if cog_col is None:
        raise ValueError(
            f"Cannot find COG ID column in {cog_result_tsv}. "
            f"Columns: {list(cog_df.columns)}"
        )

    # Find query/protein ID column
    query_col = find_gene_id_column(cog_df, fallback_to_first=True)

    # Filter for ribosomal proteins
    rp_mask = cog_df[cog_col].isin(ribosomal_cogs)
    rp_df = cog_df[rp_mask].copy()
    rp_ids = set(rp_df[query_col].astype(str))

    logger.info(
        "Found %d ribosomal proteins (out of %d total) in %s",
        len(rp_ids), len(cog_df), sample_id,
    )

    if len(rp_ids) == 0:
        logger.warning(
            "No ribosomal proteins found for %s. This may indicate a problem "
            "with the genome assembly or COG classification.", sample_id,
        )

    # Extract sequences
    rp_dir = get_output_subdir(output_dir, "annotation", "ribosomal_proteins")
    rp_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Ribosomal protein amino acid sequences
    rp_faa = rp_dir / f"{sample_id}_rp.faa"
    _extract_seqs(faa_path, rp_faa, rp_ids, include=True)
    outputs["rp_faa"] = rp_faa

    # Ribosomal protein nucleotide sequences
    rp_ffn = rp_dir / f"{sample_id}_rp.ffn"
    _extract_seqs(ffn_path, rp_ffn, rp_ids, include=True)
    outputs["rp_ffn"] = rp_ffn

    # Non-ribosomal nucleotide sequences
    non_rp_ffn = rp_dir / f"{sample_id}_non_rp.ffn"
    _extract_seqs(ffn_path, non_rp_ffn, rp_ids, include=False)
    outputs["non_rp_ffn"] = non_rp_ffn

    # Save ribosomal protein IDs
    rp_id_file = rp_dir / f"{sample_id}_rp_ids.txt"
    rp_id_file.write_text("\n".join(sorted(rp_ids)) + "\n")
    outputs["rp_ids"] = rp_id_file

    # Save filtered COG assignments
    cog_out = rp_dir / f"{sample_id}_rp_cog_assignments.tsv"
    rp_df.to_csv(cog_out, sep="\t", index=False)
    outputs["cog_assignments"] = cog_out

    return outputs


def _extract_seqs(fasta_in: Path, fasta_out: Path, ids: set[str], include: bool) -> int:
    """Extract or exclude sequences by ID from a FASTA file.

    Args:
        fasta_in: Input FASTA.
        fasta_out: Output FASTA.
        ids: Set of sequence IDs.
        include: If True, keep only sequences with IDs in `ids`.
                 If False, keep only sequences NOT in `ids`.

    Returns:
        Number of sequences written.
    """
    count = 0
    with open(fasta_out, "w") as out:
        for rec in SeqIO.parse(str(fasta_in), "fasta"):
            keep = (rec.id in ids) if include else (rec.id not in ids)
            if keep:
                SeqIO.write(rec, out, "fasta")
                count += 1
    return count
