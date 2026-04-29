"""Module for running KofamScan functional annotation."""

from __future__ import annotations

import logging
from pathlib import Path
from codonpipe.utils.io import get_output_subdir

import pandas as pd

from codonpipe.utils.io import check_tool, run_cmd

logger = logging.getLogger("codonpipe")


def run_kofamscan(
    faa_path: Path,
    output_dir: Path,
    sample_id: str,
    profile_dir: Path | None = None,
    ko_list: Path | None = None,
    cpus: int = 4,
    force: bool = False,
) -> Path:
    """Run KofamScan on predicted protein sequences.

    Args:
        faa_path: Path to Prokka .faa file.
        output_dir: Base output directory.
        sample_id: Sample identifier.
        profile_dir: Path to KOfam profile directory. If None, uses default.
        ko_list: Path to ko_list file. If None, uses default.
        cpus: Number of threads.
        force: Rerun even if output exists.

    Returns:
        Path to the KofamScan result file.
    """
    check_tool("exec_annotation")

    kofam_dir = get_output_subdir(output_dir, "annotation", "kofamscan")
    kofam_dir.mkdir(parents=True, exist_ok=True)
    result_file = kofam_dir / f"{sample_id}_kofam.tsv"
    tmp_dir = kofam_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if result_file.exists() and not force:
        logger.info("KofamScan output exists for %s, skipping", sample_id)
        return result_file

    cmd = [
        "exec_annotation",
        "-o", str(result_file),
        "--cpu", str(cpus),
        "--format", "detail-tsv",
        "--tmp-dir", str(tmp_dir),
    ]
    if profile_dir is not None:
        cmd.extend(["--profile", str(profile_dir)])
    if ko_list is not None:
        cmd.extend(["--ko-list", str(ko_list)])
    cmd.append(str(faa_path))

    run_cmd(cmd, description=f"Running KofamScan on {sample_id}", timeout=7200)

    if not result_file.exists():
        raise FileNotFoundError(f"KofamScan did not produce output for {sample_id}")

    return result_file


def parse_kofamscan(result_file: Path) -> pd.DataFrame:
    """Parse KofamScan detail-tsv output into a DataFrame.

    Multi-domain reduction:
        Genes with multiple significant KO hits (typical for fusion
        proteins or multi-functional enzymes) are reduced to a single
        row keyed on the highest-scoring KO. This is pragmatic but
        lossy — a kinase fused to a phosphatase will keep only the
        higher-scoring activity. Downstream pathway / GSEA analyses
        will therefore underestimate enrichment for the discarded KO.
        If your downstream analysis depends on the full multi-KO
        mapping, parse the raw detail-tsv instead of using this
        function. The number of dropped per-gene secondary hits is
        logged at INFO level so you can decide if it materially
        affects your results.

    Returns DataFrame with columns:
        gene_name, KO, thrshld, score, E_value, KO_definition
    """
    rows = []
    with open(result_file) as fh:
        for line in fh:
            # Skip comment lines and blank lines
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 7:
                significance = parts[0].strip()
                gene_name = parts[1].strip()
                ko = parts[2].strip()
                thrshld = parts[3].strip()
                score = parts[4].strip()
                e_value = parts[5].strip()
                ko_def = parts[6].strip()

                # Only keep significant hits (marked with *)
                if significance == "*":
                    rows.append({
                        "gene_name": gene_name,
                        "KO": ko,
                        "thrshld": thrshld,
                        "score": score,
                        "E_value": e_value,
                        "KO_definition": ko_def,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No significant KofamScan hits found in %s", result_file)
        df = pd.DataFrame(columns=["gene_name", "KO", "thrshld", "score", "E_value", "KO_definition"])

    # Keep best hit per gene (highest score). Non-numeric scores sort
    # last under na_position="last", so the deduplication preference is
    # always for genuinely scored rows over coerce-failed ones.
    n_dropped_secondary = 0
    if not df.empty:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        n_bad_scores = df["score"].isna().sum()
        if n_bad_scores > 0:
            logger.warning(
                "%d/%d KofamScan hits had non-numeric scores in %s "
                "(coerced to NaN; treated as worst-scoring during "
                "deduplication)",
                n_bad_scores, len(df), result_file,
            )
        n_before = len(df)
        df = df.sort_values("score", ascending=False, na_position="last").drop_duplicates("gene_name", keep="first")
        n_dropped_secondary = n_before - len(df)

    logger.info(
        "Parsed %d significant KO annotations from %s "
        "(dropped %d secondary multi-domain hits — see parse_kofamscan "
        "docstring for the trade-off)",
        len(df), result_file, n_dropped_secondary,
    )
    return df


def annotate_with_kofam(
    expression_df: pd.DataFrame,
    kofam_df: pd.DataFrame,
    gene_col: str = "gene",
) -> pd.DataFrame:
    """Merge KofamScan annotations onto an expression/codon table.

    Args:
        expression_df: DataFrame with a gene identifier column.
        kofam_df: Parsed KofamScan results.
        gene_col: Name of the gene column in expression_df.

    Returns:
        Merged DataFrame with KO and KO_definition columns added.
    """
    merged = expression_df.merge(
        kofam_df[["gene_name", "KO", "KO_definition"]],
        left_on=gene_col,
        right_on="gene_name",
        how="left",
    )
    if "gene_name" in merged.columns and gene_col != "gene_name":
        merged.drop(columns=["gene_name"], inplace=True)
    return merged
