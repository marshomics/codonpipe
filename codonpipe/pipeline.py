"""Pipeline orchestrator: runs all analysis steps for a single genome or batch."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from codonpipe.modules.prokka import run_prokka
from codonpipe.modules.cogclassifier import run_cogclassifier, extract_ribosomal_proteins
from codonpipe.modules.kofamscan import run_kofamscan, parse_kofamscan, annotate_with_kofam
from codonpipe.modules.rscu import (
    compute_concatenated_rscu,
    compute_rscu_genome_summary,
    compute_rscu_per_gene,
    compute_codon_frequency_table,
    compute_enc,
    run_rscu_analysis,
)
from codonpipe.modules.expression import run_expression_analysis
from codonpipe.modules.cu_statistics import run_cu_statistics
from codonpipe.modules.enrichment import run_enrichment_analysis
from codonpipe.modules.advanced_analyses import run_advanced_analyses
from codonpipe.modules.bio_ecology import run_bio_ecology_analyses
from codonpipe.modules.codon_table_formats import generate_all_codon_tables
from codonpipe.modules.statistics import run_batch_statistics
from codonpipe.plotting.plots import (
    generate_single_genome_plots,
    generate_batch_plots,
)
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES
from codonpipe.utils.io import load_batch_table, write_tsv

logger = logging.getLogger("codonpipe")


def _validate_prokka_files(prokka_files: dict[str, Path]) -> None:
    """Validate that user-supplied Prokka files exist and are non-empty."""
    for key in ("faa", "ffn"):
        if key not in prokka_files:
            raise ValueError(
                f"Pre-existing Prokka files must include '{key}'. "
                f"Got keys: {list(prokka_files.keys())}"
            )
        p = prokka_files[key]
        if not p.is_file():
            raise FileNotFoundError(f"Prokka {key} file not found: {p}")
        if p.stat().st_size == 0:
            raise RuntimeError(f"Prokka {key} file is empty: {p}")

    n_proteins = sum(1 for line in prokka_files["faa"].open() if line.startswith(">"))
    logger.info(
        "Using pre-existing Prokka files: %s (%d proteins)",
        prokka_files["faa"], n_proteins,
    )


def run_single_genome(
    genome_fasta: Path,
    output_dir: Path,
    sample_id: str | None = None,
    cpus: int = 4,
    kingdom: str = "Bacteria",
    metagenome: bool = False,
    prokka_files: dict[str, Path] | None = None,
    cogs_file: Path | None = None,
    kofam_profile: Path | None = None,
    kofam_ko_list: Path | None = None,
    skip_kofamscan: bool = False,
    kofam_results_file: Path | None = None,
    skip_expression: bool = False,
    kegg_ko_pathway: Path | None = None,
    gff_file: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Run the full codon analysis pipeline on a single genome.

    Steps:
        1. Prokka ORF prediction (or use pre-existing files)
        2. COGclassifier -> ribosomal protein identification
        3. KofamScan functional annotation
        4. RSCU analysis (all CDS + ribosomal set)
        5. CU bias statistics (ENCprime, MILC via coRdon)
        6. Expression level prediction (MELP/CAI/Fop)
        7. Pathway enrichment (hypergeometric test on high/low expression genes)
        8. Advanced analyses (COA, S-value, neutrality, PR2, delta RSCU,
           tRNA-codon correlation, COG enrichment, gene length vs bias,
           ENC-ENC' difference)
        9. Biological/ecological analyses (HGT detection, growth rate
           prediction, translational selection, phage detection, strand
           asymmetry, operon co-adaptation)
       10. Codon usage tables in all standard formats (RSCU, counts,
           per-thousand, W values, adaptation weights, CBI)
       11. Publication-ready plots

    Args:
        genome_fasta: Path to genome assembly FASTA.
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier (default: filename stem).
        cpus: Number of threads for external tools.
        kingdom: Prokka kingdom (Bacteria, Archaea, Viruses).
        metagenome: Use Prokka --metagenome mode.
        prokka_files: Dict with pre-existing Prokka output paths.
            Required keys: 'faa' (amino acid FASTA), 'ffn' (nucleotide CDS FASTA).
            Optional keys: 'gff', 'gbk', 'tsv', etc.
            When provided, Prokka is skipped entirely.
        cogs_file: Custom ribosomal COGs file.
        kofam_profile: Path to KOfam profiles directory.
        kofam_ko_list: Path to KOfam ko_list file.
        skip_kofamscan: Skip KofamScan annotation step.
        kofam_results_file: Path to pre-computed KofamScan detail-tsv output.
            When provided, KofamScan execution is skipped and this file is
            parsed directly. Overrides skip_kofamscan.
        skip_expression: Skip R-based expression analysis.
        kegg_ko_pathway: User-supplied KO-to-pathway mapping TSV for offline use.
        gff_file: GFF3 annotation file for tRNA extraction (auto-detected from
            Prokka output if omitted).
        force: Rerun all steps.

    Returns:
        Dict of output paths organized by analysis type.
    """
    if sample_id is None:
        sample_id = genome_fasta.stem

    output_dir = Path(output_dir) / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    all_outputs: dict[str, Path] = {}

    logger.info("=" * 60)
    logger.info("Starting codon analysis pipeline for: %s", sample_id)
    logger.info("=" * 60)

    # ── Step 1: Prokka (or use pre-existing files) ─────────────────────
    if prokka_files is not None:
        logger.info("[Step 1/11] Using pre-existing Prokka files (skipping Prokka)")
        _validate_prokka_files(prokka_files)
        # Convert all values to Path objects
        prokka_out = {k: Path(v) for k, v in prokka_files.items()}
        all_outputs.update({f"prokka_{k}": v for k, v in prokka_out.items()})
    else:
        logger.info("[Step 1/11] Running Prokka gene prediction")
        prokka_out = run_prokka(
            genome_fasta, output_dir, sample_id,
            kingdom=kingdom, cpus=cpus, metagenome=metagenome, force=force,
        )
        all_outputs.update({f"prokka_{k}": v for k, v in prokka_out.items()})

    faa_path = prokka_out["faa"]
    ffn_path = prokka_out["ffn"]

    # ── Step 2: COGclassifier ───────────────────────────────────────────
    logger.info("[Step 2/11] Running COGclassifier for ribosomal protein identification")
    cog_result = run_cogclassifier(faa_path, output_dir, sample_id, cpus=cpus, force=force)
    all_outputs["cog_result"] = cog_result

    rp_outputs = extract_ribosomal_proteins(
        cog_result, faa_path, ffn_path, output_dir, sample_id, cogs_file=cogs_file,
    )
    all_outputs.update(rp_outputs)

    rp_ffn = rp_outputs.get("rp_ffn")
    rp_ids_file = rp_outputs.get("rp_ids")

    # ── Step 3: KofamScan ───────────────────────────────────────────────
    kofam_df = None
    if kofam_results_file is not None:
        logger.info("[Step 3/11] Loading pre-computed KofamScan results from %s", kofam_results_file)
        try:
            kofam_df = parse_kofamscan(kofam_results_file)
            kofam_out = output_dir / "kofamscan" / f"{sample_id}_kofam_parsed.tsv"
            kofam_out.parent.mkdir(parents=True, exist_ok=True)
            kofam_df.to_csv(kofam_out, sep="\t", index=False)
            all_outputs["kofam_parsed"] = kofam_out
        except Exception as e:
            logger.warning("Failed to parse pre-computed KofamScan results: %s. Continuing without annotations.", e)
    elif not skip_kofamscan:
        logger.info("[Step 3/11] Running KofamScan annotation")
        try:
            kofam_result = run_kofamscan(
                faa_path, output_dir, sample_id,
                profile_dir=kofam_profile, ko_list=kofam_ko_list,
                cpus=cpus, force=force,
            )
            kofam_df = parse_kofamscan(kofam_result)
            kofam_out = output_dir / "kofamscan" / f"{sample_id}_kofam_parsed.tsv"
            kofam_df.to_csv(kofam_out, sep="\t", index=False)
            all_outputs["kofam_parsed"] = kofam_out
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("KofamScan failed: %s. Continuing without annotations.", e)
    else:
        logger.info("[Step 3/11] Skipping KofamScan (--skip-kofamscan)")

    # ── Step 4: RSCU analysis ───────────────────────────────────────────
    logger.info("[Step 4/11] Running RSCU analysis")
    rscu_outputs = run_rscu_analysis(ffn_path, rp_ffn, output_dir, sample_id)
    all_outputs.update(rscu_outputs)

    # Load data for plotting
    freq_df = compute_codon_frequency_table(ffn_path)
    rscu_all = compute_rscu_genome_summary(ffn_path)
    rscu_rp = compute_concatenated_rscu(rp_ffn, min_length=0) if rp_ffn and rp_ffn.exists() else None
    rscu_gene_df = compute_rscu_per_gene(ffn_path)
    enc_df = compute_enc(ffn_path)

    # Annotate RSCU with KofamScan if available
    if kofam_df is not None and not kofam_df.empty:
        annotated = annotate_with_kofam(rscu_gene_df, kofam_df)
        ann_path = output_dir / "rscu" / f"{sample_id}_rscu_annotated.tsv"
        annotated.to_csv(ann_path, sep="\t", index=False)
        all_outputs["rscu_annotated"] = ann_path

    # ── Step 5: CU bias statistics (ENCprime, MILC) ────────────────────
    encprime_df = None
    milc_df = None
    if not skip_expression:
        logger.info("[Step 5/11] Computing CU bias statistics (ENCprime, MILC)")
        try:
            cu_stat_outputs = run_cu_statistics(
                ffn_path, output_dir, sample_id, force=force,
            )
            all_outputs.update(cu_stat_outputs)

            if "encprime" in cu_stat_outputs and cu_stat_outputs["encprime"].exists():
                encprime_df = pd.read_csv(cu_stat_outputs["encprime"], sep="\t")
            if "milc" in cu_stat_outputs and cu_stat_outputs["milc"].exists():
                milc_df = pd.read_csv(cu_stat_outputs["milc"], sep="\t")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("CU statistics failed: %s. Continuing.", e)
    else:
        logger.info("[Step 5/11] Skipping CU bias statistics (--skip-expression)")

    # ── Step 6: Expression analysis ─────────────────────────────────────
    expr_df = None
    if not skip_expression and rp_ids_file and rp_ids_file.exists():
        logger.info("[Step 6/11] Running expression level prediction (MELP/CAI/Fop)")
        try:
            expr_outputs = run_expression_analysis(
                ffn_path, rp_ids_file, output_dir, sample_id, force=force,
            )
            all_outputs.update(expr_outputs)

            if "expression_combined" in expr_outputs:
                expr_df = pd.read_csv(expr_outputs["expression_combined"], sep="\t")

                # Annotate with KofamScan
                if kofam_df is not None and not kofam_df.empty:
                    expr_ann = annotate_with_kofam(expr_df, kofam_df)
                    expr_ann_path = output_dir / "expression" / f"{sample_id}_expression_annotated.tsv"
                    expr_ann.to_csv(expr_ann_path, sep="\t", index=False)
                    all_outputs["expression_annotated"] = expr_ann_path
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Expression analysis failed: %s. Continuing.", e)
    elif skip_expression:
        logger.info("[Step 6/11] Skipping expression analysis (--skip-expression)")
    else:
        logger.info("[Step 6/11] Skipping expression analysis (no ribosomal proteins found)")

    # ── Step 7: Pathway enrichment ───────────────────────────────────────
    enrichment_results = {}
    if expr_df is not None and kofam_df is not None and not kofam_df.empty:
        logger.info("[Step 7/11] Running pathway enrichment (hypergeometric test)")
        try:
            enrich_outputs = run_enrichment_analysis(
                expr_df, kofam_df, output_dir, sample_id,
                kegg_ko_pathway_file=kegg_ko_pathway,
            )
            all_outputs.update(enrich_outputs)
            # Load enrichment results for plotting
            for key, path in enrich_outputs.items():
                if key.startswith("enrichment_") and path.exists():
                    enrichment_results[key] = pd.read_csv(path, sep="\t")
        except Exception as e:
            logger.warning("Pathway enrichment failed: %s. Continuing.", e)
    else:
        logger.info(
            "[Step 7/11] Skipping pathway enrichment (%s)",
            "no expression data" if expr_df is None else "no KofamScan annotations",
        )

    # ── Step 8: Advanced analyses ────────────────────────────────────────
    logger.info("[Step 8/11] Running advanced codon usage analyses")
    advanced_results = {}
    try:
        # Resolve GFF path: explicit > Prokka output > batch table
        resolved_gff = gff_file
        if resolved_gff is None and prokka_files is not None:
            gff_candidate = prokka_files.get("gff")
            if gff_candidate and Path(gff_candidate).exists():
                resolved_gff = Path(gff_candidate)
        if resolved_gff is None:
            # Try Prokka output directory
            prokka_gff = output_dir / "prokka" / f"{sample_id}.gff"
            if prokka_gff.exists():
                resolved_gff = prokka_gff

        adv_outputs = run_advanced_analyses(
            ffn_path=ffn_path,
            output_dir=output_dir,
            sample_id=sample_id,
            rscu_gene_df=rscu_gene_df,
            enc_df=enc_df,
            rscu_rp=rscu_rp,
            expr_df=expr_df,
            encprime_df=encprime_df,
            gff_path=resolved_gff,
            cog_result_tsv=all_outputs.get("cog_result"),
        )
        # Separate DataFrames from file paths for plotting
        for key, val in adv_outputs.items():
            if isinstance(val, Path):
                all_outputs[f"advanced_{key}"] = val
            elif isinstance(val, pd.DataFrame):
                advanced_results[key] = val
    except Exception as e:
        logger.warning("Advanced analyses failed: %s. Continuing.", e)

    # ── Step 9: Biological/ecological analyses ──────────────────────────
    logger.info("[Step 9/11] Running biological and ecological analyses")
    bio_ecology_results = {}
    try:
        bio_outputs = run_bio_ecology_analyses(
            ffn_path=ffn_path,
            output_dir=output_dir,
            sample_id=sample_id,
            rscu_gene_df=rscu_gene_df,
            enc_df=enc_df,
            expr_df=expr_df,
            rp_ids_file=rp_ids_file,
            cog_result_tsv=all_outputs.get("cog_result"),
            kofam_df=kofam_df,
            gff_path=resolved_gff,
        )
        for key, val in bio_outputs.items():
            if isinstance(val, Path):
                all_outputs[f"bio_{key}"] = val
            elif isinstance(val, pd.DataFrame):
                bio_ecology_results[key] = val
            elif isinstance(val, dict):
                bio_ecology_results[key] = val
    except Exception as e:
        logger.warning("Biological/ecological analyses failed: %s. Continuing.", e)

    # ── Step 10: Codon usage tables ──────────────────────────────────────
    logger.info("[Step 10/11] Generating codon usage tables in all standard formats")
    try:
        rp_ffn = rp_outputs.get("rp_ffn")
        table_outputs = generate_all_codon_tables(
            ffn_path=ffn_path,
            rp_ffn_path=rp_ffn,
            output_dir=output_dir,
            sample_id=sample_id,
            expr_df=expr_df,
            rp_ids_file=rp_ids_file,
        )
        all_outputs.update(table_outputs)
    except Exception as e:
        logger.warning("Codon usage table generation failed: %s. Continuing.", e)

    # ── Step 11: Plots ────────────────────────────────────────────────────
    logger.info("[Step 11/11] Generating publication-ready plots")
    plot_outputs = generate_single_genome_plots(
        sample_id, output_dir,
        freq_df=freq_df,
        rscu_all=rscu_all,
        rscu_rp=rscu_rp,
        rscu_gene_df=rscu_gene_df,
        enc_df=enc_df,
        expr_df=expr_df,
        encprime_df=encprime_df,
        milc_df=milc_df,
        enrichment_results=enrichment_results if enrichment_results else None,
        advanced_results=advanced_results if advanced_results else None,
        bio_ecology_results=bio_ecology_results if bio_ecology_results else None,
    )
    all_outputs.update(plot_outputs)

    # ── Summary ─────────────────────────────────────────────────────────
    _write_summary(all_outputs, output_dir, sample_id)
    logger.info("Pipeline complete for %s. Output: %s", sample_id, output_dir)

    return all_outputs


def run_batch(
    batch_table: Path,
    output_dir: Path,
    cpus: int = 4,
    parallel: int = 1,
    metadata_cols: list[str] | None = None,
    **kwargs,
) -> dict[str, Path]:
    """Run the pipeline on multiple genomes from a batch table.

    The batch table may include 'prokka_faa' and 'prokka_ffn' columns.
    For rows where both are populated, Prokka is skipped and those files
    are used directly. Rows with empty/missing values run Prokka normally.

    Args:
        batch_table: Path to TSV/CSV with genome_path column.
        output_dir: Base output directory.
        cpus: CPUs per sample.
        parallel: Number of samples to process in parallel.
        metadata_cols: Columns from batch table to use for comparative analyses.
        **kwargs: Additional arguments passed to run_single_genome.

    Returns:
        Dict of batch-level output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_batch_table(batch_table)
    n_samples = len(df)
    logger.info("Loaded batch table with %d genomes", n_samples)

    # Check for pre-existing Prokka columns
    has_prokka_cols = "prokka_faa" in df.columns and "prokka_ffn" in df.columns
    if has_prokka_cols:
        # Count rows where both are non-empty strings
        pre_mask = (
            df["prokka_faa"].fillna("").str.strip().astype(bool)
            & df["prokka_ffn"].fillna("").str.strip().astype(bool)
        )
        n_prerun = pre_mask.sum()
        logger.info(
            "%d/%d samples have pre-existing Prokka files (Prokka will be skipped for those)",
            n_prerun, n_samples,
        )

    # Detect metadata columns — exclude pipeline-specific columns
    pipeline_cols = {"genome_path", "sample_id", "prokka_faa", "prokka_ffn", "prokka_gff", "gff_path", "kofam_results"}
    if metadata_cols is None:
        metadata_cols = [c for c in df.columns if c not in pipeline_cols]
    logger.info("Metadata columns for comparative analysis: %s", metadata_cols or "(none)")

    # Run per-sample pipeline
    sample_outputs = {}

    def _build_kwargs_for_row(row: pd.Series) -> dict:
        """Build per-sample kwargs, including prokka_files if columns are present."""
        row_kwargs = dict(kwargs)

        # Check if this row has pre-existing Prokka files
        if has_prokka_cols:
            faa_val = str(row.get("prokka_faa", "") or "").strip()
            ffn_val = str(row.get("prokka_ffn", "") or "").strip()
            if faa_val and ffn_val:
                prokka_files = {
                    "faa": Path(faa_val),
                    "ffn": Path(ffn_val),
                }
                # Optional GFF
                gff_val = str(row.get("prokka_gff", "") or "").strip()
                if gff_val:
                    prokka_files["gff"] = Path(gff_val)
                row_kwargs["prokka_files"] = prokka_files

        # Per-sample GFF path from batch table (for tRNA extraction)
        gff_path_val = str(row.get("gff_path", "") or "").strip()
        if gff_path_val and "gff_file" not in row_kwargs:
            row_kwargs["gff_file"] = Path(gff_path_val)

        # Per-sample pre-computed KofamScan results from batch table
        kofam_res_val = str(row.get("kofam_results", "") or "").strip()
        if kofam_res_val and "kofam_results_file" not in row_kwargs:
            row_kwargs["kofam_results_file"] = Path(kofam_res_val)

        return row_kwargs

    if parallel > 1:
        logger.info("Processing %d genomes with %d parallel workers", n_samples, parallel)
        with ProcessPoolExecutor(max_workers=parallel) as executor:
            futures = {}
            for _, row in df.iterrows():
                sid = row["sample_id"]
                gpath = Path(row["genome_path"])
                row_kwargs = _build_kwargs_for_row(row)
                future = executor.submit(
                    run_single_genome,
                    genome_fasta=gpath,
                    output_dir=output_dir,
                    sample_id=sid,
                    cpus=cpus,
                    **row_kwargs,
                )
                futures[future] = sid

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                    sample_outputs[sid] = result
                    logger.info("Completed: %s (%d/%d)", sid, len(sample_outputs), n_samples)
                except Exception as e:
                    logger.error("Failed: %s — %s", sid, e)
    else:
        for idx, row in df.iterrows():
            sid = row["sample_id"]
            gpath = Path(row["genome_path"])
            row_kwargs = _build_kwargs_for_row(row)
            logger.info("Processing %s (%d/%d)", sid, idx + 1, n_samples)
            try:
                result = run_single_genome(
                    genome_fasta=gpath,
                    output_dir=output_dir,
                    sample_id=sid,
                    cpus=cpus,
                    **row_kwargs,
                )
                sample_outputs[sid] = result
            except Exception as e:
                logger.error("Failed: %s — %s", sid, e)

    if not sample_outputs:
        logger.error("No samples completed successfully")
        return {}

    # ── Batch-level analyses ────────────────────────────────────────────
    logger.info("Running batch-level comparative analyses")
    batch_outputs = _run_batch_analyses(
        df, sample_outputs, output_dir, metadata_cols,
    )

    return batch_outputs


def _run_batch_analyses(
    batch_df: pd.DataFrame,
    sample_outputs: dict[str, dict[str, Path]],
    output_dir: Path,
    metadata_cols: list[str],
) -> dict[str, Path]:
    """Combine per-sample results and run comparative analyses."""
    outputs = {}

    # Combine genome-level median RSCU
    rscu_rows = []
    for sid, paths in sample_outputs.items():
        rscu_path = paths.get("rscu_median")
        if rscu_path and rscu_path.exists():
            row = pd.read_csv(rscu_path, sep="\t")
            rscu_rows.append(row)

    if not rscu_rows:
        logger.warning("No RSCU data to combine for batch analysis")
        return outputs

    combined_rscu = pd.concat(rscu_rows, ignore_index=True)

    # Merge metadata from batch table (exclude pipeline-internal columns)
    pipeline_cols = {"genome_path", "sample_id", "prokka_faa", "prokka_ffn", "prokka_gff", "gff_path", "kofam_results"}
    meta_cols_to_merge = ["sample_id"] + [
        c for c in metadata_cols if c in batch_df.columns and c not in pipeline_cols
    ]
    if meta_cols_to_merge:
        combined_rscu = combined_rscu.merge(
            batch_df[meta_cols_to_merge], on="sample_id", how="left"
        )

    combined_path = output_dir / "combined_rscu.tsv"
    combined_rscu.to_csv(combined_path, sep="\t", index=False)
    outputs["combined_rscu"] = combined_path

    # Also combine ribosomal RSCU if available
    rp_rows = []
    for sid, paths in sample_outputs.items():
        rp_path = paths.get("rscu_ribosomal")
        if rp_path and rp_path.exists():
            row = pd.read_csv(rp_path, sep="\t")
            rp_rows.append(row)
    if rp_rows:
        combined_rp = pd.concat(rp_rows, ignore_index=True)
        if meta_cols_to_merge:
            combined_rp = combined_rp.merge(
                batch_df[meta_cols_to_merge], on="sample_id", how="left"
            )
        rp_combined_path = output_dir / "combined_rscu_ribosomal.tsv"
        combined_rp.to_csv(rp_combined_path, sep="\t", index=False)
        outputs["combined_rscu_ribosomal"] = rp_combined_path

    # Statistical analyses
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in combined_rscu.columns]
    effective_meta = [c for c in metadata_cols if c in combined_rscu.columns and c not in pipeline_cols]
    if effective_meta and rscu_cols:
        stats_outputs = run_batch_statistics(
            combined_rscu, output_dir,
            metadata_cols=effective_meta,
        )
        outputs.update(stats_outputs)

    # Collect Wilcoxon results for plotting
    wilcoxon_results = {}
    for key, path in outputs.items():
        if "wilcoxon" in key and isinstance(path, Path) and path.exists():
            wilcoxon_results[key] = pd.read_csv(path, sep="\t")

    # Batch plots
    plot_outputs = generate_batch_plots(
        combined_rscu, output_dir,
        metadata_cols=effective_meta,
        wilcoxon_results=wilcoxon_results if wilcoxon_results else None,
    )
    outputs.update(plot_outputs)

    return outputs


def _write_summary(outputs: dict[str, Path], output_dir: Path, sample_id: str) -> None:
    """Write a summary of all output files."""
    summary_path = output_dir / f"{sample_id}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"CodonPipe Analysis Summary — {sample_id}\n")
        f.write("=" * 50 + "\n\n")
        for key, path in sorted(outputs.items()):
            f.write(f"  {key:40s} {path}\n")
        f.write(f"\nTotal outputs: {len(outputs)}\n")
