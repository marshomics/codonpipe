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
from codonpipe.modules.mahal_clustering import run_mahal_clustering
from codonpipe.modules.cluster_stability import run_stability_analysis
from codonpipe.modules.bio_ecology import run_bio_ecology_analyses
from codonpipe.modules.codon_table_formats import generate_all_codon_tables
from codonpipe.modules.statistics import run_batch_statistics
from codonpipe.modules.comparative import run_comparative_analyses
from codonpipe.plotting.plots import (
    generate_single_genome_plots,
    generate_batch_plots,
    generate_pairwise_comparison_plots,
    plot_rscu_heatmap_rounded_comparison,
    _mahal_rscu_to_freq_df,
)
from codonpipe.plotting.comparative_plots import generate_comparative_plots
from codonpipe.utils.codon_tables import (
    RSCU_COLUMN_NAMES,
    COL_GENE, COL_ENC_DIFF, COL_ENCPRIME_RESIDUAL,
    COL_EXPRESSION_CLASS, COL_IN_MAHAL_CLUSTER,
    COL_MELP, COL_CAI, COL_FOP,
    COL_MELP_CLASS, COL_CAI_CLASS, COL_FOP_CLASS,
    EXPRESSION_METRICS, EXPRESSION_CLASS_COLS,
    RP_PREFIX,
)
from codonpipe.utils.io import load_batch_table, write_tsv

logger = logging.getLogger("codonpipe")

# Columns in the batch table that are consumed by the pipeline itself
# (not user metadata).  Defined once so run_batch() and _run_batch_analyses()
# stay in sync.
_PIPELINE_COLS = frozenset({
    "genome_path", "sample_id", "prokka_faa", "prokka_ffn",
    "prokka_gff", "gff_path", "kofam_results",
})


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
    skip_mahal: bool = False,
    mahal_min_k: int = 2,
    mahal_max_k: int = 8,
    mahal_distance_multiplier: float = 2.0,
    run_stability: bool = False,
    stability_bootstraps: int = 100,
    stability_multipliers: list[float] | None = None,
    auto_select_multiplier: bool = False,
    stability_core_threshold: float = 0.5,
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
        9. Mahalanobis clustering — COA-based Mahalanobis distance clustering
           to identify the translationally optimised gene cluster, using
           ribosomal proteins as anchor
       9s. (Optional) Bootstrap stability analysis — sweep multiplier grid,
           compute per-gene membership frequency, recommend optimal multiplier
       10. Biological/ecological analyses (HGT detection, growth rate
           prediction, translational selection, phage detection, strand
           asymmetry, operon co-adaptation)
       11. Codon usage tables in all standard formats (RSCU, counts,
           per-thousand, W values, adaptation weights, CBI) plus
           Mahalanobis-cluster-specific tables
       12. Publication-ready plots

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
        skip_mahal: Skip Mahalanobis codon usage clustering.
        mahal_min_k: Minimum Mahalanobis components to test (default 2).
        mahal_max_k: Maximum Mahalanobis components to test (default 8).
        mahal_distance_multiplier: Threshold = multiplier × median RP
            Mahalanobis distance (default 2.0).  Lower values produce a
            tighter cluster; higher values are more permissive.
        run_stability: Run bootstrap stability analysis on the Mahalanobis
            cluster (default False).  Sweeps a grid of multipliers and
            computes per-gene membership frequency to quantify robustness.
        stability_bootstraps: Number of bootstrap replicates per multiplier
            (default 100).
        stability_multipliers: Custom list of multiplier values to test.
            Defaults to [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0].
        auto_select_multiplier: When True and run_stability is True, use
            the stability-recommended multiplier instead of the user-supplied
            mahal_distance_multiplier.  The Mahalanobis clustering step is
            re-run with the recommended value.
        stability_core_threshold: Membership frequency threshold for a gene
            to be considered "core" in the stability analysis (default 0.5).
            Set to 0.9 for a high-confidence subset.
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
        logger.info("[Step 1/12] Using pre-existing Prokka files (skipping Prokka)")
        _validate_prokka_files(prokka_files)
        # Convert all values to Path objects
        prokka_out = {k: Path(v) for k, v in prokka_files.items()}
        all_outputs.update({f"prokka_{k}": v for k, v in prokka_out.items()})
    else:
        logger.info("[Step 1/12] Running Prokka gene prediction")
        prokka_out = run_prokka(
            genome_fasta, output_dir, sample_id,
            kingdom=kingdom, cpus=cpus, metagenome=metagenome, force=force,
        )
        all_outputs.update({f"prokka_{k}": v for k, v in prokka_out.items()})

    faa_path = prokka_out["faa"]
    ffn_path = prokka_out["ffn"]

    # ── Step 2: COGclassifier ───────────────────────────────────────────
    logger.info("[Step 2/12] Running COGclassifier for ribosomal protein identification")
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
        logger.info("[Step 3/12] Loading pre-computed KofamScan results from %s", kofam_results_file)
        try:
            kofam_df = parse_kofamscan(kofam_results_file)
            kofam_out = output_dir / "kofamscan" / f"{sample_id}_kofam_parsed.tsv"
            kofam_out.parent.mkdir(parents=True, exist_ok=True)
            kofam_df.to_csv(kofam_out, sep="\t", index=False)
            all_outputs["kofam_parsed"] = kofam_out
        except Exception as e:
            logger.warning("Failed to parse pre-computed KofamScan results: %s. Continuing without annotations.", e, exc_info=True)
    elif not skip_kofamscan:
        logger.info("[Step 3/12] Running KofamScan annotation")
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
            logger.warning("KofamScan failed: %s. Continuing without annotations.", e, exc_info=True)
    else:
        logger.info("[Step 3/12] Skipping KofamScan (--skip-kofamscan)")

    # ── Step 4: RSCU analysis ───────────────────────────────────────────
    logger.info("[Step 4/12] Running RSCU analysis")
    rscu_outputs = run_rscu_analysis(ffn_path, rp_ffn, output_dir, sample_id)
    all_outputs.update(rscu_outputs)

    # Load data for plotting
    freq_df = compute_codon_frequency_table(ffn_path)
    rscu_all = compute_rscu_genome_summary(ffn_path)
    rscu_rp = compute_concatenated_rscu(rp_ffn, min_length=0) if rp_ffn and rp_ffn.exists() else None
    if rscu_rp is None:
        logger.info("SKIPPED: ribosomal RSCU computation (no ribosomal protein sequences)")
    rscu_gene_df = compute_rscu_per_gene(ffn_path)
    enc_df = compute_enc(ffn_path)

    # Annotate RSCU with KofamScan if available
    if kofam_df is not None and not kofam_df.empty:
        annotated = annotate_with_kofam(rscu_gene_df, kofam_df)
        ann_path = output_dir / "rscu" / f"{sample_id}_rscu_annotated.tsv"
        annotated.to_csv(ann_path, sep="\t", index=False)
        all_outputs["rscu_annotated"] = ann_path
    else:
        logger.info("SKIPPED: RSCU KofamScan annotation (no KofamScan data)")

    # ── Step 5: CU bias statistics (ENCprime, MILC) ────────────────────
    encprime_df = None
    milc_df = None
    if not skip_expression:
        logger.info("[Step 5/12] Computing CU bias statistics (ENCprime, MILC)")
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
            logger.warning("CU statistics failed: %s. Continuing.", e, exc_info=True)
    else:
        logger.info("[Step 5/12] Skipping CU bias statistics (--skip-expression)")

    # ── Step 6: Expression analysis ─────────────────────────────────────
    # When Mahalanobis clustering is enabled (!skip_mahal), expression
    # scoring is deferred to Step 9b so that genes are scored against the
    # broader Mahalanobis cluster reference instead of RP-only IDs.
    # Expression runs here only when Mahalanobis is skipped (skip_mahal=True).
    expr_df = None
    if not skip_expression and rp_ids_file and rp_ids_file.exists() and skip_mahal:
        logger.info("[Step 6/12] Running expression level prediction (MELP/CAI/Fop)")
        try:
            expr_outputs = run_expression_analysis(
                ffn_path, rp_ids_file, output_dir, sample_id, force=force,
            )
            all_outputs.update(expr_outputs)

            if "expression_combined" in expr_outputs:
                expr_df = pd.read_csv(expr_outputs["expression_combined"], sep="\t")

                # Merge ENC' residual (ENC - ENC') as first-class column
                if enc_df is not None and encprime_df is not None:
                    try:
                        from codonpipe.modules.advanced_analyses import compute_enc_diff
                        enc_diff_df = compute_enc_diff(enc_df, encprime_df)
                        if not enc_diff_df.empty and COL_GENE in enc_diff_df.columns:
                            expr_df = expr_df.merge(
                                enc_diff_df[[COL_GENE, COL_ENC_DIFF]].rename(
                                    columns={COL_ENC_DIFF: COL_ENCPRIME_RESIDUAL}
                                ),
                                on=COL_GENE, how="left",
                            )
                            # Re-save the updated expression table
                            expr_df.to_csv(expr_outputs["expression_combined"], sep="\t", index=False)
                            logger.info("Merged ENC' residual into expression table (%d genes with values)",
                                        expr_df[COL_ENCPRIME_RESIDUAL].notna().sum())
                    except Exception as e:
                        logger.warning("Could not merge ENC' residual: %s", e)

                # Annotate with KofamScan
                if kofam_df is not None and not kofam_df.empty:
                    expr_ann = annotate_with_kofam(expr_df, kofam_df)
                    expr_ann_path = output_dir / "expression" / f"{sample_id}_expression_annotated.tsv"
                    expr_ann.to_csv(expr_ann_path, sep="\t", index=False)
                    all_outputs["expression_annotated"] = expr_ann_path
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("Expression analysis failed: %s. Continuing.", e, exc_info=True)
    elif skip_expression:
        logger.info("[Step 6/12] Skipping expression analysis (--skip-expression)")
    elif not skip_mahal:
        logger.info("[Step 6/12] Deferring expression analysis to Step 9b (Mahalanobis reference)")
    else:
        logger.info("[Step 6/12] Skipping expression analysis (no ribosomal proteins found)")

    # ── Step 7: Pathway enrichment (RP-based tiers) ─────────────────────
    enrichment_results = {}
    if skip_mahal and expr_df is not None and kofam_df is not None and not kofam_df.empty:
        logger.info("[Step 7/12] Running pathway enrichment (hypergeometric test, RP-based tiers)")
        try:
            enrich_outputs = run_enrichment_analysis(
                expr_df, kofam_df, output_dir, sample_id,
                kegg_ko_pathway_file=kegg_ko_pathway,
            )
            all_outputs.update(enrich_outputs)
            # Load enrichment results for plotting.  Prefix with "rp_" so
            # filenames and titles clearly distinguish RP-based enrichment
            # from ACE-based enrichment produced at step 9b.
            for key, path in enrich_outputs.items():
                if key.startswith("enrichment_") and path.exists():
                    enrichment_results[f"rp_{key}"] = pd.read_csv(path, sep="\t")
        except Exception as e:
            logger.warning("Pathway enrichment failed: %s. Continuing.", e, exc_info=True)
    elif not skip_mahal:
        logger.info("[Step 7/12] Deferring pathway enrichment to Step 9c (Mahalanobis reference)")
    else:
        logger.info(
            "[Step 7/12] Skipping pathway enrichment (%s)",
            "no expression data" if expr_df is None else "no KofamScan annotations",
        )

    # ── Step 8: Advanced analyses ────────────────────────────────────────
    logger.info("[Step 8/12] Running advanced codon usage analyses")
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
        logger.warning("Advanced analyses failed: %s. Continuing.", e, exc_info=True)

    if not advanced_results:
        logger.info("SKIPPED: advanced analyses produced no results")

    # ── Step 9: Mahalanobis clustering on COA space ──────────────────────────────
    mahal_results = {}
    mahal_cluster_gene_ids = None
    mahal_cluster_rscu = None
    if not skip_mahal:
        logger.info("[Step 9/12] Running Mahalanobis clustering on COA-projected RSCU")
        try:
            rp_rscu_df = None
            if rp_ffn and rp_ffn.exists():
                try:
                    rp_rscu_df = compute_rscu_per_gene(rp_ffn)
                except Exception as e:
                    logger.debug("Could not compute per-gene RSCU for RP sequences: %s", e)

            mahal_out = run_mahal_clustering(
                rscu_gene_df=rscu_gene_df,
                output_dir=output_dir,
                sample_id=sample_id,
                ffn_path=ffn_path,
                rp_ids_file=rp_ids_file,
                rp_rscu_df=rp_rscu_df,
                expr_df=expr_df,
                min_k=mahal_min_k,
                max_k=mahal_max_k,
                distance_multiplier=mahal_distance_multiplier,
            )

            # Separate file paths from in-memory objects, but keep
            # everything in mahal_results too so downstream steps can
            # access paths like mahal_cluster_ids_path directly.
            for key, val in mahal_out.items():
                mahal_results[key] = val
                if isinstance(val, Path):
                    all_outputs[f"mahal_{key}"] = val

            mahal_cluster_gene_ids = mahal_results.get("mahal_cluster_gene_ids")
            mahal_cluster_rscu = mahal_results.get("mahal_cluster_rscu")

            # Annotate expression table with Mahalanobis cluster membership
            if mahal_cluster_gene_ids and expr_df is not None and not expr_df.empty:
                try:
                    expr_df[COL_IN_MAHAL_CLUSTER] = expr_df[COL_GENE].isin(mahal_cluster_gene_ids)
                    logger.info(
                        "Annotated expression table with Mahalanobis cluster membership "
                        "(%d genes in RP cluster)",
                        expr_df[COL_IN_MAHAL_CLUSTER].sum(),
                    )
                except Exception as e:
                    logger.warning("Could not annotate expression table with Mahalanobis clusters: %s", e)
        except Exception as e:
            logger.warning("Mahalanobis clustering failed: %s. Continuing.", e, exc_info=True)
    else:
        logger.info("[Step 9/12] Skipping Mahalanobis clustering (--skip-mahal)")

    # ── Step 9s: Bootstrap stability analysis (optional) ───────────────────────
    stability_results = {}
    if run_stability and not skip_mahal and rscu_gene_df is not None:
        logger.info("[Step 9s/12] Running bootstrap cluster stability analysis")
        try:
            rp_rscu_df_stab = None
            if rp_ffn and rp_ffn.exists():
                try:
                    rp_rscu_df_stab = compute_rscu_per_gene(rp_ffn)
                except Exception:
                    pass

            stability_results = run_stability_analysis(
                rscu_gene_df=rscu_gene_df,
                output_dir=output_dir,
                sample_id=sample_id,
                ffn_path=ffn_path,
                rp_ids_file=rp_ids_file,
                rp_rscu_df=rp_rscu_df_stab,
                expr_df=expr_df,
                n_bootstraps=stability_bootstraps,
                multiplier_grid=stability_multipliers,
                core_threshold=stability_core_threshold,
            )

            for key, val in stability_results.items():
                if isinstance(val, Path):
                    all_outputs[f"stability_{key}"] = val

            # If auto-select is enabled and we got a recommendation, re-run
            # Mahalanobis clustering at the recommended multiplier.
            rec_mult = stability_results.get("recommended_multiplier")
            if (
                auto_select_multiplier
                and rec_mult is not None
                and abs(rec_mult - mahal_distance_multiplier) > 0.01
            ):
                logger.info(
                    "Auto-selecting recommended multiplier %.2f (was %.2f); "
                    "re-running Mahalanobis clustering",
                    rec_mult, mahal_distance_multiplier,
                )
                mahal_distance_multiplier = rec_mult

                rp_rscu_df_rerun = None
                if rp_ffn and rp_ffn.exists():
                    try:
                        rp_rscu_df_rerun = compute_rscu_per_gene(rp_ffn)
                    except Exception:
                        pass

                mahal_out = run_mahal_clustering(
                    rscu_gene_df=rscu_gene_df,
                    output_dir=output_dir,
                    sample_id=sample_id,
                    ffn_path=ffn_path,
                    rp_ids_file=rp_ids_file,
                    rp_rscu_df=rp_rscu_df_rerun,
                    expr_df=expr_df,
                    min_k=mahal_min_k,
                    max_k=mahal_max_k,
                    distance_multiplier=rec_mult,
                )
                mahal_results = {}
                for key, val in mahal_out.items():
                    mahal_results[key] = val
                    if isinstance(val, Path):
                        all_outputs[f"mahal_{key}"] = val
                mahal_cluster_gene_ids = mahal_results.get("mahal_cluster_gene_ids")
                mahal_cluster_rscu = mahal_results.get("mahal_cluster_rscu")

                if mahal_cluster_gene_ids and expr_df is not None and not expr_df.empty:
                    try:
                        expr_df[COL_IN_MAHAL_CLUSTER] = expr_df[COL_GENE].isin(mahal_cluster_gene_ids)
                    except Exception:
                        pass

        except Exception as e:
            logger.warning("Stability analysis failed: %s. Continuing.", e, exc_info=True)

    # ── Promote stability core set when available ────────────────────────────
    # When bootstrap stability analysis has been run, the core gene set
    # (genes above core_threshold membership frequency) replaces the
    # Mahalanobis cluster for MELP scoring, gRodon2, and enrichment.
    # The frequency-weighted RSCU replaces the distance-weighted RSCU.
    # Safeguard: only apply if core set has >= _MIN_CORE_FOR_OVERRIDE genes.
    _MIN_CORE_FOR_OVERRIDE = 30
    stability_core_ids = stability_results.get("core_gene_ids")
    stability_core_rscu = stability_results.get("core_rscu")
    stability_core_ids_path = stability_results.get("core_ids_path")

    if (
        stability_core_ids
        and len(stability_core_ids) >= _MIN_CORE_FOR_OVERRIDE
    ):
        logger.info(
            "Stability core set (%d genes, threshold=%.2f) replaces "
            "Mahalanobis cluster (%s genes) for MELP/gRodon2/enrichment",
            len(stability_core_ids),
            stability_core_threshold,
            len(mahal_cluster_gene_ids) if mahal_cluster_gene_ids else 0,
        )
        mahal_cluster_gene_ids = stability_core_ids
        if stability_core_ids_path:
            mahal_results["mahal_cluster_ids_path"] = stability_core_ids_path
        if stability_core_rscu is not None and not stability_core_rscu.empty:
            mahal_cluster_rscu = stability_core_rscu

        # Re-annotate expression table with updated cluster membership
        if expr_df is not None and not expr_df.empty:
            try:
                expr_df[COL_IN_MAHAL_CLUSTER] = expr_df[COL_GENE].isin(stability_core_ids)
            except Exception:
                pass
    elif stability_core_ids is not None and len(stability_core_ids) < _MIN_CORE_FOR_OVERRIDE:
        logger.warning(
            "Stability core set too small (%d < %d genes); keeping "
            "Mahalanobis cluster for MELP/gRodon2. Consider lowering "
            "--stability-core-threshold (currently %.2f).",
            len(stability_core_ids), _MIN_CORE_FOR_OVERRIDE,
            stability_core_threshold,
        )

    # Refresh the cluster IDs path reference for Step 9b
    mahal_cluster_ids_path = mahal_results.get("mahal_cluster_ids_path")

    # Save Mahalanobis cluster RSCU to rscu/ directory alongside genome and RP RSCU
    if mahal_cluster_rscu is not None and not mahal_cluster_rscu.empty:
        try:
            rscu_dir = output_dir / "rscu"
            rscu_dir.mkdir(parents=True, exist_ok=True)
            mahal_rscu_path = rscu_dir / f"{sample_id}_rscu_mahal_cluster.tsv"
            mahal_cluster_rscu.to_frame("RSCU").to_csv(mahal_rscu_path, sep="\t")
            all_outputs["rscu_mahal_cluster"] = mahal_rscu_path
            logger.info("Mahalanobis cluster RSCU table saved to rscu/ directory")
        except Exception as e:
            logger.warning("Could not save Mahalanobis cluster RSCU to rscu/: %s", e)

    # ── Step 9b: Re-run expression scoring with Mahalanobis cluster as reference ─
    # Step 6 scored genes against RP-only IDs.  The Mahalanobis cluster provides a
    # broader, biologically grounded reference set (RP + co-clustered genes).
    # Re-run MELP/CAI/Fop via coRdon using Mahalanobis cluster gene IDs, then
    # promote the Mahalanobis-based classification as the primary expression_class.
    mahal_cluster_ids_path = mahal_results.get("mahal_cluster_ids_path")
    if (
        mahal_cluster_gene_ids
        and mahal_cluster_ids_path
        and not skip_expression
    ):
        logger.info(
            "[Step 9b/12] Re-running expression analysis with Mahalanobis cluster "
            "(%d genes) as reference",
            len(mahal_cluster_gene_ids),
        )
        try:
            mahal_expr_outputs = run_expression_analysis(
                ffn_path, mahal_cluster_ids_path, output_dir, sample_id,
                force=True,  # overwrite the RP-based expression files
            )
            all_outputs.update(mahal_expr_outputs)

            if "expression_combined" in mahal_expr_outputs:
                mahal_expr_df = pd.read_csv(mahal_expr_outputs["expression_combined"], sep="\t")

                # Preserve RP-based scores as secondary columns
                if expr_df is not None and not expr_df.empty:
                    rp_cols_to_keep = {}
                    for col in (*EXPRESSION_METRICS, *EXPRESSION_CLASS_COLS, COL_EXPRESSION_CLASS):
                        if col in expr_df.columns:
                            rp_cols_to_keep[col] = f"{RP_PREFIX}{col}"

                    rp_backup = expr_df[[COL_GENE] + list(rp_cols_to_keep.keys())].rename(
                        columns=rp_cols_to_keep
                    )
                    mahal_expr_df = mahal_expr_df.merge(rp_backup, on=COL_GENE, how="left")

                # Add Mahalanobis cluster membership flag
                mahal_expr_df[COL_IN_MAHAL_CLUSTER] = mahal_expr_df[COL_GENE].isin(mahal_cluster_gene_ids)

                # Merge ENC' residual if available
                if enc_df is not None and encprime_df is not None:
                    try:
                        from codonpipe.modules.advanced_analyses import compute_enc_diff
                        enc_diff_df = compute_enc_diff(enc_df, encprime_df)
                        if not enc_diff_df.empty and COL_GENE in enc_diff_df.columns:
                            mahal_expr_df = mahal_expr_df.merge(
                                enc_diff_df[[COL_GENE, COL_ENC_DIFF]].rename(
                                    columns={COL_ENC_DIFF: COL_ENCPRIME_RESIDUAL}
                                ),
                                on=COL_GENE, how="left",
                            )
                    except Exception as e:
                        logger.warning("Could not merge ENC' residual: %s", e)

                # Annotate with KofamScan
                if kofam_df is not None and not kofam_df.empty:
                    expr_ann = annotate_with_kofam(mahal_expr_df, kofam_df)
                    expr_ann_path = output_dir / "expression" / f"{sample_id}_expression_annotated.tsv"
                    expr_ann.to_csv(expr_ann_path, sep="\t", index=False)
                    all_outputs["expression_annotated"] = expr_ann_path

                # Save and promote
                mahal_expr_df.to_csv(mahal_expr_outputs["expression_combined"], sep="\t", index=False)
                expr_df = mahal_expr_df
                logger.info(
                    "Expression analysis re-scored with Mahalanobis cluster reference "
                    "(%d genes); expression_class now Mahalanobis-based",
                    len(expr_df),
                )
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(
                "Mahalanobis-based expression re-scoring failed: %s. "
                "Keeping RP-based expression scores.", e, exc_info=True,
            )

    # Fallback: if Mahalanobis-based expression wasn't produced, run RP-based
    if not skip_mahal and expr_df is None and not skip_expression and rp_ids_file and rp_ids_file.exists():
        logger.info("[Step 9b fallback] Mahalanobis unavailable; running RP-based expression analysis")
        try:
            expr_outputs = run_expression_analysis(
                ffn_path, rp_ids_file, output_dir, sample_id, force=force,
            )
            all_outputs.update(expr_outputs)
            if "expression_combined" in expr_outputs:
                expr_df = pd.read_csv(expr_outputs["expression_combined"], sep="\t")
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning("RP-based expression analysis failed: %s. Continuing.", e, exc_info=True)

    # ── Step 9c: Re-run pathway enrichment on Mahalanobis-based expression tiers ─
    # Step 7 enrichment used RP-only expression tiers.  Now that genes are
    # re-scored against the Mahalanobis cluster, re-run enrichment so downstream
    # plots and tables reflect the Mahalanobis-derived view.
    if (
        mahal_cluster_gene_ids
        and expr_df is not None
        and kofam_df is not None
        and not kofam_df.empty
    ):
        logger.info("[Step 9c/12] Re-running pathway enrichment on Mahalanobis-based expression tiers")
        try:
            mahal_enrich_outputs = run_enrichment_analysis(
                expr_df, kofam_df, output_dir, sample_id,
                kegg_ko_pathway_file=kegg_ko_pathway,
                output_subdir="enrichment_mahal",
            )
            all_outputs.update({
                f"mahal_{k}": v for k, v in mahal_enrich_outputs.items()
            })
            # Load Mahalanobis enrichment results for plotting, prefixed with "mahal_"
            for key, path in mahal_enrich_outputs.items():
                if key.startswith("enrichment_") and path.exists():
                    enrichment_results[f"mahal_{key}"] = pd.read_csv(path, sep="\t")
        except Exception as e:
            logger.warning("Mahalanobis-tier pathway enrichment failed: %s. Continuing.", e, exc_info=True)

    # Fallback: if Mahalanobis-based enrichment wasn't produced, run RP-based
    _has_mahal_enrichment = any(k.startswith("mahal_enrichment") for k in enrichment_results)
    if not skip_mahal and not _has_mahal_enrichment and expr_df is not None and kofam_df is not None and not kofam_df.empty:
        logger.info("[Step 9c fallback] Mahalanobis enrichment unavailable; running RP-based enrichment analysis")
        try:
            enrich_outputs = run_enrichment_analysis(
                expr_df, kofam_df, output_dir, sample_id,
                kegg_ko_pathway_file=kegg_ko_pathway,
            )
            all_outputs.update(enrich_outputs)
            for key, path in enrich_outputs.items():
                if key.startswith("enrichment_") and path.exists():
                    enrichment_results[f"rp_{key}"] = pd.read_csv(path, sep="\t")
        except Exception as e:
            logger.warning("RP-based pathway enrichment failed: %s. Continuing.", e, exc_info=True)

    # ── Step 9d: Recompute S-value, delta RSCU, COG enrichment with Mahalanobis ──
    # Step 8 computed these against RP-based tiers / genome-average baseline.
    # Now recompute with Mahalanobis cluster as both the expression reference
    # (tiers already updated in 9b) and as the RSCU baseline for delta RSCU.
    if mahal_cluster_rscu is not None and not mahal_cluster_rscu.empty:
        try:
            from codonpipe.modules.advanced_analyses import (
                compute_s_value, compute_delta_rscu, compute_cog_enrichment,
            )
            rscu_mahal_dict = mahal_cluster_rscu.to_dict()
            adv_dir = output_dir / "advanced"

            logger.info("[Step 9d/12] Recomputing S-value with Mahalanobis-cluster RSCU reference")
            s_val_mahal_df = compute_s_value(rscu_gene_df, rscu_rp,
                                            rscu_mahal_cluster=rscu_mahal_dict)
            if not s_val_mahal_df.empty:
                s_val_path = adv_dir / f"{sample_id}_s_value.tsv"
                s_val_mahal_df.to_csv(s_val_path, sep="\t", index=False)
                all_outputs["advanced_s_value_path"] = s_val_path
                advanced_results["s_value"] = s_val_mahal_df
                logger.info("S-value recomputed against Mahalanobis-cluster reference (%d genes)", len(s_val_mahal_df))

            # Delta RSCU with Mahalanobis-based expression tiers (genome-avg baseline)
            if expr_df is not None:
                for class_col in [COL_EXPRESSION_CLASS]:
                    if class_col in expr_df.columns:
                        delta_df = compute_delta_rscu(rscu_gene_df, expr_df, class_col)
                        if not delta_df.empty:
                            metric = class_col.replace("_class", "")
                            out_path = adv_dir / f"{sample_id}_delta_rscu_{metric}.tsv"
                            delta_df.to_csv(out_path, sep="\t", index=False)
                            all_outputs[f"advanced_delta_rscu_{metric}_path"] = out_path
                            advanced_results[f"delta_rscu_{metric}"] = delta_df

            # Delta RSCU with Mahalanobis cluster RSCU as the baseline
            # (deviation of high-expression genes from Mahalanobis cluster)
            if expr_df is not None:
                for class_col in [COL_EXPRESSION_CLASS, COL_CAI_CLASS, COL_MELP_CLASS, COL_FOP_CLASS]:
                    if class_col in expr_df.columns:
                        delta_mahal_df = compute_delta_rscu(
                            rscu_gene_df, expr_df, class_col,
                            rscu_reference=rscu_mahal_dict,
                            reference_label="mahal_cluster",
                        )
                        if not delta_mahal_df.empty:
                            metric = class_col.replace("_class", "")
                            out_path = adv_dir / f"{sample_id}_delta_rscu_{metric}_mahal_ref.tsv"
                            delta_mahal_df.to_csv(out_path, sep="\t", index=False)
                            all_outputs[f"advanced_delta_rscu_{metric}_mahal_ref_path"] = out_path
                            advanced_results[f"delta_rscu_{metric}_mahal_ref"] = delta_mahal_df

            # Preserve RP-based COG enrichment before overwriting
            if "cog_enrichment" in advanced_results:
                advanced_results["cog_enrichment_rp"] = advanced_results["cog_enrichment"]

            # Re-run COG enrichment with Mahalanobis-based expression tiers
            cog_tsv = all_outputs.get("cog_result")
            if cog_tsv and Path(cog_tsv).exists() and expr_df is not None:
                cog_enrich_mahal = compute_cog_enrichment(Path(cog_tsv), expr_df)
                if not cog_enrich_mahal.empty:
                    cog_path = adv_dir / f"{sample_id}_cog_enrichment_mahal.tsv"
                    cog_enrich_mahal.to_csv(cog_path, sep="\t", index=False)
                    all_outputs["advanced_cog_enrichment_mahal_path"] = cog_path
                    advanced_results["cog_enrichment_mahal"] = cog_enrich_mahal
                    logger.info("COG enrichment recomputed with Mahalanobis-based expression tiers")

        except Exception as e:
            logger.warning("Mahalanobis-aware metric recomputation failed: %s. Keeping RP-based values.", e)

    # ── Step 10: Biological/ecological analyses ─────────────────────────
    logger.info("[Step 10/12] Running biological and ecological analyses")
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
            mahal_cluster_rscu=mahal_cluster_rscu,
            mahal_cluster_gene_ids=mahal_cluster_gene_ids,
            mahal_cluster_ids_path=mahal_cluster_ids_path,
        )
        # Normalize keys and flatten nested dicts for plotting compatibility
        _bio_key_map = {
            "hgt_candidates": "hgt",
            "growth_rate_prediction": "growth_rate",
            "phage_mobile_elements": "phage_mobile",
        }
        for key, val in bio_outputs.items():
            if isinstance(val, Path):
                all_outputs[f"bio_{key}"] = val
            elif key == "translational_selection" and isinstance(val, dict):
                # Flatten nested translational_selection outputs
                for sub_key, sub_val in val.items():
                    if isinstance(sub_val, pd.DataFrame):
                        bio_ecology_results[sub_key] = sub_val
                    elif isinstance(sub_val, Path):
                        all_outputs[f"bio_trans_sel_{sub_key}"] = sub_val
            elif isinstance(val, pd.DataFrame):
                norm_key = _bio_key_map.get(key, key)
                bio_ecology_results[norm_key] = val
            elif isinstance(val, dict):
                norm_key = _bio_key_map.get(key, key)
                bio_ecology_results[norm_key] = val
    except Exception as e:
        logger.warning("Biological/ecological analyses failed: %s. Continuing.", e, exc_info=True)

    if not bio_ecology_results:
        logger.info("SKIPPED: bio/ecology analyses produced no results")

    # ── Step 11: Codon usage tables ──────────────────────────────────────
    logger.info("[Step 11/12] Generating codon usage tables in all standard formats")
    try:
        rp_ffn = rp_outputs.get("rp_ffn")
        table_outputs = generate_all_codon_tables(
            ffn_path=ffn_path,
            rp_ffn_path=rp_ffn,
            output_dir=output_dir,
            sample_id=sample_id,
            expr_df=expr_df,
            rp_ids_file=rp_ids_file,
            mahal_cluster_gene_ids=mahal_cluster_gene_ids,
        )
        all_outputs.update(table_outputs)
    except Exception as e:
        logger.warning("Codon usage table generation failed: %s. Continuing.", e, exc_info=True)

    # ── Step 12: Plots ────────────────────────────────────────────────────
    logger.info("[Step 12/12] Generating publication-ready plots")
    try:
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
            gff_path=resolved_gff,
            mahal_cluster_rscu=mahal_cluster_rscu,
            mahal_cluster_size=len(mahal_cluster_gene_ids) if mahal_cluster_gene_ids else None,
            mahal_cluster_gene_ids=mahal_cluster_gene_ids,
            mahal_coa_coords=mahal_results.get("mahal_coa_coords"),
            coa_inertia=mahal_results.get("mahal_coa_inertia"),
        )
        all_outputs.update(plot_outputs)
    except Exception as e:
        logger.warning("Plot generation failed: %s. Continuing.", e, exc_info=True)

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
    condition_col: str | None = None,
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
        condition_col: Column name in the batch table designating experimental
            conditions.  When provided, condition-aware within- and between-
            condition comparative analyses are generated in addition to the
            standard batch analyses.
        **kwargs: Additional arguments passed to run_single_genome.

    Returns:
        Dict of batch-level output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_batch_table(batch_table)
    n_samples = len(df)
    logger.info("Loaded batch table with %d genomes", n_samples)

    # Validate condition column
    if condition_col and condition_col not in df.columns:
        logger.error("Condition column '%s' not found in batch table (available: %s)",
                      condition_col, list(df.columns))
        raise ValueError(f"Condition column '{condition_col}' not found in batch table")
    if condition_col:
        n_cond = df[condition_col].nunique()
        logger.info("Condition column '%s': %d unique conditions — %s",
                     condition_col, n_cond, list(df[condition_col].unique()))

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
    pipeline_cols = _PIPELINE_COLS
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
        for i, (idx, row) in enumerate(df.iterrows()):
            sid = row["sample_id"]
            gpath = Path(row["genome_path"])
            row_kwargs = _build_kwargs_for_row(row)
            logger.info("Processing %s (%d/%d)", sid, i + 1, n_samples)
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
        condition_col=condition_col,
    )

    return batch_outputs


def _run_batch_analyses(
    batch_df: pd.DataFrame,
    sample_outputs: dict[str, dict[str, Path]],
    output_dir: Path,
    metadata_cols: list[str],
    condition_col: str | None = None,
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
    pipeline_cols = _PIPELINE_COLS
    meta_cols_to_merge = ["sample_id"] + [
        c for c in metadata_cols if c in batch_df.columns and c not in pipeline_cols
    ]
    if meta_cols_to_merge:
        combined_rscu = combined_rscu.merge(
            batch_df[meta_cols_to_merge], on="sample_id", how="left"
        )

    batch_rscu_dir = output_dir / "batch_rscu"
    batch_rscu_dir.mkdir(parents=True, exist_ok=True)

    combined_path = batch_rscu_dir / "combined_rscu.tsv"
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
        rp_combined_path = batch_rscu_dir / "combined_rscu_ribosomal.tsv"
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

    # ── Pairwise qualitative comparison plots (works with any ≥2 genomes) ──
    if len(sample_outputs) >= 2:
        try:
            pairwise_outputs = generate_pairwise_comparison_plots(
                sample_outputs=sample_outputs,
                combined_rscu=combined_rscu,
                output_dir=output_dir,
            )
            outputs.update(pairwise_outputs)
        except Exception as e:
            logger.warning("Pairwise comparison plots failed: %s", e)

    # ── Comparative RSCU heatmaps for exactly 2 genomes (3 versions) ──
    if len(sample_outputs) == 2:
        comp_plot_dir = output_dir / "batch_pairwise" / "plots"
        comp_plot_dir.mkdir(parents=True, exist_ok=True)

        _comp_variants = [
            ("genome", "All CDS", "rscu_median", False),
            ("ribosomal", "Ribosomal Proteins", "rscu_ribosomal", False),
            ("mahal", "Mahalanobis cluster", "rscu_mahal_cluster", True),
        ]
        for suffix, label, key, is_index_based in _comp_variants:
            try:
                comp_freq_dfs: dict[str, pd.DataFrame] = {}
                for sid, sout in sample_outputs.items():
                    p = sout.get(key)
                    if p and Path(p).exists():
                        if is_index_based:
                            raw = pd.read_csv(p, sep="\t", index_col=0)["RSCU"]
                        else:
                            raw = pd.read_csv(p, sep="\t")
                            # Column-based: single row with RSCU column names
                            if len(raw) > 0:
                                raw = raw.iloc[0].drop("sample_id", errors="ignore")
                        comp_freq_dfs[sid] = _mahal_rscu_to_freq_df(raw)
                if len(comp_freq_dfs) == 2:
                    comp_p = comp_plot_dir / f"rscu_heatmap_comparison_{suffix}"
                    plot_rscu_heatmap_rounded_comparison(
                        comp_freq_dfs, comp_p,
                        title=f"RSCU Comparison — {label}",
                    )
                    outputs[f"rscu_heatmap_comparison_{suffix}"] = comp_p.with_suffix(".png")
                    logger.info("Comparative RSCU heatmap (%s) saved", suffix)
                else:
                    logger.info(
                        "SKIPPED: comparative RSCU heatmap (%s) — data available "
                        "for %d/2 samples", suffix, len(comp_freq_dfs),
                    )
            except Exception as e:
                logger.warning("Comparative RSCU heatmap (%s) failed: %s", suffix, e)

    # ── Condition-aware comparative analyses ──────────────────────────
    if condition_col:
        logger.info("Running condition-aware comparative analyses (condition_col='%s')", condition_col)
        try:
            metrics_df, comp_outputs = run_comparative_analyses(
                sample_outputs=sample_outputs,
                batch_df=batch_df,
                output_dir=output_dir,
                condition_col=condition_col,
                metadata_cols=effective_meta,
            )
            outputs.update(comp_outputs)

            # Load statistical results for plotting
            between_tests_df = None
            rscu_tests_df = None
            rscu_disp_df = None
            perm_result = None

            if "between_condition_tests" in comp_outputs:
                p = comp_outputs["between_condition_tests"]
                if p.exists():
                    between_tests_df = pd.read_csv(p, sep="\t")

            if "between_condition_rscu_tests" in comp_outputs:
                p = comp_outputs["between_condition_rscu_tests"]
                if p.exists():
                    rscu_tests_df = pd.read_csv(p, sep="\t")

            if "within_condition_rscu_dispersion" in comp_outputs:
                p = comp_outputs["within_condition_rscu_dispersion"]
                if p.exists():
                    rscu_disp_df = pd.read_csv(p, sep="\t")

            if "permanova_rscu" in comp_outputs:
                p = comp_outputs["permanova_rscu"]
                if p.exists():
                    perm_df = pd.read_csv(p, sep="\t")
                    if len(perm_df):
                        perm_result = perm_df.iloc[0].to_dict()

            # Load bio/ecology comparison data for plotting
            hgt_burden = None
            strand_asym_patterns_df = None
            optimal_codons_df = None
            gc3_gc12_df = None

            if "between_condition_hgt_burden" in comp_outputs:
                # Re-derive from comparative module (dict not serialisable to TSV)
                from codonpipe.modules.comparative import between_condition_hgt_burden
                try:
                    hgt_burden = between_condition_hgt_burden(
                        sample_outputs, metrics_df, condition_col,
                    )
                except Exception as e:
                    logger.debug("Could not compute HGT burden comparison: %s", e)

            if "between_condition_strand_asymmetry_patterns" in comp_outputs:
                p = comp_outputs["between_condition_strand_asymmetry_patterns"]
                if p.exists():
                    strand_asym_patterns_df = pd.read_csv(p, sep="\t")

            if "between_condition_optimal_codons" in comp_outputs:
                p = comp_outputs["between_condition_optimal_codons"]
                if p.exists():
                    optimal_codons_df = pd.read_csv(p, sep="\t")

            if "between_condition_gc3_gc12" in comp_outputs:
                p = comp_outputs["between_condition_gc3_gc12"]
                if p.exists():
                    gc3_gc12_df = pd.read_csv(p, sep="\t")

            # Load expression-class RSCU test results
            rp_rscu_tests_df = None
            he_rscu_tests_df = None
            enrichment_comp_df = None

            if "between_condition_ribosomal_rscu" in comp_outputs:
                p = comp_outputs["between_condition_ribosomal_rscu"]
                if p.exists():
                    rp_rscu_tests_df = pd.read_csv(p, sep="\t")

            if "between_condition_high_expression_rscu" in comp_outputs:
                p = comp_outputs["between_condition_high_expression_rscu"]
                if p.exists():
                    he_rscu_tests_df = pd.read_csv(p, sep="\t")

            if "between_condition_enrichment_comparison" in comp_outputs:
                p = comp_outputs["between_condition_enrichment_comparison"]
                if p.exists():
                    enrichment_comp_df = pd.read_csv(p, sep="\t")

            # Generate comparative plots
            logger.info("Generating condition-aware comparative plots")
            comp_plot_outputs = generate_comparative_plots(
                metrics_df=metrics_df,
                output_dir=output_dir,
                condition_col=condition_col,
                between_tests_df=between_tests_df,
                rscu_tests_df=rscu_tests_df,
                rscu_disp_df=rscu_disp_df,
                perm_result=perm_result,
                hgt_burden=hgt_burden,
                strand_asym_patterns_df=strand_asym_patterns_df,
                optimal_codons_df=optimal_codons_df,
                gc3_gc12_df=gc3_gc12_df,
                rp_rscu_tests_df=rp_rscu_tests_df,
                he_rscu_tests_df=he_rscu_tests_df,
                enrichment_comp_df=enrichment_comp_df,
            )
            outputs.update(comp_plot_outputs)
            logger.info("Condition-aware comparative analyses complete: %d outputs",
                        len(comp_outputs) + len(comp_plot_outputs))
        except Exception as e:
            logger.warning("Condition-aware comparative analyses failed: %s. Continuing.", e, exc_info=True)
    else:
        logger.info("No condition column specified; skipping condition-aware analyses")

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
