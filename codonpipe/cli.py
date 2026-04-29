"""Command-line interface for CodonPipe."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from codonpipe import __version__
from codonpipe.utils.logging import setup_logger


@click.group()
@click.version_option(__version__, prog_name="codonpipe")
def main():
    """CodonPipe: End-to-end codon usage analysis for microbial genomes."""
    pass


@main.command()
@click.argument("genome", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path),
              help="Output directory.")
@click.option("-s", "--sample-id", default=None, help="Sample identifier (default: filename stem).")
@click.option("-t", "--threads", default=4, type=int, show_default=True,
              help="Number of CPU threads for external tools.")
@click.option("--kingdom", default="Bacteria", show_default=True,
              type=click.Choice(["Bacteria", "Archaea", "Viruses"]),
              help="Prokka --kingdom flag.")
@click.option("--metagenome", is_flag=True, help="Use Prokka --metagenome mode.")
@click.option("--prokka-faa", type=click.Path(exists=True, path_type=Path), default=None,
              help="Pre-existing Prokka .faa file (amino acid sequences). Skips Prokka if both --prokka-faa and --prokka-ffn are provided.")
@click.option("--prokka-ffn", type=click.Path(exists=True, path_type=Path), default=None,
              help="Pre-existing Prokka .ffn file (nucleotide CDS sequences). Skips Prokka if both --prokka-faa and --prokka-ffn are provided.")
@click.option("--prokka-gff", type=click.Path(exists=True, path_type=Path), default=None,
              help="Pre-existing Prokka .gff file (optional, carried through to outputs).")
@click.option("--gff", "gff_file", type=click.Path(exists=True, path_type=Path), default=None,
              help="GFF3 annotation file for tRNA extraction (auto-detected from Prokka if omitted).")
@click.option("--cogs-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Custom ribosomal COG accessions file (one per line).")
@click.option("--kofam-profile", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to KOfam profile directory.")
@click.option("--kofam-ko-list", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to KOfam ko_list file.")
@click.option("--skip-kofamscan", is_flag=True, help="Skip KofamScan annotation.")
@click.option("--kofam-results", type=click.Path(exists=True, path_type=Path), default=None,
              help="Pre-computed KofamScan detail-tsv results file. When provided, KofamScan is skipped and this file is parsed directly.")
@click.option("--skip-expression", is_flag=True,
              help="Skip all R-based analyses (MELP/CAI/Fop/ENCprime/MILC).")
@click.option("--skip-mahal", is_flag=True,
              help="Skip Mahalanobis clustering of codon usage.")
@click.option("--skip-gsea", is_flag=True,
              help="Skip pre-ranked Gene Set Enrichment Analysis.")
@click.option("--gsea-permutations", type=int, default=1000, show_default=True,
              help="Number of permutations for GSEA p-value estimation.")
@click.option("--skip-codon-optimization", is_flag=True,
              help="Skip the codon-optimization comparison step (genome / RP "
                   "/ Mahal-cluster reference frames). Step 12 of the "
                   "13-step per-genome pipeline; output goes to "
                   "<sample_dir>/codon_optimization/. Skipping doesn't "
                   "affect any other step.")
@click.option("--mahal-min-k", type=int, default=2, show_default=True,
              help="Minimum number of Mahalanobis clustering components to test.")
@click.option("--mahal-max-k", type=int, default=8, show_default=True,
              help="Maximum number of Mahalanobis clustering components to test.")
@click.option("--mahal-distance-multiplier", type=float, default=2.0, show_default=True,
              help="Mahalanobis cluster radius as a multiplier of the median RP distance. "
                   "Lower values (e.g. 1.5) produce a tighter cluster; higher values (e.g. 3.0) are more permissive.")
@click.option("--run-stability", is_flag=True,
              help="Run bootstrap stability analysis on the Mahalanobis cluster. "
                   "Sweeps a grid of multipliers and reports per-gene membership frequency.")
@click.option("--stability-bootstraps", type=int, default=100, show_default=True,
              help="Number of bootstrap replicates per multiplier for stability analysis.")
@click.option("--stability-multipliers", type=str, default=None,
              help="Comma-separated multiplier values to test (e.g. '1.0,1.5,2.0,2.5,3.0'). "
                   "Defaults to 1.0,1.25,1.5,1.75,2.0,2.5,3.0.")
@click.option("--auto-select-multiplier", is_flag=True,
              help="Automatically use the stability-recommended multiplier instead of "
                   "--mahal-distance-multiplier. Requires --run-stability.")
@click.option("--stability-core-threshold", type=float, default=0.5, show_default=True,
              help="Membership frequency threshold for a gene to be 'core'. "
                   "0.5 = majority-rule consensus; 0.9 = high-confidence subset.")
@click.option("--kegg-ko-pathway", type=click.Path(exists=True, path_type=Path), default=None,
              help="KO-to-pathway mapping TSV for offline enrichment (auto-downloaded from KEGG if omitted).")
@click.option("--force", is_flag=True, help="Overwrite existing outputs.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--log-file", type=click.Path(path_type=Path), default=None,
              help="Write log to file.")
def run(
    genome: Path,
    output_dir: Path,
    sample_id: str | None,
    threads: int,
    kingdom: str,
    metagenome: bool,
    prokka_faa: Path | None,
    prokka_ffn: Path | None,
    prokka_gff: Path | None,
    gff_file: Path | None,
    cogs_file: Path | None,
    kofam_profile: Path | None,
    kofam_ko_list: Path | None,
    skip_kofamscan: bool,
    kofam_results: Path | None,
    skip_expression: bool,
    skip_mahal: bool,
    skip_gsea: bool,
    gsea_permutations: int,
    skip_codon_optimization: bool,
    mahal_min_k: int,
    mahal_max_k: int,
    mahal_distance_multiplier: float,
    run_stability: bool,
    stability_bootstraps: int,
    stability_multipliers: str | None,
    auto_select_multiplier: bool,
    stability_core_threshold: float,
    kegg_ko_pathway: Path | None,
    force: bool,
    verbose: bool,
    log_file: Path | None,
):
    """Run codon analysis on a single genome assembly.

    GENOME is the path to a FASTA file containing the genome assembly.

    If Prokka has already been run, supply --prokka-faa and --prokka-ffn to
    skip the Prokka step and use existing output files directly.
    """
    logger = setup_logger(log_file=log_file, verbose=verbose)

    # Validate: if one Prokka file is given, both must be
    if (prokka_faa is None) != (prokka_ffn is None):
        logger.error("--prokka-faa and --prokka-ffn must be provided together (got only one)")
        sys.exit(1)

    from codonpipe.pipeline import run_single_genome

    # Build the pre-existing Prokka paths dict if supplied
    prokka_files = None
    if prokka_faa is not None and prokka_ffn is not None:
        prokka_files = {"faa": prokka_faa, "ffn": prokka_ffn}
        if prokka_gff is not None:
            prokka_files["gff"] = prokka_gff

    # If pre-computed KofamScan results are provided, skip KofamScan
    if kofam_results is not None:
        skip_kofamscan = True

    # Parse stability multipliers from comma-separated string
    stab_mult_list = None
    if stability_multipliers is not None:
        try:
            stab_mult_list = [float(x.strip()) for x in stability_multipliers.split(",")]
        except ValueError:
            logger.error("Invalid --stability-multipliers value: %s", stability_multipliers)
            sys.exit(1)

    try:
        outputs = run_single_genome(
            genome_fasta=genome,
            output_dir=output_dir,
            sample_id=sample_id,
            cpus=threads,
            kingdom=kingdom,
            metagenome=metagenome,
            prokka_files=prokka_files,
            cogs_file=cogs_file,
            kofam_profile=kofam_profile,
            kofam_ko_list=kofam_ko_list,
            skip_kofamscan=skip_kofamscan,
            kofam_results_file=kofam_results,
            skip_expression=skip_expression,
            skip_mahal=skip_mahal,
            skip_gsea=skip_gsea,
            gsea_permutations=gsea_permutations,
            skip_codon_optimization=skip_codon_optimization,
            mahal_min_k=mahal_min_k,
            mahal_max_k=mahal_max_k,
            mahal_distance_multiplier=mahal_distance_multiplier,
            run_stability=run_stability,
            stability_bootstraps=stability_bootstraps,
            stability_multipliers=stab_mult_list,
            auto_select_multiplier=auto_select_multiplier,
            stability_core_threshold=stability_core_threshold,
            kegg_ko_pathway=kegg_ko_pathway,
            gff_file=gff_file,
            force=force,
        )
        logger.info("Done. %d output files generated.", len(outputs))
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=verbose)
        sys.exit(1)


@main.command()
@click.argument("batch_table", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_dir", required=True, type=click.Path(path_type=Path),
              help="Output directory.")
@click.option("-t", "--threads", default=4, type=int, show_default=True,
              help="CPU threads per sample.")
@click.option("-p", "--parallel", default=1, type=int, show_default=True,
              help="Number of samples to process in parallel.")
@click.option("--condition-col", default=None,
              help="Column in the batch table designating experimental conditions. "
                   "Enables within- and between-condition comparative analyses.")
@click.option("--metadata-cols", multiple=True, default=None,
              help="Metadata columns from batch table for comparative analyses (repeatable).")
@click.option("--kingdom", default="Bacteria", show_default=True,
              type=click.Choice(["Bacteria", "Archaea", "Viruses"]))
@click.option("--metagenome", is_flag=True, help="Use Prokka --metagenome mode.")
@click.option("--cogs-file", type=click.Path(exists=True, path_type=Path), default=None,
              help="Custom ribosomal COG accessions file.")
@click.option("--kofam-profile", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--kofam-ko-list", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--skip-kofamscan", is_flag=True, help="Skip KofamScan annotation.")
@click.option("--kofam-results", type=click.Path(exists=True, path_type=Path), default=None,
              help="Pre-computed KofamScan detail-tsv results file (applies to all samples). Per-sample files can also be specified via a 'kofam_results' column in the batch table.")
@click.option("--skip-expression", is_flag=True, help="Skip R-based expression analysis.")
@click.option("--skip-mahal", is_flag=True, help="Skip Mahalanobis clustering of codon usage.")
@click.option("--skip-gsea", is_flag=True, help="Skip pre-ranked Gene Set Enrichment Analysis.")
@click.option("--gsea-permutations", type=int, default=1000, show_default=True,
              help="Number of permutations for GSEA p-value estimation.")
@click.option("--skip-codon-optimization", is_flag=True,
              help="Skip the codon-optimization comparison step (Step 12).")
@click.option("--mahal-min-k", type=int, default=2, show_default=True,
              help="Minimum number of Mahalanobis clustering components to test.")
@click.option("--mahal-max-k", type=int, default=8, show_default=True,
              help="Maximum number of Mahalanobis clustering components to test.")
@click.option("--mahal-distance-multiplier", type=float, default=2.0, show_default=True,
              help="Mahalanobis cluster radius as a multiplier of the median RP distance. "
                   "Lower values (e.g. 1.5) produce a tighter cluster; higher values (e.g. 3.0) are more permissive.")
@click.option("--run-stability", is_flag=True,
              help="Run bootstrap stability analysis on the Mahalanobis cluster.")
@click.option("--stability-bootstraps", type=int, default=100, show_default=True,
              help="Number of bootstrap replicates per multiplier for stability analysis.")
@click.option("--stability-multipliers", type=str, default=None,
              help="Comma-separated multiplier values to test (e.g. '1.0,1.5,2.0,2.5,3.0').")
@click.option("--auto-select-multiplier", is_flag=True,
              help="Automatically use the stability-recommended multiplier. Requires --run-stability.")
@click.option("--stability-core-threshold", type=float, default=0.5, show_default=True,
              help="Membership frequency threshold for a gene to be 'core'. "
                   "0.5 = majority-rule consensus; 0.9 = high-confidence subset.")
@click.option("--kegg-ko-pathway", type=click.Path(exists=True, path_type=Path), default=None,
              help="KO-to-pathway mapping TSV for offline enrichment.")
@click.option("--gff", "gff_file", type=click.Path(exists=True, path_type=Path), default=None,
              help="GFF3 annotation file for tRNA extraction (overrides per-sample auto-detection).")
@click.option("--force", is_flag=True, help="Overwrite existing outputs.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--log-file", type=click.Path(path_type=Path), default=None)
def batch(
    batch_table: Path,
    output_dir: Path,
    threads: int,
    parallel: int,
    condition_col: str | None,
    metadata_cols: tuple[str, ...],
    kingdom: str,
    metagenome: bool,
    cogs_file: Path | None,
    kofam_profile: Path | None,
    kofam_ko_list: Path | None,
    skip_kofamscan: bool,
    kofam_results: Path | None,
    skip_expression: bool,
    skip_mahal: bool,
    mahal_min_k: int,
    mahal_max_k: int,
    mahal_distance_multiplier: float,
    run_stability: bool,
    stability_bootstraps: int,
    stability_multipliers: str | None,
    auto_select_multiplier: bool,
    stability_core_threshold: float,
    kegg_ko_pathway: Path | None,
    gff_file: Path | None,
    skip_gsea: bool,
    gsea_permutations: int,
    skip_codon_optimization: bool,
    force: bool,
    verbose: bool,
    log_file: Path | None,
):
    """Run codon analysis on multiple genomes from a batch table.

    BATCH_TABLE is a TSV/CSV/TXT file (tab- or comma-delimited) with at minimum
    a 'genome_path' column.
    Optional columns: 'sample_id', plus any metadata columns for comparative analysis.

    To skip Prokka for samples that already have gene predictions, include
    'prokka_faa' and 'prokka_ffn' columns with paths to existing files.
    Samples with both columns populated will skip Prokka; samples with empty
    values in those columns will run Prokka normally.

    Include a 'gff_path' column with paths to GFF3 files for tRNA extraction
    in the advanced analyses (tRNA-codon co-adaptation). If omitted, the
    pipeline auto-detects GFF from Prokka output.

    Include a 'kofam_results' column with paths to pre-computed KofamScan
    detail-tsv output files to skip KofamScan for those samples. The --kofam-results
    CLI option applies a single file to all samples.

    Example batch table (mixed — some pre-run, some not):

    \b
        genome_path          sample_id  prokka_faa            prokka_ffn            phylum
        /data/genome1.fasta  sample_A   /data/g1.faa          /data/g1.ffn          Firmicutes
        /data/genome2.fasta  sample_B                                               Bacteroidetes
    """
    logger = setup_logger(log_file=log_file, verbose=verbose)

    from codonpipe.pipeline import run_batch

    # Parse stability multipliers from comma-separated string
    stab_mult_list = None
    if stability_multipliers is not None:
        try:
            stab_mult_list = [float(x.strip()) for x in stability_multipliers.split(",")]
        except ValueError:
            logger.error("Invalid --stability-multipliers value: %s", stability_multipliers)
            sys.exit(1)

    try:
        meta_cols = list(metadata_cols) if metadata_cols else None
        outputs = run_batch(
            batch_table=batch_table,
            output_dir=output_dir,
            cpus=threads,
            parallel=parallel,
            metadata_cols=meta_cols,
            condition_col=condition_col,
            kingdom=kingdom,
            metagenome=metagenome,
            cogs_file=cogs_file,
            kofam_profile=kofam_profile,
            kofam_ko_list=kofam_ko_list,
            skip_kofamscan=skip_kofamscan,
            kofam_results_file=kofam_results,
            skip_expression=skip_expression,
            skip_mahal=skip_mahal,
            mahal_min_k=mahal_min_k,
            mahal_max_k=mahal_max_k,
            mahal_distance_multiplier=mahal_distance_multiplier,
            run_stability=run_stability,
            stability_bootstraps=stability_bootstraps,
            stability_multipliers=stab_mult_list,
            auto_select_multiplier=auto_select_multiplier,
            stability_core_threshold=stability_core_threshold,
            kegg_ko_pathway=kegg_ko_pathway,
            gff_file=gff_file,
            force=force,
            skip_gsea=skip_gsea,
            gsea_permutations=gsea_permutations,
            skip_codon_optimization=skip_codon_optimization,
        )
        logger.info("Batch analysis complete. %d output files.", len(outputs))
    except Exception as e:
        logger.error("Batch pipeline failed: %s", e, exc_info=verbose)
        sys.exit(1)


@main.command("gene-set")
@click.argument("sample_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--goi-file", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="File listing GOI Prokka locus tags, one per line. "
                   "Lines starting with '#' are treated as comments. "
                   "Trailing whitespace-delimited tokens after the locus tag are ignored.")
@click.option("-s", "--sample-id", default=None,
              help="Sample identifier. Defaults to the directory name.")
@click.option("-o", "--output-dir", "output_dir", default=None,
              type=click.Path(path_type=Path),
              help="Output directory. Defaults to <SAMPLE_DIR>/gene_set/.")
@click.option("--n-permutations", default=999, type=int, show_default=True,
              help="Permutation count for Aitchison and HGT-flag tests.")
@click.option("--no-length-matching", is_flag=True,
              help="Disable length-matched permutation null (uniform random instead). "
                   "Length matching is recommended; disable only for sensitivity analysis.")
@click.option("--no-figure", is_flag=True,
              help="Skip the six-panel summary figure.")
@click.option("--rng-seed", default=42, type=int, show_default=True,
              help="Random seed for permutation tests and bootstrap CIs.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def gene_set_cmd(
    sample_dir: Path,
    goi_file: Path,
    sample_id: str | None,
    output_dir: Path | None,
    n_permutations: int,
    no_length_matching: bool,
    no_figure: bool,
    rng_seed: int,
    verbose: bool,
):
    """Compare a shortlist of genes-of-interest to the rest of the genome.

    Takes the per-sample output directory produced by `codonpipe run` (or
    `codonpipe batch`) and a GOI list of Prokka locus tags. Reports:

    \b
      - per-GOI metrics with within-genome percentile ranks
      - Mann-Whitney + Cliff's delta on each scalar metric (CAI, MELP,
        Fop, ENC, ENCprime, MILC, GC3, mahalanobis_dist,
        mahal_cluster_distance, membership_score, cbi_rp, cbi_mahal)
      - per-codon RSCU comparison vs genome / RP / Mahal-cluster references
      - Aitchison-distance permutation test with length-matched controls
      - one-sided HGT-flag enrichment (if hgt_candidates.tsv is present)
      - two-sided Mahal-cluster membership-rate permutation test
        (if gmm_clusters.tsv is present)
      - a seven-panel summary figure (PNG + SVG) including a
        genome-centroid vs cluster-centroid Mahalanobis biplot

    Example:

    \b
        codonpipe gene-set output_run/G0370_i3 \\
            --goi-file my_goi.txt \\
            -o output_run/G0370_i3/gene_set/
    """
    logger = setup_logger(verbose=verbose)

    from codonpipe.modules.gene_set import (
        analyze_gene_set, load_sample_outputs, read_goi_file,
    )

    sid = sample_id or sample_dir.name
    out_dir = output_dir or (sample_dir / "gene_set")
    out_dir.mkdir(parents=True, exist_ok=True)

    goi_ids = read_goi_file(goi_file)
    if not goi_ids:
        logger.error("GOI file %s contained no usable locus tags.", goi_file)
        sys.exit(1)
    logger.info("Loaded %d GOI locus tags from %s", len(goi_ids), goi_file)

    try:
        loaded = load_sample_outputs(sample_dir, sid)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    try:
        outputs = analyze_gene_set(
            **loaded,
            goi_ids=goi_ids,
            output_dir=out_dir,
            sample_id=sid,
            n_permutations=n_permutations,
            length_matched=not no_length_matching,
            rng_seed=rng_seed,
            make_figure=not no_figure,
        )
    except ValueError as e:
        logger.error("Gene-set analysis failed: %s", e)
        sys.exit(1)

    logger.info("Gene-set analysis complete. Outputs:")
    for kind, path in outputs.items():
        logger.info("  %-22s %s", kind, path)


@main.command("signatures")
@click.argument("sample_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-s", "--sample-id", default=None,
              help="Sample identifier. Defaults to the directory name.")
@click.option("-o", "--output-dir", "output_dir", default=None,
              type=click.Path(path_type=Path),
              help="Output directory. Defaults to <SAMPLE_DIR>/signatures/.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def signatures_cmd(sample_dir: Path, sample_id: str | None, output_dir: Path | None, verbose: bool):
    """Emit cross-genome-comparable gene + genome signatures for one sample.

    Reads the per-sample CodonPipe output directory and writes:

    \b
      <sample_id>_gene_signature.tsv    one row per gene, with within-genome-
                                        normalized scalars + a 38-d CLR-Δ
                                        codon-preference vector
      <sample_id>_genome_signature.tsv  one row per genome, with the
                                        translational-selection geometry vector
                                        (CLR(Mahal) − CLR(genome) + CLR(RP) −
                                        CLR(genome) + Aitchison distances) plus
                                        ecology summary scalars

    Both files are designed to be concatenated across thousands of genomes for
    cross-genome clustering. Run `codonpipe corpus` on a directory of these
    signature files to do the actual cross-genome aggregation.

    Example:

    \b
        codonpipe signatures output_run/G0370_i3 \\
            -o output_run/G0370_i3/signatures/
    """
    logger = setup_logger(verbose=verbose)
    from codonpipe.modules.cross_genome import write_signatures_for_sample

    sid = sample_id or sample_dir.name
    out_dir = output_dir or (sample_dir / "signatures")
    try:
        outputs = write_signatures_for_sample(sample_dir, sid, out_dir)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Signature build failed: %s", e, exc_info=verbose)
        sys.exit(1)
    logger.info("Signatures complete:")
    for k, v in outputs.items():
        logger.info("  %-22s %s", k, v)


@main.command("corpus")
@click.argument("input_dirs", nargs=-1, required=True,
                type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output-dir", "output_dir", required=True,
              type=click.Path(path_type=Path),
              help="Where to write corpus_*.tsv outputs and the summary figure.")
@click.option("--features", default="geometry",
              type=click.Choice(["geometry", "all"]),
              help="Feature set for clustering: 'geometry' = CLR-Δ vectors + "
                   "Aitchison distances only (recommended for codon-strategy "
                   "clustering); 'all' = full vector including ecology scalars.")
@click.option("--use-umap", is_flag=True,
              help="Apply UMAP after PCA for the 2-D embedding (requires "
                   "umap-learn). Without this flag, the first two PCs are used.")
@click.option("--include-gene-level", is_flag=True,
              help="Concatenate the gene-level signatures into "
                   "corpus_gene_signature.tsv. Skipped by default because "
                   "gene-level corpora are large (millions of rows for "
                   "thousands of genomes). Implied by --gene-level-clustering "
                   "and --gene-level-by-category.")
@click.option("--gene-level-clustering", is_flag=True,
              help="Cluster genes across the corpus by their CLR-Δ codon "
                   "vectors. Emits corpus_gene_clusters.tsv and a four-panel "
                   "gene-level UMAP figure with overlays for cluster, host "
                   "genome, host phylum, and host GC3.")
@click.option("--gene-level-by-category", default=None,
              type=click.Choice(["KO", "COG", "COG_ID"], case_sensitive=False),
              help="Within each functional category (KO or COG_ID), cluster "
                   "the genes labelled with that category across the corpus. "
                   "Answers 'do all genes of this function share a codon "
                   "strategy, or split into ecological sub-strategies?'. "
                   "Categories with fewer than --gene-level-min-category-size "
                   "genes are skipped.")
@click.option("--gene-level-min-category-size", type=int, default=10,
              show_default=True,
              help="Minimum genes per category for per-category clustering.")
@click.option("--gene-level-max-genes", type=int, default=1_000_000,
              show_default=True,
              help="Safety cap on the gene-level corpus size for general "
                   "clustering. Above this, --gene-level-clustering raises "
                   "rather than running for hours.")
@click.option("--phylogeny", "phylogeny_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional phylogeny for Mantel-test validation. Accepts "
                   "either a Newick file (.nwk/.newick/.tree, requires "
                   "biopython) or a precomputed pairwise distance-matrix TSV "
                   "with sample_ids as both row and column labels.")
@click.option("--metadata", "metadata_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Optional metadata TSV with a sample_id column. Categorical "
                   "columns are tested for cluster association via "
                   "chi-squared + Cramer's V.")
@click.option("--hdbscan-min-cluster-size", type=int, default=None,
              help="Override the heuristic HDBSCAN min_cluster_size. Default "
                   "scales to ~2%% of corpus, floor 5, ceiling 50.")
@click.option("--focus-genome", "focus_genomes", multiple=True,
              help="Render a single-genome locator figure for this sample_id "
                   "(repeatable). Highlights the genome on the embedding and "
                   "shows its z-score / percentile rank on each scalar feature. "
                   "Use to give reviewers a quick where-does-X-sit answer.")
@click.option("--rng-seed", default=42, type=int, show_default=True,
              help="Random seed for PCA / UMAP / HDBSCAN / Mantel.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def corpus_cmd(
    input_dirs: tuple[Path, ...],
    output_dir: Path,
    features: str,
    use_umap: bool,
    include_gene_level: bool,
    gene_level_clustering: bool,
    gene_level_by_category: str | None,
    gene_level_min_category_size: int,
    gene_level_max_genes: int,
    phylogeny_path: Path | None,
    metadata_path: Path | None,
    hdbscan_min_cluster_size: int | None,
    focus_genomes: tuple[str, ...],
    rng_seed: int,
    verbose: bool,
):
    """Aggregate per-genome signatures into a cross-genome corpus.

    Discovers `*_genome_signature.tsv` (and optionally `*_gene_signature.tsv`)
    files under one or more input directories, standardizes the chosen feature
    block, runs PCA (and UMAP if installed) for dimension reduction, clusters
    with HDBSCAN (KMeans fallback), and optionally validates against a
    phylogeny via Mantel test and against metadata via chi-squared.

    Outputs in --output-dir:

    \b
      corpus_genome_signature.tsv   stacked per-genome features
      corpus_genome_clusters.tsv    sample_id, cluster, embed_dim1, embed_dim2,
                                    + any metadata columns
      corpus_validation.tsv         Mantel + cluster-vs-metadata tests
                                    (only if --phylogeny / --metadata given)
      corpus_gene_signature.tsv     stacked per-gene features (only if
                                    --include-gene-level)
      corpus_dimension_reduction.png/.svg   3-panel summary figure

    Defensible-by-default: 'geometry' features remove mutational background by
    construction, so cross-genome distance reflects translational selection
    rather than GC content. Use --phylogeny to confirm: a low Mantel r between
    signature distance and phylogenetic distance is evidence the clustering is
    capturing biology rather than just recovering taxonomy.

    Example:

    \b
        codonpipe corpus output_run/G0370_i3/signatures \\
                         output_run/NC_009515/signatures \\
                         output_run/.../signatures \\
            -o corpus_results/ \\
            --phylogeny species_tree.nwk \\
            --metadata genome_metadata.tsv \\
            --use-umap
    """
    logger = setup_logger(verbose=verbose)
    from codonpipe.modules.cross_genome import build_corpus

    try:
        outputs = build_corpus(
            input_dirs=list(input_dirs),
            output_dir=output_dir,
            features=features,
            use_umap=use_umap,
            include_gene_level=include_gene_level,
            gene_level_clustering=gene_level_clustering,
            gene_level_by_category=gene_level_by_category,
            gene_level_min_category_size=gene_level_min_category_size,
            gene_level_max_genes=gene_level_max_genes,
            phylogeny_path=phylogeny_path,
            metadata_path=metadata_path,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            focus_genomes=list(focus_genomes) if focus_genomes else None,
            rng_seed=rng_seed,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Corpus build failed: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Corpus build failed: %s", e, exc_info=verbose)
        sys.exit(1)

    logger.info("Corpus build complete:")
    for k, v in outputs.items():
        logger.info("  %-25s %s", k, v)


@main.command("codon-optimization")
@click.argument("sample_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-s", "--sample-id", default=None,
              help="Sample identifier. Defaults to the directory name.")
@click.option("-o", "--output-dir", "output_dir", default=None,
              type=click.Path(path_type=Path),
              help="Output directory. Defaults to <SAMPLE_DIR>/codon_optimization/.")
@click.option("--ffn-path", "ffn_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Path to the gene CDS .ffn file. Use when codonpipe was "
                   "run with an external --prokka-ffn (the file isn't copied "
                   "into the sample directory in that case). Defaults to "
                   "auto-discovery under <SAMPLE_DIR>/[annotation/]prokka/.")
@click.option("--no-figure", is_flag=True,
              help="Skip the figures, write only the TSV outputs.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def codon_optimization_cmd(
    sample_dir: Path,
    sample_id: str | None,
    output_dir: Path | None,
    ffn_path: Path | None,
    no_figure: bool,
    verbose: bool,
):
    """Compare genome / RP / Mahal-cluster codon-usage profiles for one organism.

    Produces a synthesis-ready recommendation table plus three figures
    that visualize how the three reference frames differ and quantify the
    gain from Mahal-cluster-based codon optimization vs the classic
    RP-based approach.

    Outputs in --output-dir:

    \b
      <sid>_codon_optimization_table.tsv     per-codon w-values + optimal flags
      <sid>_codon_optimization_summary.tsv   per-AA-family agreement
      <sid>_codon_optimization_recommend.tsv synthesis-ready preferred codons
      <sid>_codon_optimization_gain.tsv      per-gene cbi_rp vs cbi_mahal
      <sid>_three_way_rscu.png/.svg          RSCU bars per codon
      <sid>_optimization_agreement.png/.svg  per-AA RP-vs-Mahal table
      <sid>_optimization_gain.png/.svg       per-gene CBI scatter + gain dist

    Defensible reason to prefer Mahal-derived weights for codon
    optimization: the Mahal cluster is identified by codon-usage
    similarity rather than by RP gene annotations, so it captures the
    organism's actual translationally-optimized cohort and excludes RP
    annotation outliers (truncations, paralogs, modified variants) that
    can pull the RP-based reference centroid off-target.

    Example:

    \b
        codonpipe codon-optimization output_run/G0370_i3 \\
            -o output_run/G0370_i3/codon_optimization/
    """
    logger = setup_logger(verbose=verbose)
    from codonpipe.modules.codon_optimization import run_codon_optimization

    sid = sample_id or sample_dir.name
    out_dir = output_dir or (sample_dir / "codon_optimization")

    try:
        outputs = run_codon_optimization(
            sample_dir, sid, out_dir,
            make_figures=not no_figure,
            ffn_path=ffn_path,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Codon-optimization analysis failed: %s", e, exc_info=verbose)
        sys.exit(1)

    logger.info("Codon optimization complete:")
    for k, v in outputs.items():
        logger.info("  %-25s %s", k, v)


@main.command("install-grodon")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("--timeout", default=600, type=int, show_default=True,
              help="Maximum seconds to wait for installation.")
def install_grodon_cmd(verbose: bool, timeout: int):
    """Install gRodon2 and its R/Bioconductor dependencies.

    Requires R (>= 4.0) with Rscript on PATH. Installs:

    \b
      - BiocManager (CRAN)
      - Biostrings, coRdon (Bioconductor)
      - matrixStats, dplyr, jsonlite, remotes (CRAN)
      - gRodon2 (GitHub: jlw-ecoevo/gRodon2)

    Run this once after creating the conda environment:

    \b
        conda env create -f environment.yml
        conda activate codonpipe
        codonpipe install-grodon
    """
    logger = setup_logger(verbose=verbose)

    from codonpipe.modules.grodon import install_grodon

    success = install_grodon(timeout=timeout)
    if success:
        logger.info("gRodon2 is ready to use.")
    else:
        logger.error(
            "gRodon2 installation failed. Check the output above for details. "
            "You can also install manually in R:\n"
            "  remotes::install_github('jlw-ecoevo/gRodon2')"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
