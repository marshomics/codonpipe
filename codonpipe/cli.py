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
@click.option("--skip-gmm", is_flag=True,
              help="Skip GMM codon usage clustering.")
@click.option("--gmm-min-k", type=int, default=2, show_default=True,
              help="Minimum number of GMM components to test.")
@click.option("--gmm-max-k", type=int, default=8, show_default=True,
              help="Maximum number of GMM components to test.")
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
    skip_gmm: bool,
    gmm_min_k: int,
    gmm_max_k: int,
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
            skip_gmm=skip_gmm,
            gmm_min_k=gmm_min_k,
            gmm_max_k=gmm_max_k,
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
@click.option("--skip-gmm", is_flag=True, help="Skip GMM codon usage clustering.")
@click.option("--gmm-min-k", type=int, default=2, show_default=True,
              help="Minimum number of GMM components to test.")
@click.option("--gmm-max-k", type=int, default=8, show_default=True,
              help="Maximum number of GMM components to test.")
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
    skip_gmm: bool,
    gmm_min_k: int,
    gmm_max_k: int,
    kegg_ko_pathway: Path | None,
    gff_file: Path | None,
    force: bool,
    verbose: bool,
    log_file: Path | None,
):
    """Run codon analysis on multiple genomes from a batch table.

    BATCH_TABLE is a TSV/CSV file with at minimum a 'genome_path' column.
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
            skip_gmm=skip_gmm,
            gmm_min_k=gmm_min_k,
            gmm_max_k=gmm_max_k,
            kegg_ko_pathway=kegg_ko_pathway,
            gff_file=gff_file,
            force=force,
        )
        logger.info("Batch analysis complete. %d output files.", len(outputs))
    except Exception as e:
        logger.error("Batch pipeline failed: %s", e, exc_info=verbose)
        sys.exit(1)


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
