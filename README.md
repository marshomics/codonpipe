# CodonPipe

End-to-end codon usage analysis pipeline for microbial genomes. Takes a genome assembly (or hundreds of them) and produces RSCU values, expression level predictions, codon usage tables, statistical comparisons, and publication-ready figures.

## Installation

### Conda (recommended)

```bash
git clone https://github.com/jmarsh/codonpipe.git
cd codonpipe
mamba env create -f environment.yml
mamba activate codonpipe
pip install -e .
```

This installs all Python, R, and bioinformatics dependencies (Prokka, COGclassifier, KofamScan, R with coRdon/Biostrings).

### gRodon2

gRodon2 (growth rate prediction from codon usage bias) is not available through conda. Install it after creating the environment:

```bash
codonpipe install-grodon
```

This installs BiocManager, Biostrings, coRdon, matrixStats, dplyr, jsonlite, remotes, and gRodon2 from GitHub. The command handles R user library paths automatically. If gRodon2 is not installed, the pipeline skips growth rate prediction and continues with everything else.

### pip only

```bash
pip install .
```

You'll need Prokka (>=1.14), COGclassifier (>=1.0), KofamScan (>=1.3), HMMER (>=3.3), and R (>=4.2) with coRdon and Biostrings available on your PATH. See `environment.yml` for the full dependency list.

## Input modes

CodonPipe has two modes: single genome and batch. The mode determines what analyses run.

### Single genome

```bash
codonpipe run genome.fasta -o results/ -t 8
```

Input is one FASTA file. CodonPipe runs the 11-step per-genome pipeline (described below) and produces tables, statistics, and plots for that organism. No comparative analyses.

### Batch mode

```bash
codonpipe batch genomes.tsv -o results/ -t 8 -p 4 \
    --metadata-cols phylum --metadata-cols geo_category
```

Input is a TSV or CSV manifest table. CodonPipe runs the per-genome pipeline on every row, then produces comparative analyses across all genomes.

#### Batch table format

The only required column is `genome_path`. All other columns are optional.

| Column | Required | Purpose |
|--------|----------|---------|
| `genome_path` | yes | Path to FASTA assembly |
| `sample_id` | no | Sample identifier (defaults to filename stem) |
| `prokka_faa` | no | Path to existing Prokka .faa file |
| `prokka_ffn` | no | Path to existing Prokka .ffn file |
| `prokka_gff` | no | Path to existing Prokka .gff file |
| `gff_path` | no | GFF3 file for tRNA extraction |
| `kofam_results` | no | Pre-computed KofamScan detail-tsv output |
| any other columns | no | Treated as metadata for comparative analysis |

Example with mixed pre-run and fresh samples:

```
genome_path	sample_id	prokka_faa	prokka_ffn	phylum	geo_category
/data/genome1.fasta	sample_A	/data/g1.faa	/data/g1.ffn	Firmicutes	western
/data/genome2.fasta	sample_B			Bacteroidetes	eastern
/data/genome3.fasta	sample_C	/data/g3.faa	/data/g3.ffn	Proteobacteria	central
```

sample_A and sample_C use existing Prokka files; sample_B runs Prokka from scratch. Both `prokka_faa` and `prokka_ffn` must be populated for a given row to skip Prokka. Rows can be mixed freely.

The delimiter is auto-detected (tabs preferred, commas accepted).

## Per-genome pipeline (11 steps)

Every genome, whether processed via `run` or `batch`, goes through these steps:

1. **Prokka** ORF prediction (or use pre-existing .faa/.ffn files)
2. **COGclassifier** assigns COG categories, then extracts ribosomal proteins using 96 ribosomal COG accessions
3. **KofamScan** annotates predicted proteins with KEGG Orthology IDs
4. **RSCU analysis** computes per-gene RSCU for all CDS, genome-level median RSCU (59 sense codons), concatenated ribosomal protein RSCU, codon frequency tables, and ENC with GC3 content
5. **CU bias statistics** via coRdon: ENCprime (GC-corrected ENC, Novembre 2002) and MILC (Measure Independent of Length and Composition, Supek & Vlahovicek 2005) per gene
6. **Expression prediction** via coRdon: MELP, CAI, and Fop (Frequency of optimal codons, Ikemura 1981) using ribosomal proteins as the highly expressed reference set; genes classified into high (>=95th percentile), medium, and low (<=5th percentile) expression tiers per metric
7. **Pathway enrichment** via hypergeometric test for KEGG pathways over-represented in high- and low-expression gene sets, with Benjamini-Hochberg FDR correction
8. **Advanced analyses**: correspondence analysis (COA) on codon usage, S-value adaptation to the ribosomal reference set, GC12-vs-GC3 neutrality analysis, PR2 (purine/pyrimidine ratio), delta-RSCU distance from the genome average, tRNA-codon co-adaptation correlation (if GFF provided), COG enrichment in high/low bias genes, gene length vs codon bias, and ENC-ENCprime difference
9. **Biological/ecological analyses**: HGT candidate detection via Mahalanobis distance on RSCU profiles, growth rate prediction (classic CAI-based and gRodon2 if installed), translational selection analysis (Fop gradient, positional effects across 5'/middle/3' gene regions), phage and mobile element detection, strand asymmetry, and operon co-adaptation
10. **Codon usage tables** in six formats: RSCU, absolute counts, per-thousand frequencies, W values (relative adaptiveness), adaptation weights, and CBI (Codon Bias Index)
11. **Publication-ready plots** at 300 DPI in PNG and SVG (editable in Adobe Illustrator)

Steps 5 and 6 require R with coRdon. Pass `--skip-expression` to skip them.

Step 3 (KofamScan) can be skipped with `--skip-kofamscan` or bypassed with pre-computed results via `--kofam-results`.

Step 9's gRodon2 predictions require gRodon2 to be installed (see Installation above). If absent, the pipeline substitutes a simpler CAI-based growth rate estimate and continues.

## Batch-level analyses

These run after all per-genome pipelines complete. What you get depends on the number of genomes, whether you supply metadata columns, and whether you specify a condition column.

### Always (>=2 genomes)

Pairwise qualitative comparison plots that need no statistical tests and no condition labels:

- RSCU overlay bar chart (all genomes on one plot, grouped by codon)
- RSCU difference heatmap (pairwise delta-RSCU between genomes)
- Genome metrics comparison (ENC, GC3, growth rate side by side)
- Expression tier comparison (high/medium/low gene counts per genome)
- Enrichment comparison heatmap (shared and divergent KEGG pathway enrichments)
- Bio/ecology multi-panel comparison (HGT Mahalanobis distributions, strand asymmetry, operon co-adaptation, optimal codon counts)
- HGT and mobile element burden bar chart (HGT candidates, mobilome, phage genes per genome)
- Ribosomal protein RSCU divergence from genome average (distance and correlation per genome)
- gRodon2 batch comparison (doubling times with CIs, CUBHE vs ConsistencyHE scatter, gRodon2 vs CAI comparison, if gRodon2 data available)
- Codon adaptation fingerprint radar plots (per-amino-acid RSCU profiles overlaid as spider plots)
- Translational selection landscape (ENC vs GC3 with Wright's expected curve, expression metric distributions, most variable codons heatmap)
- HGT ecology advanced (Mahalanobis density curves, foreign DNA burden fractions, HGT gene deviation scatter)
- Growth rate and CUB strategy (gRodon2 predictions, ribosomal-vs-genome RSCU divergence plotted against doubling time, Fop distributions)

### With metadata columns (`--metadata-cols`)

When you specify one or more metadata columns (phylum, geography, treatment, etc.), the pipeline additionally runs:

- PCA and UMAP dimensionality reduction, colored by each metadata column
- Hierarchical clustering heatmaps (Manhattan distance, complete linkage)
- Z-score normalized RSCU tables
- Pairwise Wilcoxon rank-sum tests per codon for each amino acid family (requires >=5 samples per group)
- Bonferroni-corrected significance heatmaps
- Per-amino-acid boxplots grouped by metadata value

If no metadata columns are specified explicitly, the pipeline auto-detects any non-pipeline columns in the batch table and uses those.

### With a condition column (`--condition-col`)

The condition column designates experimental groups (e.g., `treatment` with values "control" and "treated"). This enables formal statistical comparison between groups:

**Within-condition analyses** (any number of conditions):
- Per-condition summary statistics for all genome-level metrics (median CAI, ENC, GC3, growth rate, HGT fraction, etc.)
- Per-condition RSCU dispersion heatmap

**Between-condition analyses** (requires >=2 unique conditions):
- Metric-level tests (Kruskal-Wallis with FDR correction) across 90+ genome-level metrics
- Per-codon RSCU tests (Mann-Whitney U) with Cliff's delta effect sizes
- PERMANOVA on RSCU profiles (multivariate condition test)
- Ribosomal and high-expression gene RSCU comparison between conditions
- Pathway enrichment pattern comparison
- HGT burden and strand asymmetry pattern differences
- Optimal codon set comparison
- GC3 and GC12 distribution comparison

Each test produces a TSV results table and a corresponding visualization.

## CLI reference

### `codonpipe run`

```
codonpipe run GENOME -o OUTPUT_DIR [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | (required) | Output directory |
| `-s, --sample-id` | filename stem | Sample identifier |
| `-t, --threads` | 4 | CPU threads for external tools |
| `--kingdom` | Bacteria | Prokka kingdom (Bacteria, Archaea, Viruses) |
| `--metagenome` | off | Prokka metagenome mode |
| `--prokka-faa` | — | Pre-existing .faa file (must pair with --prokka-ffn) |
| `--prokka-ffn` | — | Pre-existing .ffn file (must pair with --prokka-faa) |
| `--prokka-gff` | — | Pre-existing .gff file (optional) |
| `--gff` | auto-detect | GFF3 for tRNA extraction |
| `--cogs-file` | bundled 96 COGs | Custom ribosomal COG accessions |
| `--kofam-profile` | — | KOfam profiles directory |
| `--kofam-ko-list` | — | KOfam ko_list file |
| `--skip-kofamscan` | off | Skip KofamScan step |
| `--kofam-results` | — | Pre-computed KofamScan detail-tsv |
| `--skip-expression` | off | Skip R-based analyses (MELP/CAI/Fop/ENCprime/MILC) |
| `--kegg-ko-pathway` | auto-downloaded | KO-to-pathway mapping TSV |
| `--force` | off | Overwrite existing outputs |
| `-v, --verbose` | off | Debug-level logging |
| `--log-file` | — | Write log to file |

### `codonpipe batch`

```
codonpipe batch BATCH_TABLE -o OUTPUT_DIR [OPTIONS]
```

All options from `run` are available, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `-p, --parallel` | 1 | Samples to process concurrently |
| `--metadata-cols` | auto-detect | Metadata columns for comparative analysis (repeatable) |
| `--condition-col` | — | Column designating experimental conditions |

### `codonpipe install-grodon`

```
codonpipe install-grodon [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-v, --verbose` | off | Debug logging |
| `--timeout` | 600 | Max seconds for installation |

## Output structure

```
results/
├── sample_A/
│   ├── prokka/                              # Prokka output files
│   ├── cogclassifier/                       # COG classification
│   ├── ribosomal_proteins/                  # Extracted RP sequences + IDs
│   ├── kofamscan/                           # KofamScan annotations
│   ├── rscu/
│   │   ├── sample_A_rscu_all_genes.tsv      # Per-gene RSCU
│   │   ├── sample_A_rscu_median.tsv         # Genome-level median RSCU
│   │   ├── sample_A_rscu_ribosomal.tsv      # Ribosomal protein RSCU
│   │   ├── sample_A_codon_frequency.tsv     # Codon frequency table
│   │   ├── sample_A_enc.tsv                 # ENC + GC3 per gene
│   │   └── sample_A_rscu_annotated.tsv      # RSCU with KO annotations
│   ├── cu_statistics/
│   │   ├── sample_A_encprime.tsv            # ENCprime per gene
│   │   └── sample_A_milc.tsv               # MILC per gene
│   ├── expression/
│   │   ├── sample_A_melp.tsv
│   │   ├── sample_A_cai.tsv
│   │   ├── sample_A_fop.tsv
│   │   ├── sample_A_expression.tsv          # Combined scores + tier classes
│   │   └── sample_A_expression_annotated.tsv
│   ├── enrichment/
│   │   ├── sample_A_expression_by_tier.tsv  # All genes with classes + KO
│   │   ├── sample_A_CAI_high_enrichment.tsv
│   │   ├── sample_A_CAI_low_enrichment.tsv
│   │   ├── sample_A_MELP_high_enrichment.tsv
│   │   ├── sample_A_MELP_low_enrichment.tsv
│   │   ├── sample_A_Fop_high_enrichment.tsv
│   │   └── sample_A_Fop_low_enrichment.tsv
│   ├── advanced/
│   │   ├── sample_A_coa.tsv                 # Correspondence analysis
│   │   ├── sample_A_s_value.tsv             # S-value adaptation
│   │   ├── sample_A_gc12_gc3.tsv            # Neutrality analysis
│   │   ├── sample_A_pr2.tsv                 # Purine/pyrimidine ratio
│   │   ├── sample_A_delta_rscu.tsv          # Distance from genome avg
│   │   ├── sample_A_trna_codon_correlation.tsv  # tRNA co-adaptation
│   │   ├── sample_A_cog_enrichment.tsv
│   │   └── sample_A_gene_length_bias.tsv
│   ├── bio_ecology/
│   │   ├── sample_A_hgt_candidates.tsv
│   │   ├── sample_A_growth_rate_prediction.tsv
│   │   ├── sample_A_grodon2_prediction.tsv  # if gRodon2 installed
│   │   ├── sample_A_translational_selection_fop_gradient.tsv
│   │   ├── sample_A_translational_selection_position_effects.tsv
│   │   ├── sample_A_phage_mobile_elements.tsv
│   │   ├── sample_A_strand_asymmetry.tsv
│   │   └── sample_A_operon_coadaptation.tsv
│   ├── codon_tables/
│   │   ├── sample_A_rscu.tsv
│   │   ├── sample_A_absolute_counts.tsv
│   │   ├── sample_A_per_thousand.tsv
│   │   ├── sample_A_relative_adaptiveness.tsv
│   │   ├── sample_A_adaptation_weights.tsv
│   │   └── sample_A_cbi.tsv
│   └── plots/                               # PNG + SVG for each figure
│
├── combined_rscu.tsv                        # Merged genome-level RSCU + metadata
├── combined_rscu_ribosomal.tsv              # Merged ribosomal RSCU
├── statistics/                              # Pairwise Wilcoxon tests
│   ├── rscu_zscored.tsv
│   └── {metadata}_{amino_acid}_wilcoxon.tsv
├── comparative/                             # Condition-aware analyses
│   ├── sample_metrics.tsv
│   ├── within_condition_stats.tsv
│   ├── within_condition_rscu_dispersion.tsv
│   ├── between_condition_tests.tsv
│   ├── between_condition_rscu_tests.tsv
│   ├── permanova_rscu.tsv
│   └── between_condition_*.tsv              # Ribosomal, HGT, enrichment, etc.
├── comparison/plots/                        # Pairwise qualitative plots
│   ├── rscu_overlay.png
│   ├── rscu_delta_heatmap.png
│   ├── codon_adaptation_fingerprint.png
│   ├── translational_selection_landscape.png
│   ├── hgt_ecology_advanced.png
│   ├── growth_cub_strategy.png
│   └── ...
└── plots/                                   # Batch statistical plots
    ├── pca_{metadata}.png
    ├── umap_{metadata}.png
    ├── heatmap_clustered_{metadata}.png
    ├── boxplot_{metadata}_{amino_acid}.png
    └── significance_{metadata}_all_wilcoxon.png
```

The `statistics/`, `comparative/`, `comparison/plots/`, and `plots/` directories only appear in batch mode. The `comparative/` directory only appears when `--condition-col` is used.

## What runs when

| Analysis | `run` (1 genome) | `batch` (>=2, no metadata) | `batch` + metadata | `batch` + condition |
|----------|:-:|:-:|:-:|:-:|
| 11-step per-genome pipeline | yes | yes | yes | yes |
| Pairwise qualitative plots | — | yes | yes | yes |
| PCA, UMAP, heatmaps | — | — | yes | yes |
| Wilcoxon tests per codon | — | — | yes (>=5/group) | yes (>=5/group) |
| Within-condition stats | — | — | — | yes |
| Between-condition tests | — | — | — | yes (>=2 conditions) |
| PERMANOVA | — | — | — | yes (>=2 conditions) |

## Ribosomal COG accessions

CodonPipe ships with 96 COG accessions for bacterial and archaeal ribosomal proteins (bundled in `codonpipe/data/ribosomal_cogs.txt`). Override with `--cogs-file` if your reference set differs.

## RSCU column conventions

Serine, leucine, and arginine are split into 4-fold and 2-fold subfamilies in all RSCU tables (e.g., `Ser4-UCU` vs `Ser2-AGU`, `Leu4-CUU` vs `Leu2-UUA`, `Arg4-CGU` vs `Arg2-AGA`). The two groups occupy different codon boxes and cannot interconvert via single nucleotide substitutions, so pooling all 6 codons would inflate or deflate RSCU values (Shields et al. 1988, Sharp et al. 1986). ENC computation (Wright 1990) treats them as 6-fold families as intended by the original formulation.

## Dependencies

- Python >= 3.9
- Prokka >= 1.14
- COGclassifier >= 1.0
- KofamScan >= 1.3
- HMMER >= 3.3
- R >= 4.2 with coRdon, Biostrings, IRanges
- gRodon2 (optional, for growth rate prediction)
- See `environment.yml` for the full list

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
