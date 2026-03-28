# CodonPipe

End-to-end codon usage analysis pipeline for microbial genomes. Takes a genome assembly (or hundreds of them) and produces RSCU values, expression level predictions, codon usage tables, statistical comparisons, and publication-ready figures.

## What it does

CodonPipe runs eight steps per genome:

1. **Prokka** — predicts ORFs and generates amino acid / nucleotide CDS files
2. **COGclassifier** — assigns COG categories, then extracts ribosomal proteins using 96 ribosomal COG accessions
3. **KofamScan** — annotates predicted proteins with KEGG Orthology IDs
4. **RSCU analysis** — computes Relative Synonymous Codon Usage per gene, genome-level medians, concatenated ribosomal protein RSCU, codon frequency tables, and Effective Number of Codons (ENC)
5. **CU bias statistics** — uses `coRdon` to compute ENCprime (GC-corrected ENC; Novembre 2002) and MILC (Measure Independent of Length and Composition; Supek & Vlahovicek 2005) per gene
6. **Expression prediction** — uses `coRdon` to calculate MELP, CAI, and Fop (Frequency of optimal codons; Ikemura 1981) scores with ribosomal proteins as the highly-expressed reference set; classifies genes into high/medium/low expression tiers per metric
7. **Pathway enrichment** — hypergeometric test identifies KEGG pathways significantly over-represented among high- and low-expression genes for each metric, with Benjamini-Hochberg FDR correction
8. **Visualization** — generates codon frequency bar charts, RSCU bar plots, ENC–GC3 and ENC'–GC3 plots, MILC distribution, expression distributions (MELP/CAI/Fop), expression tier summaries, and pathway enrichment bar charts

In **batch mode**, CodonPipe additionally produces PCA, UMAP, and clustered heatmap comparisons across genomes, pairwise Wilcoxon rank-sum tests with Bonferroni correction for each amino acid family, and per-codon boxplots grouped by any metadata column you provide (phylum, geography, etc.).

## Installation

### Conda (recommended)

```bash
git clone https://github.com/jmarsh/codonpipe.git
cd codonpipe
mamba env create -f environment.yml
mamba activate codonpipe
pip install -e .
```

### pip only (external tools must be installed separately)

```bash
pip install .
```

You still need Prokka, COGclassifier, KofamScan, and R with `coRdon` available in your PATH.

## Usage

### Single genome

```bash
codonpipe run genome.fasta -o results/ -t 8
```

Options:
```
-s, --sample-id       Sample identifier (default: filename stem)
-t, --threads         CPU threads for external tools [default: 4]
--kingdom             Bacteria | Archaea | Viruses [default: Bacteria]
--metagenome          Use Prokka metagenome mode
--prokka-faa PATH     Pre-existing Prokka .faa file (skip Prokka)
--prokka-ffn PATH     Pre-existing Prokka .ffn file (skip Prokka)
--prokka-gff PATH     Pre-existing Prokka .gff file (optional)
--cogs-file           Custom ribosomal COG accessions (one per line)
--kofam-profile       Path to KOfam profiles directory
--kofam-ko-list       Path to KOfam ko_list file
--skip-kofamscan      Skip KofamScan step
--skip-expression     Skip all R-based analyses (MELP/CAI/Fop/ENCprime/MILC)
--kegg-ko-pathway     KO-to-pathway mapping TSV (auto-downloaded from KEGG if omitted)
--force               Overwrite existing outputs
-v, --verbose         Debug-level logging
--log-file            Write log to file
```

### Using pre-existing Prokka output

If Prokka has already been run outside the pipeline, supply the `.faa` and `.ffn` files directly and Prokka will be skipped:

```bash
codonpipe run genome.fasta -o results/ \
    --prokka-faa /path/to/existing.faa \
    --prokka-ffn /path/to/existing.ffn
```

Both `--prokka-faa` and `--prokka-ffn` must be provided together. The `.gff` file is optional.

### Batch mode

Prepare a TSV or CSV file with at minimum a `genome_path` column:

```
genome_path	sample_id	phylum	geo_category
/data/genome1.fasta	sample_A	Firmicutes	western
/data/genome2.fasta	sample_B	Bacteroidetes	eastern
```

Then run:

```bash
codonpipe batch genomes.tsv -o results/ -t 8 -p 4 \
    --metadata-cols phylum --metadata-cols geo_category
```

`-p 4` processes 4 genomes in parallel. Metadata columns trigger comparative statistics and visualizations.

### Batch mode with pre-existing Prokka output

Add `prokka_faa` and `prokka_ffn` columns to the batch table. Samples where both columns are populated skip Prokka; samples with empty values run Prokka as usual. You can mix both in one table:

```
genome_path	sample_id	prokka_faa	prokka_ffn	phylum
/data/genome1.fasta	sample_A	/data/g1.faa	/data/g1.ffn	Firmicutes
/data/genome2.fasta	sample_B			Bacteroidetes
/data/genome3.fasta	sample_C	/data/g3.faa	/data/g3.ffn	Proteobacteria
```

In this example, sample_A and sample_C use existing Prokka files; sample_B runs Prokka from scratch.

## Output structure

```
results/
├── sample_A/
│   ├── prokka/                    # Prokka output files
│   ├── cogclassifier/             # COG classification results
│   ├── ribosomal_proteins/        # Extracted RP sequences + IDs
│   ├── kofamscan/                 # KofamScan annotations
│   ├── rscu/
│   │   ├── sample_A_rscu_all_genes.tsv
│   │   ├── sample_A_rscu_median.tsv
│   │   ├── sample_A_rscu_ribosomal.tsv
│   │   ├── sample_A_codon_frequency.tsv
│   │   ├── sample_A_enc.tsv
│   │   └── sample_A_rscu_annotated.tsv
│   ├── cu_statistics/
│   │   ├── sample_A_encprime.tsv
│   │   └── sample_A_milc.tsv
│   ├── expression/
│   │   ├── sample_A_melp.tsv
│   │   ├── sample_A_cai.tsv
│   │   ├── sample_A_fop.tsv
│   │   ├── sample_A_expression.tsv       # Combined: MELP, CAI, Fop + per-metric classes
│   │   └── sample_A_expression_annotated.tsv
│   ├── enrichment/
│   │   ├── sample_A_expression_by_tier.tsv  # All genes with classes + KO accessions
│   │   ├── sample_A_CAI_high_enrichment.tsv
│   │   ├── sample_A_CAI_low_enrichment.tsv
│   │   ├── sample_A_MELP_high_enrichment.tsv
│   │   ├── sample_A_MELP_low_enrichment.tsv
│   │   ├── sample_A_Fop_high_enrichment.tsv
│   │   └── sample_A_Fop_low_enrichment.tsv
│   └── plots/
│       ├── sample_A_codon_frequency.png
│       ├── sample_A_rscu_all.png
│       ├── sample_A_rscu_ribosomal.png
│       ├── sample_A_rscu_heatmap.png
│       ├── sample_A_enc_gc3.png
│       ├── sample_A_encprime_gc3.png
│       ├── sample_A_milc_dist.png
│       ├── sample_A_expression_dist.png
│       ├── sample_A_expression_tiers.png
│       ├── sample_A_enrichment_CAI_high.png
│       └── sample_A_enrichment_CAI_low.png   # (etc. for MELP, Fop)
├── combined_rscu.tsv              # Batch: merged genome-level RSCU
├── combined_rscu_ribosomal.tsv    # Batch: merged ribosomal RSCU
├── statistics/
│   ├── rscu_zscored.tsv
│   └── phylum_Ser_wilcoxon.tsv    # Per-amino-acid pairwise tests
└── plots/
    ├── pca_phylum.png
    ├── umap_geo_category.png
    ├── heatmap_clustered_phylum.png
    ├── boxplot_geo_category_Ser.png
    └── significance_phylum_all_wilcoxon.png
```

## Analyses produced

**Per genome:**
- Codon frequency table (absolute counts, frequency, per-thousand, RSCU)
- Per-gene RSCU for all CDS
- Genome-level median RSCU (59 sense codons)
- Concatenated RSCU for the ribosomal protein set
- Effective Number of Codons (ENC) with GC3 content
- ENCprime (GC-corrected ENC, Novembre 2002) per gene
- MILC (Measure Independent of Length and Composition, Supek & Vlahovicek 2005) per gene
- MELP, CAI, and Fop (Frequency of optimal codons, Ikemura 1981) expression scores per gene
- Per-metric expression classification (MELP_class, CAI_class, Fop_class): high ≥ 95th percentile, low ≤ 5th percentile
- KO functional annotations merged onto expression and RSCU tables
- Hypergeometric pathway enrichment for high- and low-expression gene sets per metric (Benjamini-Hochberg FDR)
- Expression-by-tier table listing every gene with its scores, classes, and KO accession

**Batch comparisons:**
- PCA and UMAP dimensionality reduction colored by metadata
- Hierarchical clustering heatmaps (Manhattan distance, complete linkage)
- Z-score normalized RSCU tables
- Pairwise Wilcoxon rank-sum tests per codon, per amino acid family
- Bonferroni-corrected significance heatmaps
- Per-amino-acid boxplots grouped by metadata

## Ribosomal COG accessions

CodonPipe ships with 96 COG accessions for bacterial/archaeal ribosomal proteins (bundled in `codonpipe/data/ribosomal_cogs.txt`). Override with `--cogs-file` if your reference set differs.

## Dependencies

- Python ≥ 3.9
- Prokka ≥ 1.14
- COGclassifier ≥ 1.0
- KofamScan ≥ 1.3
- R ≥ 4.2 with coRdon, Biostrings, IRanges, data.table
- See `environment.yml` for the full list

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
