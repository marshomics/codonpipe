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

## Per-genome pipeline (13 steps)

Every genome, whether processed via `run` or `batch`, goes through these steps:

1. **Prokka** ORF prediction (or use pre-existing .faa/.ffn files)
2. **COGclassifier** assigns COG categories, then extracts ribosomal proteins using 96 ribosomal COG accessions
3. **KofamScan** annotates predicted proteins with KEGG Orthology IDs
4. **RSCU analysis** computes per-gene RSCU for all CDS, genome-level median RSCU (59 sense codons), concatenated ribosomal protein RSCU, codon frequency tables, and ENC with GC3 content
5. **CU bias statistics** via coRdon: ENCprime (GC-corrected ENC, Novembre 2002) and MILC (Measure Independent of Length and Composition, Supek & Vlahovicek 2005) per gene
6. **Expression prediction** via coRdon: MELP, CAI, and Fop (Frequency of optimal codons, Ikemura 1981) using ribosomal proteins as the highly expressed reference set; genes classified into high (>=90th percentile), medium, and low (<=10th percentile) expression tiers per metric. Tier cutoffs are configurable through the `_classify_by_percentile` helper if a more or less stringent split is needed.
7. **Pathway enrichment** via hypergeometric test for KEGG pathways over-represented in high- and low-expression gene sets, with Benjamini-Hochberg FDR correction
8. **Advanced analyses**: correspondence analysis (COA) on codon usage, S-value adaptation to the ribosomal reference set, GC12-vs-GC3 neutrality analysis, PR2 (purine/pyrimidine ratio), delta-RSCU distance from the genome average, tRNA-codon co-adaptation correlation (if GFF provided), COG enrichment in high/low bias genes, gene length vs codon bias, and ENC-ENCprime difference
9. **Biological/ecological analyses**: HGT candidate detection via Mahalanobis distance on RSCU profiles (using the 38 independent codon dimensions, since the full 59-codon space is rank-deficient by ~21 due to per-AA-family sum constraints), growth rate prediction via two complementary models (a CAI-based proxy and gRodon2 if installed; see caveats below), translational selection analysis (Fop gradient, positional effects across 5'/middle/3' gene regions), phage and mobile element detection, strand asymmetry, and operon co-adaptation
10. **Bio/ecology analyses** — see step 9 above; a single block in the codebase.
11. **Codon usage tables** in six formats: RSCU, absolute counts, per-thousand frequencies, W values (relative adaptiveness), adaptation weights, and CBI (Codon Bias Index).
12. **Codon-optimization comparison** — three-way comparison of genome / RP / Mahal-cluster reference frames. Emits a synthesis-ready preferred-codon table (Mahal-default with RP and genome alternates), per-AA agreement summaries (RP-vs-Mahal and genome-vs-Mahal), per-gene CAI under all three frames computed from the `.ffn` file, and five publication-ready figures. Output goes to a dedicated `codon_optimization/` subfolder. Skipped automatically when any reference is unavailable (e.g. when `--skip-mahal` is set), or explicitly with `--skip-codon-optimization`.
13. **Publication-ready plots** at 300 DPI in PNG and SVG (editable in Adobe Illustrator).

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
| `--skip-codon-optimization` | off | Skip Step 12 (codon-optimization comparison). Output normally goes to `<sample_dir>/codon_optimization/`. |
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

### `codonpipe gene-set`

```
codonpipe gene-set SAMPLE_DIR --goi-file GOI.txt [OPTIONS]
```

Compares a shortlist of genes-of-interest (specified by Prokka locus tags) against the rest of the genome's codon usage. Reads the per-sample output directory produced by `codonpipe run` (or `codonpipe batch`) — no recomputation of upstream steps.

| Option | Default | Description |
|--------|---------|-------------|
| `--goi-file` | (required) | Text file with locus tags, one per line. Lines beginning with `#` are comments; trailing whitespace-delimited tokens after the locus tag are ignored. |
| `-s, --sample-id` | dir name | Sample identifier (defaults to the directory name). |
| `-o, --output-dir` | `<SAMPLE_DIR>/gene_set/` | Output directory. |
| `--n-permutations` | 999 | Permutation count for Aitchison and HGT-flag tests. |
| `--no-length-matching` | off | Use uniform-random null instead of length-matched. Recommended only as a sensitivity check. |
| `--no-figure` | off | Skip the six-panel summary figure. |
| `--rng-seed` | 42 | Random seed for permutation tests and bootstrap CIs. |
| `-v, --verbose` | off | Debug-level logging. |

The analysis answers four questions with the data the per-genome pipeline already produced.

(1) Are the GOI translationally optimized? Mann-Whitney + Cliff's delta with bootstrap 95% CI on each scalar metric (CAI, MELP, Fop, ENC, ENCprime, MILC, GC3, mahalanobis_dist, mahal_cluster_distance, membership_score, cbi_rp, cbi_mahal, length, gc3_deviation), tested against rest-of-genome with global BH FDR.

(2) Are the GOI horizontally acquired? Each GOI is positioned in the bio_ecology HGT detector's 38-d Mahalanobis space, plus a one-sided permutation test on the GOI's HGT-flag rate against length-matched control sets for hgt_flag_combined, hgt_flag_fdr, and gc3_outlier.

(3) Are the GOI part of the translationally optimized core? When the pipeline's Mahalanobis-clustering step has run (and its outputs are in `gmm_clustering/`), each GOI gets two complementary readouts: distance to the *cluster centroid* (`mahal_cluster_distance`, biologically distinct from the genome-centroid `mahalanobis_dist` used for HGT detection — one tells you "how far is this gene from the optimized core?", the other tells you "how unusual is this gene's codon usage relative to the bulk genome?"), and a two-sided permutation test on the GOI's `in_optimized_set` rate vs length-matched controls. The Mahal-cluster CBI (`cbi_mahal`) is also included alongside the RP-derived `cbi_rp` so direct comparison of the two reference frames is one column lookup.

(4) Do the GOI share a codon-usage signature distinct from references? Concatenated GOI mean-RSCU vector compared via Aitchison distance (CLR + Euclidean on the 38 independent codon dimensions) to the genome / RP / Mahal-cluster references, with a length-matched permutation null reporting both `p_more_divergent` and `p_more_similar` so the user sees which direction the deviation runs.

Outputs written to `--output-dir`:

- `<sample_id>_goi_summary.tsv` — one row per GOI gene with every available metric and its within-genome percentile rank (computed against rest-of-genome, never against itself). Includes the distinction between `mahalanobis_dist` (genome centroid, from `hgt_candidates.tsv`) and `mahal_cluster_distance` (optimized-cluster centroid, from `gmm_clusters.tsv`), plus `in_optimized_set` and both CBI variants.
- `<sample_id>_goi_distribution_tests.tsv` — per-metric Mann-Whitney + Cliff's delta + KS, BH-corrected globally.
- `<sample_id>_goi_rscu_comparison.tsv` — per (codon × reference) Mann-Whitney with global BH; useful for identifying which codons drive any signal.
- `<sample_id>_goi_aitchison_perm.tsv` — observed Aitchison distance to each reference, permutation null statistics, two-sided p-value and one-sided directional p-values.
- `<sample_id>_goi_hgt_enrichment.tsv` — one-sided permutation test on each HGT flag's GOI rate vs length-matched controls.
- `<sample_id>_goi_mahal_membership.tsv` — two-sided permutation test on the GOI's Mahal-cluster membership rate against length-matched controls. Reports both `p_more_in_cluster` (GOI is enriched in the optimized core) and `p_less_in_cluster` (GOI is depleted from it).
- `<sample_id>_goi_partition.tsv` — three-way genome partition: each gene is assigned to `mahal_cluster` (in the optimized core), `bulk` (typical genome member), or `outlier` (HGT-flagged via `hgt_flag_combined`, requires unusual codon usage AND unusual GC3). Cluster membership takes precedence when both flags are set. Reports per-category Fisher's exact test (with BH FDR across the three categories) plus an omnibus chi-squared / Cramer's V test of the full 2×3 (GOI × category) table. The `partition` column is also merged into `<sample_id>_goi_summary.tsv` so each GOI row carries its category assignment without needing a join.
- `<sample_id>_goi_panel.png` / `.svg` — seven-panel summary figure (A: RSCU bars vs reference sets; B: effect-size forest plot per metric; C: COA biplot with GOI highlighted; D: Wright ENC-vs-GC3 plot; E: three-way genome-partition stacked bar with per-category BH-corrected Fisher p-values and the omnibus chi-squared Cramer's V; F: genome-centroid vs cluster-centroid Mahalanobis biplot, separating "unusual relative to genome" from "near the optimized core"; G: per-GOI percentile-rank heatmap with a green/grey strip marking Mahal-cluster membership). When the Mahal-clustering / HGT outputs aren't available, Panel E falls back to the genome-centroid Mahalanobis distance histogram. Panel F is the cleanest single-glance answer to "where do my GOIs sit relative to both reference frames at once?".

The pipeline also produces three additional publication-ready figures, each focused on one of the most common follow-up questions about a GOI list:

- `<sample_id>_goi_expression_tiers.png` / `.svg` and `<sample_id>_goi_expression_tier_tests.tsv` — paired stacked bars per expression metric (MELP, CAI, Fop, plus their `rp_`-prefixed RP-relative variants) showing GOI vs background composition across high/medium/low tiers. Per (metric, tier) Fisher's exact test with BH FDR across the full panel; significance asterisks (*, **, ***) are drawn directly on each significant tier segment. Answers "which of my GOIs are predicted highly expressed?".
- `<sample_id>_goi_anomaly_map.png` / `.svg` — two-panel anomaly visualization. Left: scatter of Mahalanobis distance (genome centroid) vs GC3 deviation, all genome genes plotted as background, GOIs colored by partition (mahal_cluster / bulk / outlier), with 95th-percentile reference lines on each axis to delimit the typical-codon-usage region. Right: per-flag count bars (`hgt_flag_combined`, `hgt_flag_fdr`, `gc3_outlier`) showing GOI flag rates against the background prevalence. Answers "which of my GOIs are HGT candidates or in regions of distinct codon usage?".
- `<sample_id>_goi_internal_structure.png` / `.svg`, `<sample_id>_goi_internal_clusters.tsv`, `<sample_id>_goi_internal_drivers.tsv` — four-panel within-GOI structure analysis. Hierarchical clustering (Ward linkage) on Aitchison distance between GOIs in CLR-Δ space; the number of clusters is auto-selected by silhouette score with a min-cluster-size guard. The figure shows: left, Ward dendrogram; centre, heatmap of per-GOI CLR-Δ codon profiles with rows ordered by the dendrogram and columns ordered by their own clustering; right strip, cluster id + partition + expression class annotations; right panel, top codon and scalar drivers per cluster (Mann-Whitney + BH, ranked by Cliff's delta). Answers "do my GOIs form discrete groups, and what codon preferences or scalar metrics distinguish them?". Skipped when fewer than 6 GOIs are matched, since silhouette-driven cluster selection isn't reliable below that.

The `--goi-file` accepts annotated lines, e.g.:

```
# my favourite operon — sample G0370_i3
G0370_i3_00132   # phbA
G0370_i3_00133   # phbB
G0370_i3_00134   # phbC
```

Length-matched permutations bin the background by length quartile and reproduce the per-bin counts of the GOI list, since CAI, MELP, ENC, and Mahalanobis distance all have measurable length dependence. Pass `--no-length-matching` only as a sensitivity check.

### `codonpipe signatures`

```
codonpipe signatures SAMPLE_DIR [OPTIONS]
```

Emit cross-genome-comparable gene + genome signature files for one sample. Designed for downstream cross-genome aggregation: every CodonPipe run produces these files (or you generate them after the fact), then `codonpipe corpus` concatenates and clusters them.

| Option | Default | Description |
|--------|---------|-------------|
| `-s, --sample-id` | dir name | Sample identifier. |
| `-o, --output-dir` | `<SAMPLE_DIR>/signatures/` | Output directory. |
| `-v, --verbose` | off | Debug-level logging. |

Outputs:

- `<sample_id>_gene_signature.tsv` — one row per gene. Columns: `sample_id`, `gene`, `length`, scalar metrics (ENC, GC3, ENCprime, MILC, MELP, CAI, Fop, cbi_rp, cbi_mahal, mahalanobis_dist_genome, mahalanobis_dist_cluster, membership_score, in_optimized_set), and a 38-dimensional CLR-Δ codon-preference vector (`delta_clr_<codon>`). The CLR-Δ vector is each gene's centered-log-ratio RSCU minus the genome's mean CLR-RSCU — directly comparable across genomes because CLR removes per-AA-family sum constraints and the subtraction removes mutational background.
- `<sample_id>_genome_signature.tsv` — single row per genome. Two blocks of features. Geometry block (cross-genome-comparable, ~80 columns): `delta_clr_mahal_<codon>` × 38 (Mahal-cluster CLR minus genome CLR), `delta_clr_rp_<codon>` × 38 (RP CLR minus genome CLR), and three Aitchison distances (`aitchison_genome_to_mahal`, `aitchison_genome_to_rp`, `aitchison_rp_to_mahal`). Ecology block (~20 scalars): median CAI/MELP/Fop/ENC/ENCprime/MILC/GC3, GC3 IQR, fraction in optimized set, gRodon2 doubling time and CIs, HGT candidate fraction, phage / mobile-element gene counts, strand-asymmetry significant codon count, tRNA gene count.

The geometry block is the cleanest signature for clustering on translational selection patterns — both `delta_clr_*` vectors share the host's GC composition by construction, so cross-genome distance reflects selection rather than mutation. The ecology block adds organism-level summary scalars for "ecology-aware" clustering.

### `codonpipe corpus`

```
codonpipe corpus INPUT_DIRS... -o OUTPUT_DIR [OPTIONS]
```

Aggregate per-genome signatures across thousands of genomes, standardize, dimension-reduce, cluster, and validate. Discovers `*_genome_signature.tsv` (and optionally `*_gene_signature.tsv`) files under one or more input directories.

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | (required) | Where to write `corpus_*.tsv` outputs and the summary figure. |
| `--features` | `geometry` | `geometry` (CLR-Δ + Aitchison only, recommended for codon-strategy clustering) or `all` (full vector including ecology scalars). |
| `--use-umap` | off | Apply UMAP after PCA for the 2-D embedding. Requires `umap-learn`; without this flag, the first two PCs are used. |
| `--include-gene-level` | off | Concatenate the gene-level signatures into `corpus_gene_signature.tsv`. Off by default because gene-level corpora can run to millions of rows for thousands of genomes. Implied by `--gene-level-clustering` and `--gene-level-by-category`. |
| `--gene-level-clustering` | off | Cluster genes across the corpus by their CLR-Δ codon vectors. Emits `corpus_gene_clusters.tsv` and a four-panel UMAP figure with overlays for cluster id, host genome, host phylum, and host GC3. Useful for finding gene "codon-strategy classes" that span multiple genomes. |
| `--gene-level-by-category` | — | One of `KO` or `COG`. Cluster genes within each functional-category class to see whether genes of the same KO/COG share a codon strategy across the corpus or split into ecological sub-strategies. Emits `corpus_gene_clusters_by_<cat>.tsv` plus a per-category summary and a diversity bar chart. |
| `--gene-level-min-category-size` | 10 | Minimum genes per category for per-category clustering. |
| `--gene-level-max-genes` | 1,000,000 | Safety cap on the gene-level corpus size for general clustering. |
| `--phylogeny` | — | Path to a Newick file (`.nwk`/`.newick`/`.tree`, requires biopython) or a precomputed pairwise distance-matrix TSV (with sample_ids as both row and column labels). When provided, runs a Mantel test of signature distance vs phylogenetic distance. |
| `--metadata` | — | Path to a metadata TSV with a `sample_id` column. Categorical columns are tested for cluster association via chi-squared + Cramer's V. |
| `--hdbscan-min-cluster-size` | auto | Override the heuristic. Default scales to ~2% of corpus, floored at 5, capped at 50. |
| `--rng-seed` | 42 | Random seed. |
| `-v, --verbose` | off | Debug logging. |

Outputs:

- `corpus_genome_signature.tsv` — stacked per-genome features (one row per genome).
- `corpus_genome_clusters.tsv` — `sample_id`, `cluster`, `embed_dim1`, `embed_dim2`, plus any metadata columns. Cluster `-1` means HDBSCAN flagged the sample as noise (not assigned to any cluster).
- `corpus_validation.tsv` — Mantel test result (signature distance vs phylogenetic distance) and chi-squared / Cramer's V tests of cluster-vs-metadata associations. BH-corrected.
- `corpus_gene_signature.tsv` — only if `--include-gene-level`. Stacked per-gene features across the corpus.
- `corpus_cluster_drivers.tsv` — per (cluster, feature) Mann-Whitney comparing each cluster's members against the rest of the corpus, with Cliff's delta effect sizes and BH-corrected p-values applied globally. Sorted by `(cluster_id, |effect|)` so the strongest distinguishing features per cluster appear first.

The pipeline produces five publication-ready corpus figures (replacing the old single three-panel summary), each focused on one analytical question:

- `corpus_multi_overlay.png` / `.svg` — same 2-D embedding overlaid with up to six different colorings: cluster id, a categorical metadata column (phylum/habitat/etc. when available), median GC3, gRodon2 doubling time, fraction of genes in the optimized cluster, and HGT candidate fraction. Acts as a built-in confounder check: if the GC3 panel shows a clean gradient that mirrors the cluster panel, your clusters are recovering GC content rather than translational selection.
- `corpus_cluster_signature.png` / `.svg` — two stacked heatmaps. Top: cluster mean CLR-Δ codon profiles with hierarchical clustering on both axes for interpretable banding. Bottom: full genome × codon CLR-Δ matrix with rows ordered by cluster (rasterized for thousands of genomes). Together these show what each cluster prefers / avoids relative to its host genome's bulk.
- `corpus_cluster_drivers.png` / `.svg` — per-cluster forest plot of the top BH-significant differential features, ordered by absolute Cliff's delta. Each cluster gets its own colour. Falls back to top-by-effect when no driver crosses BH-0.05.
- `corpus_mantel_scatter.png` / `.svg` — pairwise distance scatter, signature vs phylogenetic, stratified by within-group vs between-group when a categorical metadata column (phylum, class, etc.) is available. Hexbin density for corpora >1500 pairs, individual points otherwise. Reports the overall Mantel `r` plus separate within/between trend lines so you can see whether phylogenetic signal sits at the major-clade level or persists within clades. Only emitted when `--phylogeny` is provided.
- `corpus_focus_<sample_id>.png` / `.svg` (one per `--focus-genome`) — single-genome locator. Left: the genome highlighted on the embedding with a red star, all other genomes shown in their cluster colours. Right: violin plots of the corpus distribution for each scalar feature, with the focus genome's z-score marked as a red star and its percentile rank annotated. Pass `--focus-genome SAMPLE_ID` (repeatable) for one figure per genome.

Defensible-by-default. The `geometry` feature set removes mutational background by construction — both `delta_clr_*` vectors share the host's GC composition, so subtraction cancels the GC-content axis that otherwise dominates raw RSCU clustering. Use `--phylogeny` to confirm: a low Mantel `r` between signature distance and phylogenetic distance is evidence the clustering is capturing translational ecology rather than just recovering taxonomy. A high `r` with the geometry feature set is also informative — it suggests selection signature tracks phylogeny in your dataset, which is biologically meaningful (vertical inheritance of codon-usage strategy) rather than an artefact.

Recommended workflow for thousands of genomes:

```
# 1. Generate signatures for every CodonPipe run (parallelizable per sample)
for sample in output_run/*/; do
    codonpipe signatures "$sample" -o "$sample/signatures"
done

# 2. Aggregate, cluster, validate
codonpipe corpus output_run/*/signatures \\
    -o corpus_results/ \\
    --features geometry \\
    --phylogeny species_tree.nwk \\
    --metadata genome_metadata.tsv \\
    --use-umap
```

Validation strategy: with thousands of genomes, expect HDBSCAN to find anywhere from 5 to 50 clusters with the default heuristic. Inspect `corpus_validation.tsv` to see how cluster membership associates with phylum, isolation source, growth rate, etc. A defensible workflow always checks (1) Mantel `r` against phylogeny — too high (~0.95+) means the signature is mostly recovering taxonomy and you should revisit the feature set, (2) cluster enrichment against ecology metadata — clusters that align cleanly with habitat or growth-rate class are likely capturing real biology, (3) phylogenetically-controlled tests if you have a tree — Pagel's λ on each feature column tells you whether the variance is phylogenetic or trait-like.

### Codon-optimization outputs (Step 12)

The codon-optimization analysis runs automatically as Step 12 of every
`codonpipe run` and `codonpipe batch` invocation. There is no longer a
standalone `codonpipe codon-optimization` subcommand. To opt out, pass
`--skip-codon-optimization` on `run` or `batch`. To re-derive the
analysis from scratch on an existing sample directory, import
`codonpipe.modules.codon_optimization.run_codon_optimization` directly.

The analysis compares the genome / ribosomal-protein / Mahalanobis-cluster
codon-usage profiles for one organism, quantifies the gain from
Mahal-cluster-derived codon optimization vs the classic RP-derived
approach, and emits a synthesis-ready preferred-codon table. Output
goes to `<sample_dir>/codon_optimization/`.

Files produced:

- `<sample_id>_codon_optimization_table.tsv` — per-codon comparison: amino_acid, family, codon, codon_col, genome_rscu, rp_rscu, mahal_rscu, rp_w, mahal_w, rp_optimal, mahal_optimal, family_agree, delta_w_mahal_minus_rp, delta_rscu_mahal_minus_rp, delta_rscu_mahal_minus_genome. The full data behind every plot.
- `<sample_id>_codon_optimization_summary.tsv` — per-AA-family summary: which codon is optimal under each scheme, do they agree, and the maximum within-family Δw shift.
- `<sample_id>_codon_optimization_recommend.tsv` — synthesis-ready table with `recommended_codon` (Mahal-derived by default), `alternative_codon` (RP-derived when it differs), `rationale`, `confidence` (high/medium/low based on Δw magnitude), and the rp_w / mahal_w values at the recommendation. Drop into a synthesis order.
- `<sample_id>_codon_optimization_gain.tsv` — per-gene CBI under both reference frames: cbi_rp, cbi_mahal, gain_mahal_minus_rp, gain_pct, plus Mahal-cluster membership_score and in_optimized_set when available. Quantifies which genes already match each reference better.
- `<sample_id>_three_way_rscu.png` / `.svg` — per-codon RSCU bar chart with all three references overlaid. Bottom panel shows the per-codon Δw shift (Mahal − RP). Families where RP and Mahal disagree on the optimum are highlighted with a coloured background.
- `<sample_id>_optimization_agreement.png` / `.svg` — compact per-AA-family table showing the RP-optimal codon vs the Mahal-optimal codon side by side, with ✓/✗ for agreement and a red bar showing the within-family max-Δw magnitude. Reads at a glance.
- `<sample_id>_optimization_gain.png` / `.svg` — three-panel quantification of the gain: scatter of per-gene cbi_rp vs cbi_mahal coloured by membership_score with the diagonal line; histogram of the gain distribution annotated with the fraction of genes that benefit from Mahal-style optimization; bar chart of the top genes most improved.

A parallel set of outputs compares Mahal-cluster-derived weights against the **genome-mean** reference frame (the naive "just match the genome's overall codon distribution" baseline), so you can quantify how much Mahal-style optimization actually moves the needle over a naive approach:

- `<sample_id>_codon_optimization_summary_vs_genome.tsv` — per-AA-family agreement: genome_optimal_codon vs mahal_optimal_codon, agree (bool), max_codon_w_shift_vs_genome.
- `<sample_id>_codon_optimization_recommend_vs_genome.tsv` — synthesis-ready recommendation table with the genome-mean as the comparison anchor instead of the RP set.
- `<sample_id>_codon_optimization_three_way_cai.tsv` — per-gene CAI under all three reference frames (cai_genome, cai_rp, cai_mahal) computed directly from the .ffn file, plus pairwise gains: gain_mahal_vs_genome, gain_mahal_vs_rp, gain_rp_vs_genome. Joins onto Mahal-cluster context (in_optimized_set, membership_score) when available.
- `<sample_id>_optimization_agreement_vs_genome.png` / `.svg` — per-AA-family table showing genome-most-frequent codon vs Mahal-optimal codon side by side with ✓/✗ and a max-Δw bar.
- `<sample_id>_optimization_gain_vs_genome.png` / `.svg` — three-panel quantification of Mahal-vs-genome optimization gain: per-gene cai_genome vs cai_mahal scatter, gain distribution, top-N most-improved genes.

Why both comparisons matter. The Mahal-vs-RP comparison is the classical "is the Mahal cluster a better anchor than ribosomal proteins?" question. The Mahal-vs-genome comparison is the practical "is Mahal-style optimization worth doing at all over naive genome-matching?" question. For low-GC organisms whose genome bulk already aligns with the optimized cluster, the answer to the first is often "yes by a wide margin" while the second is "yes by a much smaller margin". For high-GC organisms with strong selection on the cluster, both are typically large.

Defensible reason to prefer Mahal-derived weights for codon optimization. The Mahal cluster is identified by codon-usage similarity rather than by RP gene annotations, so it captures the organism's actual translationally-optimized cohort and excludes RP annotation outliers (truncations, paralogs, modified variants) that can pull the RP-based reference centroid off-target. When the two reference frames agree on the optimal codon for an amino acid (the common case), the recommended codon is unambiguous. When they disagree, the recommendation table flags the disagreement with a `confidence` value so the user can make the call rather than blindly defaulting to one frame. The per-gene gain figure shows whether Mahal-style optimization actually pays off in the user's organism: if the median (cbi_mahal − cbi_rp) is positive and most genes sit above the diagonal in the scatter, Mahal-derived codons match the host's optimization signal better than RP-derived codons.

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
