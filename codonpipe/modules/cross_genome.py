"""Cross-genome aggregation: per-genome signatures + corpus clustering.

Two layers:

1. Per-genome signature builders (compute_gene_signature, compute_genome_signature)
   compute cross-genome-comparable feature vectors from the existing per-genome
   pipeline outputs. Both use centered-log-ratio (CLR) representations of RSCU
   and subtract the genome's own mean so that the resulting features are
   normalized away from each genome's mutational background.

2. Corpus aggregator (build_corpus, cluster_corpus, mantel_test) consumes the
   per-genome signature files from many CodonPipe runs, standardizes them
   across the corpus, runs PCA (and UMAP if available) for dimension reduction,
   clusters the result with HDBSCAN (KMeans fallback), and optionally compares
   the clustering to a phylogeny via a Mantel test.

Defensible-comparison rationale: codon usage across bacteria is dominated by
two confounded forces — mutational bias (genome-wide GC content) and
translational selection. Raw RSCU clustered across organisms with GC content
spanning 0.15–0.85 mostly recovers GC, not biology. Both layers here operate
on within-genome-normalized features (CLR(gene) − CLR(genome_mean) for the
gene vector; CLR(Mahal_cluster) − CLR(genome_mean) for the genome signature)
so that the cross-genome distance reflects selection, not mutation.

References:
    Aitchison 1986 — CLR transform for compositional data.
    Karlin & Mrazek 2000 — predicted-high-expression metric across genomes.
    Sharp et al. 2005 — translational selection statistics.
    Roller et al. 2013 — universal codon-usage signatures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe.modules.gene_set import (
    _drop_redundant_codon_per_family,
    _resolve_path,
    aitchison_distance,
    clr_transform,
    load_sample_outputs,
)
from codonpipe.utils.codon_tables import (
    COL_GENE,
    RSCU_COLUMN_NAMES,
)
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")


# ──────────────────────────────────────────────────────────────────────────────
# Per-gene signature
# ──────────────────────────────────────────────────────────────────────────────


def compute_gene_signature(
    rscu_gene_df: pd.DataFrame,
    rscu_genome: dict[str, float] | None,
    sample_id: str,
    *,
    enc_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    milc_df: pd.DataFrame | None = None,
    hgt_df: pd.DataFrame | None = None,
    mahal_cluster_df: pd.DataFrame | None = None,
    cbi_rp_df: pd.DataFrame | None = None,
    cbi_mahal_df: pd.DataFrame | None = None,
    annotation_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-gene cross-genome-comparable feature table.

    Each row is one gene. Columns:
        sample_id, gene, length,
        ENC, GC3, ENCprime, MILC, MELP, CAI, Fop,
        cbi_rp, cbi_mahal,
        mahalanobis_dist_genome, mahalanobis_dist_cluster, membership_score,
        in_optimized_set,
        delta_clr_<codon> × 38   (gene CLR − genome-mean CLR, the cross-genome-
                                  comparable codon-preference vector)

    The CLR-difference vector is the gene's compositional shift away from its
    host's mean codon preference, in centered-log-ratio space. It's directly
    comparable across genomes:
      - CLR removes per-AA-family sum constraints
      - Subtracting genome mean removes mutational background
      - 38 independent codons (one dropped per AA family) avoids rank
        deficiency in downstream covariance / clustering work.

    Args:
        rscu_gene_df: Per-gene RSCU table (gene, length, 59 RSCU columns).
        rscu_genome: Genome-mean (or median) RSCU as {column_name: value}.
            If None, the per-gene mean of *rscu_gene_df* is used. Pass
            None only as a sanity-check fallback; for cross-genome work
            you want the canonical pipeline output (rscu_median.tsv).
        sample_id: Genome identifier; written to every row so concatenated
            corpus tables retain provenance.
    """
    if rscu_gene_df is None or rscu_gene_df.empty:
        return pd.DataFrame()

    rscu_cols_full = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    rscu_cols = _drop_redundant_codon_per_family(rscu_cols_full)
    if not rscu_cols:
        return pd.DataFrame()

    base = rscu_gene_df[[COL_GENE, "length"]].copy()
    if enc_df is not None and not enc_df.empty:
        enc_cols = [c for c in (COL_GENE, "ENC", "GC3") if c in enc_df.columns]
        base = base.merge(enc_df[enc_cols], on=COL_GENE, how="left")
    for df, score_cols in (
        (expr_df, ["MELP", "CAI", "Fop", "rp_MELP", "rp_CAI", "rp_Fop"]),
        (encprime_df, ["ENCprime"]),
        (milc_df, ["MILC"]),
        (hgt_df, ["mahalanobis_dist", "gc3_deviation"]),
        (mahal_cluster_df, ["mahal_cluster_distance", "membership_score",
                            "in_optimized_set"]),
        (cbi_rp_df, ["cbi_rp"]),
        (cbi_mahal_df, ["cbi_mahal"]),
        # Functional annotations (KO, COG) — used by gene-level cross-genome
        # clustering and per-category breakdowns. Merged in unchanged.
        (annotation_df, ["KO", "KO_definition", "COG_ID", "COG_letter",
                         "COG_category", "COG_NAME", "product"]),
    ):
        if df is None or df.empty:
            continue
        cols = [COL_GENE] + [c for c in score_cols if c in df.columns]
        if len(cols) > 1:
            base = base.merge(df[cols], on=COL_GENE, how="left")

    # Genome reference: prefer caller-provided dict (which matches the
    # rscu_median.tsv output of the pipeline); fall back to per-gene mean.
    if rscu_genome is not None and len(rscu_genome) > 0:
        genome_vec = np.array([rscu_genome.get(c, np.nan) for c in rscu_cols])
    else:
        genome_vec = rscu_gene_df[rscu_cols].mean().values
    if np.any(np.isnan(genome_vec)):
        # Backfill any missing codons from per-gene mean (rare, only when
        # rscu_median.tsv has been hand-edited or the genome had unobserved AAs).
        fallback = rscu_gene_df[rscu_cols].mean().values
        nan_mask = np.isnan(genome_vec)
        genome_vec[nan_mask] = fallback[nan_mask]
    clr_genome = clr_transform(genome_vec)

    # CLR-Δ per gene
    gene_rscu_mat = rscu_gene_df[rscu_cols].values
    delta_clr_mat = np.empty_like(gene_rscu_mat, dtype=float)
    for i in range(gene_rscu_mat.shape[0]):
        delta_clr_mat[i, :] = clr_transform(gene_rscu_mat[i, :]) - clr_genome

    delta_cols = [f"delta_clr_{c}" for c in rscu_cols]
    delta_df = pd.DataFrame(delta_clr_mat, columns=delta_cols)
    delta_df.insert(0, COL_GENE, rscu_gene_df[COL_GENE].values)

    # Rename HGT mahalanobis_dist to disambiguate from cluster-centroid distance
    if "mahalanobis_dist" in base.columns:
        base = base.rename(columns={"mahalanobis_dist": "mahalanobis_dist_genome"})
    if "mahal_cluster_distance" in base.columns:
        base = base.rename(columns={"mahal_cluster_distance": "mahalanobis_dist_cluster"})

    out = base.merge(delta_df, on=COL_GENE, how="left")
    out.insert(0, "sample_id", sample_id)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Per-genome signature
# ──────────────────────────────────────────────────────────────────────────────


def compute_genome_signature(
    sample_id: str,
    rscu_genome: dict[str, float] | None,
    rscu_rp: dict[str, float] | None,
    rscu_mahal_cluster: dict[str, float] | None,
    *,
    rscu_gene_df: pd.DataFrame | None = None,
    enc_df: pd.DataFrame | None = None,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    milc_df: pd.DataFrame | None = None,
    mahal_cluster_df: pd.DataFrame | None = None,
    grodon_df: pd.DataFrame | None = None,
    hgt_df: pd.DataFrame | None = None,
    phage_mobile_df: pd.DataFrame | None = None,
    strand_asymmetry_df: pd.DataFrame | None = None,
    trna_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Single-row per-genome cross-genome-comparable signature.

    Returns a DataFrame with one row, columns partitioned into:
      Geometry block (cross-genome-comparable):
        delta_clr_mahal_<codon> × 38   — Mahal-cluster CLR − genome CLR
        delta_clr_rp_<codon> × 38      — RP CLR − genome CLR
        aitchison_genome_to_mahal      — strength of selection (data-driven anchor)
        aitchison_genome_to_rp         — strength of selection (RP anchor)
        aitchison_rp_to_mahal          — agreement between the two anchors

      Ecology / summary block (also comparable across genomes):
        n_genes_total, n_genes_with_metrics
        n_optimized, frac_in_optimized_set, mean_membership_score, median_membership_score
        median_cai, median_melp, median_fop, median_rp_cai, median_rp_melp, median_rp_fop
        median_enc, enc_iqr, median_encprime, median_milc
        median_gc3, gc3_iqr, mean_gene_length
        grodon2_doubling_time_h, grodon2_lower_ci, grodon2_upper_ci
        hgt_candidate_frac, gc3_outlier_frac, hgt_combined_frac
        n_phage_genes, n_mobile_genes
        strand_asymmetry_significant_count
        n_trna_genes
    """
    rscu_cols_full = [c for c in RSCU_COLUMN_NAMES if (
        rscu_gene_df is None or c in rscu_gene_df.columns
    )]
    rscu_cols = _drop_redundant_codon_per_family(rscu_cols_full)

    def _vec(d: dict | None) -> np.ndarray | None:
        if d is None:
            return None
        v = np.array([d.get(c, np.nan) for c in rscu_cols])
        if np.all(np.isnan(v)):
            return None
        return v

    g_vec = _vec(rscu_genome)
    rp_vec = _vec(rscu_rp)
    mahal_vec = _vec(rscu_mahal_cluster)

    # Geometry block — fail soft when references are missing.
    geom: dict[str, float] = {}
    delta_mahal_cols = [f"delta_clr_mahal_{c}" for c in rscu_cols]
    delta_rp_cols = [f"delta_clr_rp_{c}" for c in rscu_cols]

    if g_vec is not None:
        clr_g = clr_transform(g_vec)
        if mahal_vec is not None:
            clr_m = clr_transform(mahal_vec)
            geom["aitchison_genome_to_mahal"] = float(np.linalg.norm(clr_m - clr_g))
            for col, v in zip(delta_mahal_cols, clr_m - clr_g):
                geom[col] = float(v)
        else:
            geom["aitchison_genome_to_mahal"] = float("nan")
            for col in delta_mahal_cols:
                geom[col] = float("nan")
        if rp_vec is not None:
            clr_r = clr_transform(rp_vec)
            geom["aitchison_genome_to_rp"] = float(np.linalg.norm(clr_r - clr_g))
            for col, v in zip(delta_rp_cols, clr_r - clr_g):
                geom[col] = float(v)
        else:
            geom["aitchison_genome_to_rp"] = float("nan")
            for col in delta_rp_cols:
                geom[col] = float("nan")
    else:
        for c in (delta_mahal_cols + delta_rp_cols
                  + ["aitchison_genome_to_mahal", "aitchison_genome_to_rp"]):
            geom[c] = float("nan")

    if rp_vec is not None and mahal_vec is not None:
        geom["aitchison_rp_to_mahal"] = aitchison_distance(rp_vec, mahal_vec)
    else:
        geom["aitchison_rp_to_mahal"] = float("nan")

    # Ecology / summary block
    eco: dict[str, float] = {}
    if rscu_gene_df is not None and not rscu_gene_df.empty:
        eco["n_genes_total"] = int(len(rscu_gene_df))
        eco["mean_gene_length"] = float(rscu_gene_df["length"].mean()) if "length" in rscu_gene_df.columns else float("nan")
    else:
        eco["n_genes_total"] = 0
        eco["mean_gene_length"] = float("nan")

    if mahal_cluster_df is not None and not mahal_cluster_df.empty:
        if "in_optimized_set" in mahal_cluster_df.columns:
            in_opt = mahal_cluster_df["in_optimized_set"].fillna(False).astype(bool)
            eco["n_optimized"] = int(in_opt.sum())
            eco["frac_in_optimized_set"] = float(in_opt.mean())
        if "membership_score" in mahal_cluster_df.columns:
            eco["mean_membership_score"] = float(mahal_cluster_df["membership_score"].mean(skipna=True))
            eco["median_membership_score"] = float(mahal_cluster_df["membership_score"].median(skipna=True))

    if expr_df is not None and not expr_df.empty:
        for c, key in (("CAI", "median_cai"), ("MELP", "median_melp"), ("Fop", "median_fop"),
                       ("rp_CAI", "median_rp_cai"), ("rp_MELP", "median_rp_melp"), ("rp_Fop", "median_rp_fop")):
            if c in expr_df.columns:
                eco[key] = float(expr_df[c].median(skipna=True))

    if enc_df is not None and not enc_df.empty:
        if "ENC" in enc_df.columns:
            eco["median_enc"] = float(enc_df["ENC"].median(skipna=True))
            eco["enc_iqr"] = float(enc_df["ENC"].quantile(0.75) - enc_df["ENC"].quantile(0.25))
        if "GC3" in enc_df.columns:
            eco["median_gc3"] = float(enc_df["GC3"].median(skipna=True))
            eco["gc3_iqr"] = float(enc_df["GC3"].quantile(0.75) - enc_df["GC3"].quantile(0.25))

    if encprime_df is not None and not encprime_df.empty and "ENCprime" in encprime_df.columns:
        eco["median_encprime"] = float(encprime_df["ENCprime"].median(skipna=True))
    if milc_df is not None and not milc_df.empty and "MILC" in milc_df.columns:
        eco["median_milc"] = float(milc_df["MILC"].median(skipna=True))

    # gRodon2 (already a per-genome scalar; loaded as a 1-row DataFrame typically)
    if grodon_df is not None and not grodon_df.empty:
        for c, key in (("d", "grodon2_doubling_time_h"),
                       ("lower_ci", "grodon2_lower_ci"),
                       ("upper_ci", "grodon2_upper_ci")):
            if c in grodon_df.columns:
                val = grodon_df[c].iloc[0]
                eco[key] = float(val) if pd.notna(val) else float("nan")

    # HGT and ecology fractions
    if hgt_df is not None and not hgt_df.empty:
        n = len(hgt_df)
        if n > 0:
            for c, key in (("hgt_flag_combined", "hgt_combined_frac"),
                           ("hgt_flag_fdr", "hgt_candidate_frac"),
                           ("gc3_outlier", "gc3_outlier_frac")):
                if c in hgt_df.columns:
                    eco[key] = float(hgt_df[c].fillna(False).astype(bool).mean())

    if phage_mobile_df is not None and not phage_mobile_df.empty:
        # Pipeline output uses 'is_phage' / 'is_mobile' boolean columns
        for c, key in (("is_phage", "n_phage_genes"), ("is_mobile", "n_mobile_genes")):
            if c in phage_mobile_df.columns:
                eco[key] = int(phage_mobile_df[c].fillna(False).astype(bool).sum())

    if strand_asymmetry_df is not None and not strand_asymmetry_df.empty:
        if "significant" in strand_asymmetry_df.columns:
            eco["strand_asymmetry_significant_count"] = int(
                strand_asymmetry_df["significant"].fillna(False).astype(bool).sum()
            )

    if trna_df is not None and not trna_df.empty:
        # trna_df is the long-format tRNA table (anticodon × decoded codon),
        # so unique anticodons gives the actual number of tRNA genes
        # (independent of wobble decoding rows).
        if "anticodon" in trna_df.columns and "tRNA_copy_number" in trna_df.columns:
            per_ac = trna_df.drop_duplicates(subset=["anticodon"])[["anticodon", "tRNA_copy_number"]]
            eco["n_trna_genes"] = int(per_ac["tRNA_copy_number"].sum())

    row = {"sample_id": sample_id}
    row.update(geom)
    row.update(eco)
    return pd.DataFrame([row])


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample loader → both signatures in one call
# ──────────────────────────────────────────────────────────────────────────────


def _load_gene_annotations(sample_dir: Path, sample_id: str) -> pd.DataFrame | None:
    """Build a per-gene annotation DataFrame (KO + COG) from pipeline outputs.

    Tries (in order):
      1. expression/<sid>_expression_annotated.tsv (has KO + COG already merged)
      2. kofamscan/<sid>_kofam_parsed.tsv + cogclassifier/cog_classify.tsv
         merged separately

    Returns a DataFrame keyed by 'gene' with whatever annotation columns
    were available, or None if nothing was found. Handles both the
    role-based pipeline layout and the legacy flat layout via
    _resolve_path.
    """
    sample_dir = Path(sample_dir)

    pieces: list[pd.DataFrame] = []

    # 1. Pre-merged expression_annotated TSV (often has KO + KO_definition,
    #    sometimes COG too depending on pipeline version). Use what's there.
    expr_ann_path = _resolve_path(
        sample_dir, "scores", f"{sample_id}_expression_annotated.tsv"
    )
    if expr_ann_path is None:
        expr_ann_path = _resolve_path(
            sample_dir, "expression", f"{sample_id}_expression_annotated.tsv"
        )
    if expr_ann_path is not None:
        try:
            df = pd.read_csv(expr_ann_path, sep="\t")
            keep = [c for c in (
                COL_GENE, "KO", "KO_definition", "COG_ID", "COG_letter",
                "COG_category", "COG_NAME", "product",
            ) if c in df.columns]
            if COL_GENE in keep and len(keep) > 1:
                pieces.append(df[keep].drop_duplicates(subset=[COL_GENE]))
        except Exception as e:
            logger.warning("Failed to read %s: %s", expr_ann_path, e)

    # 2. Raw KofamScan output — merge if expression_annotated didn't already
    #    cover KO (it normally does, but be defensive).
    have_ko = pieces and "KO" in pieces[0].columns
    if not have_ko:
        kofam_path = _resolve_path(
            sample_dir, "kofamscan", f"{sample_id}_kofam_parsed.tsv"
        )
        if kofam_path is not None:
            try:
                kf = pd.read_csv(kofam_path, sep="\t")
                if "gene_name" in kf.columns:
                    kf = kf.rename(columns={"gene_name": COL_GENE})
                keep = [COL_GENE] + [c for c in ("KO", "KO_definition") if c in kf.columns]
                if COL_GENE in kf.columns and len(keep) > 1:
                    pieces.append(kf[keep].drop_duplicates(subset=[COL_GENE]))
            except Exception as e:
                logger.warning("Failed to read KofamScan output %s: %s", kofam_path, e)

    # 3. COGclassifier output — always merge if expression_annotated didn't
    #    already include COG_ID (production expression_annotated.tsv usually
    #    has KO but not COG, so this branch is the typical path).
    have_cog = any("COG_ID" in p.columns for p in pieces)
    if not have_cog:
        cog_path = _resolve_path(sample_dir, "cogclassifier", "cog_classify.tsv")
        if cog_path is not None:
            try:
                cg = pd.read_csv(cog_path, sep="\t")
                if "QUERY_ID" in cg.columns:
                    cg = cg.rename(columns={"QUERY_ID": COL_GENE})
                if "COG_LETTER" in cg.columns:
                    cg = cg.rename(columns={"COG_LETTER": "COG_letter"})
                if "COG_DESCRIPTION" in cg.columns:
                    cg = cg.rename(columns={"COG_DESCRIPTION": "COG_category"})
                keep = [COL_GENE] + [c for c in (
                    "COG_ID", "COG_letter", "COG_category", "COG_NAME"
                ) if c in cg.columns]
                if COL_GENE in cg.columns and len(keep) > 1:
                    pieces.append(cg[keep].drop_duplicates(subset=[COL_GENE]))
            except Exception as e:
                logger.warning("Failed to read COGclassifier output %s: %s", cog_path, e)

    if not pieces:
        return None

    # Outer-merge so genes annotated by only one tool still get a row
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=COL_GENE, how="outer")
    return out


def write_signatures_for_sample(
    sample_dir: Path,
    sample_id: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Read pipeline outputs for one sample, write gene + genome signatures.

    Returns dict of paths: 'gene_signature' and 'genome_signature'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_sample_outputs(sample_dir, sample_id)
    # Pull a few extra ecology files that load_sample_outputs doesn't read
    # (because the gene_set module doesn't need them). Use the same path
    # resolver so both layouts (role-based + flat) work transparently.
    sample_dir = Path(sample_dir)
    grodon_path = _resolve_path(sample_dir, "bio_ecology", f"{sample_id}_grodon2_prediction.tsv")
    grodon_df = pd.read_csv(grodon_path, sep="\t") if grodon_path is not None else None
    phage_path = _resolve_path(sample_dir, "bio_ecology", f"{sample_id}_phage_mobile_elements.tsv")
    phage_df = pd.read_csv(phage_path, sep="\t") if phage_path is not None else None
    strand_path = _resolve_path(sample_dir, "bio_ecology", f"{sample_id}_strand_asymmetry.tsv")
    strand_df = pd.read_csv(strand_path, sep="\t") if strand_path is not None else None
    trna_path = _resolve_path(sample_dir, "advanced", f"{sample_id}_trna_counts.tsv")
    trna_df = pd.read_csv(trna_path, sep="\t") if trna_path is not None else None

    annotation_df = _load_gene_annotations(sample_dir, sample_id)
    gene_sig = compute_gene_signature(
        rscu_gene_df=loaded["rscu_gene_df"],
        rscu_genome=loaded["rscu_genome"],
        sample_id=sample_id,
        enc_df=loaded["enc_df"],
        expr_df=loaded["expr_df"],
        encprime_df=loaded["encprime_df"],
        milc_df=loaded["milc_df"],
        hgt_df=loaded["hgt_df"],
        mahal_cluster_df=loaded["mahal_cluster_df"],
        cbi_rp_df=loaded["cbi_rp_df"],
        cbi_mahal_df=loaded["cbi_mahal_df"],
        annotation_df=annotation_df,
    )
    genome_sig = compute_genome_signature(
        sample_id=sample_id,
        rscu_genome=loaded["rscu_genome"],
        rscu_rp=loaded["rscu_rp"],
        rscu_mahal_cluster=loaded["rscu_mahal_cluster"],
        rscu_gene_df=loaded["rscu_gene_df"],
        enc_df=loaded["enc_df"],
        expr_df=loaded["expr_df"],
        encprime_df=loaded["encprime_df"],
        milc_df=loaded["milc_df"],
        mahal_cluster_df=loaded["mahal_cluster_df"],
        grodon_df=grodon_df,
        hgt_df=loaded["hgt_df"],
        phage_mobile_df=phage_df,
        strand_asymmetry_df=strand_df,
        trna_df=trna_df,
    )

    gene_path = output_dir / f"{sample_id}_gene_signature.tsv"
    genome_path = output_dir / f"{sample_id}_genome_signature.tsv"
    gene_sig.to_csv(gene_path, sep="\t", index=False)
    genome_sig.to_csv(genome_path, sep="\t", index=False)
    logger.info(
        "Wrote signatures for %s: %d genes, %d genome features",
        sample_id, len(gene_sig), len(genome_sig.columns) - 1,
    )
    return {"gene_signature": gene_path, "genome_signature": genome_path}


# ──────────────────────────────────────────────────────────────────────────────
# Corpus aggregator
# ──────────────────────────────────────────────────────────────────────────────


def discover_signatures(input_dirs: list[Path]) -> tuple[list[Path], list[Path]]:
    """Find all *_genome_signature.tsv and *_gene_signature.tsv under input dirs.

    Each input dir can be either:
      - a per-sample directory containing the signature files at top level,
      - a directory of per-sample directories (search one level deep),
      - a direct path to a signature TSV (passed straight through).

    Returns (genome_paths, gene_paths) lists, deduplicated and sorted.
    """
    genome_files: set[Path] = set()
    gene_files: set[Path] = set()
    for d in input_dirs:
        d = Path(d)
        if d.is_file() and d.name.endswith("_genome_signature.tsv"):
            genome_files.add(d)
            gene_path = d.parent / d.name.replace("_genome_signature.tsv", "_gene_signature.tsv")
            if gene_path.exists():
                gene_files.add(gene_path)
            continue
        if d.is_file() and d.name.endswith("_gene_signature.tsv"):
            gene_files.add(d)
            continue
        if d.is_dir():
            # Top-level + one subdir level
            for pattern in ("*_genome_signature.tsv", "*/*_genome_signature.tsv"):
                for p in d.glob(pattern):
                    genome_files.add(p)
            for pattern in ("*_gene_signature.tsv", "*/*_gene_signature.tsv"):
                for p in d.glob(pattern):
                    gene_files.add(p)
    return sorted(genome_files), sorted(gene_files)


# Geometry features for clustering on selection structure (excludes ecology
# scalars). Pattern matched at runtime since codon names depend on
# RSCU_COLUMN_NAMES; here we list the prefix patterns.
_GEOMETRY_PREFIXES = ("delta_clr_mahal_", "delta_clr_rp_")
_GEOMETRY_SCALARS = (
    "aitchison_genome_to_mahal",
    "aitchison_genome_to_rp",
    "aitchison_rp_to_mahal",
)


def _geometry_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith(_GEOMETRY_PREFIXES)]
    cols += [c for c in _GEOMETRY_SCALARS if c in df.columns]
    return cols


def _all_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c != "sample_id" and pd.api.types.is_numeric_dtype(df[c])
    ]


def _robust_zscore(mat: np.ndarray) -> np.ndarray:
    """Median/MAD standardization. Returns z-scored matrix, ignoring all-NaN columns."""
    out = mat.astype(float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) < 2:
            out[:, j] = 0.0
            continue
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        if mad < 1e-12:
            out[:, j] = 0.0
        else:
            # 1.4826 makes MAD ≈ σ for normal data
            out[:, j] = (col - med) / (1.4826 * mad)
    # NaN → 0 after standardization (treats missing as median value)
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def cluster_corpus(
    corpus_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    use_umap: bool = False,
    n_components: int = 2,
    pca_components: int = 30,
    hdbscan_min_cluster_size: int | None = None,
    rng_seed: int = 42,
) -> dict:
    """Standardize → PCA (→ optional UMAP) → HDBSCAN/KMeans on corpus_df[feature_cols].

    Returns a dict with:
      embedding: 2D array, shape (n_samples, n_components)
      cluster: 1D array of cluster labels (-1 = noise for HDBSCAN)
      pca_explained: variance fraction per PC (length pca_components)
      method_dim: 'umap' or 'pca'
      method_cluster: 'hdbscan' or 'kmeans'
    """
    if not feature_cols:
        raise ValueError("No feature columns selected")
    n = len(corpus_df)
    if n < 4:
        raise ValueError(f"Need at least 4 samples to cluster, got {n}")

    X = corpus_df[feature_cols].values
    Xz = _robust_zscore(X)

    # PCA pre-step (caps dimensionality before UMAP / clustering)
    from sklearn.decomposition import PCA
    n_pcs = min(pca_components, Xz.shape[0] - 1, Xz.shape[1])
    pca = PCA(n_components=n_pcs, random_state=rng_seed)
    Xpca = pca.fit_transform(Xz)
    pca_explained = pca.explained_variance_ratio_

    method_dim = "pca"
    if use_umap:
        try:
            import umap
            n_neighbors = max(5, min(30, n - 1))
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                min_dist=0.1, metric="euclidean", random_state=rng_seed)
            embedding = reducer.fit_transform(Xpca)
            method_dim = "umap"
        except ImportError:
            logger.warning("umap-learn not available; falling back to PCA for the 2-D embedding")
            embedding = Xpca[:, :n_components]
    else:
        embedding = Xpca[:, :n_components]

    # HDBSCAN with KMeans fallback. The default min_cluster_size is a
    # heuristic — 2% of corpus size, floored at 5, capped at 50. The 2%
    # value has no published anchor; it was chosen to (1) avoid spurious
    # micro-clusters when the corpus is large and (2) avoid merging
    # genuinely distinct ecological groups when the corpus is small.
    # Override via ``--hdbscan-min-cluster-size`` for any publication-
    # grade clustering, and check robustness by sweeping a small grid
    # (e.g. 5, 10, 25, 50) and reporting cluster stability.
    if hdbscan_min_cluster_size is None:
        hdbscan_min_cluster_size = int(np.clip(round(n * 0.02), 5, 50))

    method_cluster = "hdbscan"
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, n_jobs=1)
        labels = clusterer.fit_predict(Xpca)
    except Exception as e:
        logger.warning("HDBSCAN unavailable (%s); falling back to KMeans (k=8)", e)
        from sklearn.cluster import KMeans
        k = max(2, min(8, n // 5))
        clusterer = KMeans(n_clusters=k, random_state=rng_seed, n_init="auto")
        labels = clusterer.fit_predict(Xpca)
        method_cluster = "kmeans"

    return {
        "embedding": embedding,
        "cluster": labels,
        "pca_explained": pca_explained,
        "method_dim": method_dim,
        "method_cluster": method_cluster,
        "feature_cols": feature_cols,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-cluster driver analysis
# ──────────────────────────────────────────────────────────────────────────────


def _cliffs_delta_corpus(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    # Vectorised — n is bounded by corpus size (typically <10k) which is fine
    diff = x[:, None] - y[None, :]
    return float(((diff > 0).sum() - (diff < 0).sum()) / (nx * ny))


def compute_corpus_cluster_drivers(
    corpus_df: pd.DataFrame,
    feature_cols: list[str],
    cluster_labels: np.ndarray,
    *,
    skip_noise: bool = True,
) -> pd.DataFrame:
    """Per-cluster differential feature analysis: Mann-Whitney U + Cliff's delta.

    For each (cluster, feature) pair, tests whether cluster members differ
    from the rest of the corpus (excluding HDBSCAN noise points if
    *skip_noise* is True). Reports BH-corrected p-values globally across
    the full (cluster × feature) test panel.

    Args:
        corpus_df: Per-genome feature matrix (one row per genome).
        feature_cols: Numeric feature columns to test.
        cluster_labels: 1D array of cluster ids aligned to corpus_df rows
            (-1 = HDBSCAN noise).
        skip_noise: When True, noise (-1) points are excluded from both the
            in-cluster and out-of-cluster sides of every test, so noise
            doesn't dilute the comparison.

    Returns:
        DataFrame with: cluster_id, feature, n_in, n_out, median_in,
        median_out, U_statistic, p_value, cliffs_delta, abs_effect,
        p_adjusted, significant.
        Sorted by (cluster_id ascending, abs_effect descending) so the
        top drivers per cluster appear first.
    """
    from scipy import stats as sp_stats

    if len(corpus_df) != len(cluster_labels):
        raise ValueError("cluster_labels length doesn't match corpus_df")
    if not feature_cols:
        return pd.DataFrame()

    arr_labels = np.asarray(cluster_labels)
    valid_idx = arr_labels != -1 if skip_noise else np.ones(len(arr_labels), dtype=bool)
    n_valid_clusters = len(set(arr_labels[valid_idx])) if valid_idx.any() else 0
    if n_valid_clusters < 2:
        return pd.DataFrame()

    rows = []
    sub_df = corpus_df.iloc[valid_idx].reset_index(drop=True) if skip_noise else corpus_df
    sub_labels = arr_labels[valid_idx] if skip_noise else arr_labels

    for cluster_id in sorted(set(sub_labels)):
        in_mask = sub_labels == cluster_id
        if in_mask.sum() < 3 or (~in_mask).sum() < 3:
            continue
        for feature in feature_cols:
            if feature not in sub_df.columns:
                continue
            x = sub_df.loc[in_mask, feature].dropna().values
            y = sub_df.loc[~in_mask, feature].dropna().values
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
            except ValueError:
                continue
            d = _cliffs_delta_corpus(x, y)
            rows.append({
                "cluster_id": int(cluster_id),
                "feature": feature,
                "n_in": int(len(x)),
                "n_out": int(len(y)),
                "median_in": float(np.median(x)),
                "median_out": float(np.median(y)),
                "U_statistic": float(u),
                "p_value": float(p),
                "cliffs_delta": float(d),
                "abs_effect": float(abs(d)),
            })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["p_adjusted"] = benjamini_hochberg(out["p_value"].values)
    out["significant"] = out["p_adjusted"] < 0.05
    return out.sort_values(
        ["cluster_id", "abs_effect"], ascending=[True, False],
    ).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Gene-level cross-genome clustering
# ──────────────────────────────────────────────────────────────────────────────


def _gene_feature_columns(gene_df: pd.DataFrame) -> list[str]:
    """The CLR-Δ codon vector — what we cluster genes on across genomes."""
    return [c for c in gene_df.columns if c.startswith("delta_clr_")]


def cluster_corpus_genes(
    gene_corpus_df: pd.DataFrame,
    *,
    use_umap: bool = True,
    n_components: int = 2,
    pca_components: int = 20,
    hdbscan_min_cluster_size: int | None = None,
    rng_seed: int = 42,
    max_genes: int = 1_000_000,
) -> dict:
    """Cluster genes across the entire corpus by their CLR-Δ codon vector.

    Standardize → PCA → (UMAP if installed, else PCA) → HDBSCAN with KMeans
    fallback. Operates on the per-gene `delta_clr_*` vectors which are
    cross-genome-comparable by construction (CLR removes per-AA-family sum
    constraints; the gene-minus-genome-mean step removes mutational
    background).

    For corpora above *max_genes*, raises ValueError. Above ~500k genes the
    PCA + HDBSCAN chain gets slow (minutes); above ~1M memory becomes the
    bottleneck.

    Returns dict with:
        embedding         (n_genes, 2)
        cluster           1D array of cluster labels (-1 = HDBSCAN noise)
        feature_cols      columns used as features
        method_dim        'umap' or 'pca'
        method_cluster    'hdbscan' or 'kmeans'
        pca_explained     variance fraction per PC
    """
    n = len(gene_corpus_df)
    if n > max_genes:
        raise ValueError(
            f"Gene-level corpus has {n:,} genes (cap = {max_genes:,}). "
            f"Pass max_genes=N to override, or subsample upstream."
        )
    if n < 4:
        raise ValueError(f"Need at least 4 genes to cluster, got {n}")

    feature_cols = _gene_feature_columns(gene_corpus_df)
    if len(feature_cols) < 5:
        raise ValueError(
            f"Need at least 5 delta_clr_* feature columns; found {len(feature_cols)}. "
            f"Did the gene_signature.tsv files include CLR-Δ vectors?"
        )

    X = gene_corpus_df[feature_cols].values
    Xz = _robust_zscore(X)

    from sklearn.decomposition import PCA
    n_pcs = min(pca_components, Xz.shape[0] - 1, Xz.shape[1])
    pca = PCA(n_components=n_pcs, random_state=rng_seed)
    Xpca = pca.fit_transform(Xz)
    pca_explained = pca.explained_variance_ratio_

    method_dim = "pca"
    if use_umap:
        try:
            import umap
            n_neighbors = max(15, min(50, n // 100 if n > 1000 else 15))
            reducer = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors,
                min_dist=0.1, metric="euclidean", random_state=rng_seed,
                low_memory=True,
            )
            embedding = reducer.fit_transform(Xpca)
            method_dim = "umap"
        except ImportError:
            logger.warning("umap-learn not available; using first 2 PCs as embedding")
            embedding = Xpca[:, :n_components]
    else:
        embedding = Xpca[:, :n_components]

    if hdbscan_min_cluster_size is None:
        # Heuristic: 0.1% of corpus, floor 10, cap 200
        hdbscan_min_cluster_size = int(np.clip(round(n * 0.001), 10, 200))

    method_cluster = "hdbscan"
    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size, n_jobs=1,
        )
        labels = clusterer.fit_predict(Xpca)
        # If HDBSCAN flagged everything as noise (uniformly-distributed clusters
        # don't have density gradients it can latch onto), fall back to KMeans.
        if (labels == -1).all():
            logger.info(
                "HDBSCAN flagged all %d genes as noise; falling back to KMeans for gene-level clustering",
                n,
            )
            raise RuntimeError("hdbscan_all_noise")
    except Exception as e:
        if "hdbscan_all_noise" not in str(e):
            logger.warning("HDBSCAN failed (%s); falling back to KMeans (k=10)", e)
        from sklearn.cluster import KMeans
        k = max(2, min(10, n // 50))
        clusterer = KMeans(n_clusters=k, random_state=rng_seed, n_init="auto")
        labels = clusterer.fit_predict(Xpca)
        method_cluster = "kmeans"

    return {
        "embedding": embedding,
        "cluster": labels,
        "feature_cols": feature_cols,
        "method_dim": method_dim,
        "method_cluster": method_cluster,
        "pca_explained": pca_explained,
    }


def cluster_corpus_genes_by_category(
    gene_corpus_df: pd.DataFrame,
    category_col: str,
    *,
    min_category_size: int = 10,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Within-category gene clustering — does each KO/COG class split?

    For each category in *category_col* with at least *min_category_size*
    genes, clusters the within-category gene subset on its CLR-Δ codon
    vectors. Uses HDBSCAN (or KMeans fallback) per category. Categories
    with fewer than min_category_size genes are skipped.

    Returns a long-format DataFrame:
        category, sample_id, gene, sub_cluster, n_in_category,
        embed_dim1, embed_dim2

    Plus a sentinel summary row per category with sub_cluster = "_summary"
    (unused — kept simple; one row per gene only).

    A separate summary DataFrame can be built downstream from the
    sub_cluster value_counts.
    """
    if category_col not in gene_corpus_df.columns:
        logger.warning("Category column '%s' not in gene corpus; skipping per-category clustering", category_col)
        return pd.DataFrame()

    feature_cols = _gene_feature_columns(gene_corpus_df)
    if len(feature_cols) < 5:
        logger.warning("Too few delta_clr columns for per-category gene clustering")
        return pd.DataFrame()

    rows = []
    cat_counts = gene_corpus_df[category_col].value_counts(dropna=True)
    eligible = cat_counts[cat_counts >= min_category_size].index.tolist()
    if not eligible:
        logger.warning(
            "No %s categories with >= %d genes; skipping per-category clustering",
            category_col, min_category_size,
        )
        return pd.DataFrame()
    logger.info(
        "Per-category clustering on %s: %d categories with >= %d genes "
        "(out of %d total non-null categories)",
        category_col, len(eligible), min_category_size, len(cat_counts),
    )

    from sklearn.decomposition import PCA
    from sklearn.cluster import HDBSCAN, KMeans

    for cat in eligible:
        sub = gene_corpus_df[gene_corpus_df[category_col] == cat].copy()
        n = len(sub)
        X = sub[feature_cols].values
        Xz = _robust_zscore(X)
        # PCA cap before HDBSCAN; helps for tiny categories where the raw
        # 38-d space is sparse.
        n_pcs = min(10, n - 1, Xz.shape[1])
        pca = PCA(n_components=n_pcs, random_state=rng_seed)
        Xpca = pca.fit_transform(Xz)

        # 2D embedding for the table — PCA only at this scale (UMAP per
        # category would explode in cost across hundreds of categories)
        embed = Xpca[:, :2] if Xpca.shape[1] >= 2 else \
            np.column_stack([Xpca[:, 0], np.zeros(n)])

        # HDBSCAN with a min_cluster_size scaled to the category size
        min_cs = max(3, min(int(np.sqrt(n)), 20))
        try:
            clusterer = HDBSCAN(min_cluster_size=min_cs, n_jobs=1)
            labels = clusterer.fit_predict(Xpca)
        except Exception:
            k = max(2, min(5, n // 5))
            clusterer = KMeans(n_clusters=k, random_state=rng_seed, n_init="auto")
            labels = clusterer.fit_predict(Xpca)

        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append({
                "category_col": category_col,
                "category": cat,
                "n_in_category": n,
                "sample_id": r.get("sample_id", ""),
                COL_GENE: r.get(COL_GENE, ""),
                "sub_cluster": int(labels[i]),
                "embed_dim1": float(embed[i, 0]),
                "embed_dim2": float(embed[i, 1]) if embed.shape[1] > 1 else 0.0,
            })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Validation: Mantel test
# ──────────────────────────────────────────────────────────────────────────────


def mantel_test(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    n_perm: int = 999,
    rng_seed: int = 42,
) -> dict:
    """Mantel test: Pearson r between two distance matrices, with permutation p.

    Both inputs must be square symmetric matrices of the same size, with
    sample order aligned (row i of dist_a corresponds to row i of dist_b).

    Returns dict with r, p_value, n_perm.
    """
    if dist_a.shape != dist_b.shape or dist_a.shape[0] != dist_a.shape[1]:
        raise ValueError("Mantel inputs must be matched square matrices")
    n = dist_a.shape[0]
    if n < 4:
        return {"r": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "note": "n<4, Mantel undefined"}

    # Upper-triangle indices (excluding diagonal)
    iu = np.triu_indices(n, k=1)
    a = dist_a[iu]
    b = dist_b[iu]
    if np.std(a) == 0 or np.std(b) == 0:
        return {"r": float("nan"), "p_value": float("nan"), "n_perm": 0,
                "note": "zero variance in distance vector"}
    r_obs = float(np.corrcoef(a, b)[0, 1])

    rng = np.random.default_rng(rng_seed)
    n_geq = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        b_perm = dist_b[np.ix_(perm, perm)][iu]
        r_perm = np.corrcoef(a, b_perm)[0, 1]
        if abs(r_perm) >= abs(r_obs):
            n_geq += 1
    p_value = (n_geq + 1) / (n_perm + 1)
    return {"r": r_obs, "p_value": float(p_value), "n_perm": n_perm}


def read_phylogeny_distance_matrix(path: Path, sample_ids: list[str]) -> np.ndarray | None:
    """Read a precomputed pairwise distance matrix or parse a Newick file.

    Returns a square distance matrix aligned to *sample_ids* order, or None
    if the file can't be loaded / aligned.

    Supported formats:
      - .tsv with first column = sample_id, headers = sample_ids
      - .nwk / .newick / .tree: parsed via Bio.Phylo (if biopython available)
    """
    path = Path(path)
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix in (".nwk", ".newick", ".tree"):
        try:
            from io import StringIO
            from Bio import Phylo
            with open(path) as fh:
                tree = Phylo.read(fh, "newick")
            terminals = {leaf.name: leaf for leaf in tree.get_terminals()}
            n = len(sample_ids)
            mat = np.full((n, n), np.nan)
            for i, a in enumerate(sample_ids):
                for j, b in enumerate(sample_ids):
                    if a in terminals and b in terminals:
                        mat[i, j] = float(tree.distance(terminals[a], terminals[b]))
            return mat
        except ImportError:
            logger.warning("biopython not available; cannot parse Newick %s", path)
            return None
        except Exception as e:
            logger.warning("Failed to parse Newick %s: %s", path, e)
            return None

    try:
        df = pd.read_csv(path, sep="\t", index_col=0)
    except Exception as e:
        logger.warning("Failed to read distance matrix %s: %s", path, e)
        return None

    missing = [s for s in sample_ids if s not in df.index or s not in df.columns]
    if missing:
        logger.warning(
            "Phylogeny matrix is missing %d/%d sample_ids (e.g. %s); skipping Mantel test",
            len(missing), len(sample_ids), missing[:5],
        )
        return None
    return df.loc[sample_ids, sample_ids].values.astype(float)


# ──────────────────────────────────────────────────────────────────────────────
# Corpus orchestrator
# ──────────────────────────────────────────────────────────────────────────────


def build_corpus(
    input_dirs: list[Path],
    output_dir: Path,
    *,
    features: str = "geometry",
    use_umap: bool = False,
    include_gene_level: bool = False,
    gene_level_clustering: bool = False,
    gene_level_by_category: str | None = None,
    gene_level_min_category_size: int = 10,
    gene_level_max_genes: int = 1_000_000,
    phylogeny_path: Path | None = None,
    metadata_path: Path | None = None,
    hdbscan_min_cluster_size: int | None = None,
    rng_seed: int = 42,
    focus_genomes: list[str] | None = None,
) -> dict[str, Path]:
    """Concatenate per-genome signatures, cluster, and validate.

    Args:
        input_dirs: Paths to per-sample directories or signature files.
        output_dir: Where to write the corpus outputs.
        features: 'geometry' (CLR-Δ + Aitchison only) or 'all' (full vector).
        use_umap: If True and umap-learn installed, do UMAP after PCA.
        include_gene_level: Also concatenate gene_signature files (slow on large corpora).
        phylogeny_path: Optional Newick or precomputed distance-matrix TSV.
        metadata_path: Optional TSV with sample_id + annotation columns; merged
            into corpus_genome_clusters.tsv for downstream colouring.
        hdbscan_min_cluster_size: Override the heuristic min_cluster_size.
        rng_seed: Random seed.

    Returns dict of output paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genome_paths, gene_paths = discover_signatures(input_dirs)
    if not genome_paths:
        raise FileNotFoundError(
            f"No *_genome_signature.tsv files found under {input_dirs}. "
            "Run `codonpipe signatures` on each sample first."
        )
    logger.info("Discovered %d genome signature files", len(genome_paths))

    corpus = pd.concat(
        [pd.read_csv(p, sep="\t") for p in genome_paths],
        ignore_index=True, sort=False,
    )
    if "sample_id" not in corpus.columns:
        raise ValueError("Genome-signature concatenation lacks 'sample_id'")
    corpus = corpus.drop_duplicates(subset=["sample_id"], keep="first")
    logger.info("Corpus: %d unique genomes, %d feature columns", len(corpus), len(corpus.columns) - 1)

    if features == "geometry":
        feature_cols = _geometry_columns(corpus)
    elif features == "all":
        feature_cols = _all_feature_columns(corpus)
    else:
        raise ValueError(f"features={features!r} not in ('geometry', 'all')")
    if not feature_cols:
        raise ValueError("No usable feature columns after selection")

    out: dict[str, Path] = {}

    # Persist the full corpus matrix (with features only)
    corpus_path = output_dir / "corpus_genome_signature.tsv"
    corpus.to_csv(corpus_path, sep="\t", index=False)
    out["corpus_genome_signature"] = corpus_path

    # Cluster
    cluster_result = cluster_corpus(
        corpus, feature_cols,
        use_umap=use_umap,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        rng_seed=rng_seed,
    )
    embed = cluster_result["embedding"]
    labels = cluster_result["cluster"]
    cluster_df = pd.DataFrame({
        "sample_id": corpus["sample_id"].values,
        "cluster": labels,
        "embed_dim1": embed[:, 0],
        "embed_dim2": embed[:, 1] if embed.shape[1] > 1 else np.zeros(len(embed)),
    })

    # Optional metadata join
    if metadata_path is not None and Path(metadata_path).exists():
        meta = pd.read_csv(metadata_path, sep="\t")
        if "sample_id" in meta.columns:
            cluster_df = cluster_df.merge(meta, on="sample_id", how="left")
        else:
            logger.warning("Metadata file lacks 'sample_id' column; ignoring")

    cluster_path = output_dir / "corpus_genome_clusters.tsv"
    cluster_df.to_csv(cluster_path, sep="\t", index=False)
    out["corpus_genome_clusters"] = cluster_path

    # ── Validation: Mantel test against phylogeny ─────────────────────────
    val_rows = []
    if phylogeny_path is not None:
        sample_ids = list(corpus["sample_id"].values)
        phylo_dist = read_phylogeny_distance_matrix(Path(phylogeny_path), sample_ids)
        if phylo_dist is not None:
            from scipy.spatial.distance import pdist, squareform
            X = corpus[feature_cols].values
            Xz = _robust_zscore(X)
            sig_dist = squareform(pdist(Xz, metric="euclidean"))
            mantel = mantel_test(sig_dist, phylo_dist, rng_seed=rng_seed)
            val_rows.append({
                "test": "mantel_signature_vs_phylogeny",
                "r": mantel["r"],
                "p_value": mantel["p_value"],
                "n_perm": mantel["n_perm"],
                "n_samples": len(sample_ids),
                "interpretation": (
                    "high |r| → cross-genome distance recovers phylogeny "
                    "(possibly via GC content); low |r| → distance reflects "
                    "non-phylogenetic structure (ecology, selection)"
                ),
            })

    # Cluster-vs-metadata associations (chi-squared on each metadata column)
    if metadata_path is not None and "cluster" in cluster_df.columns:
        from scipy.stats import chi2_contingency
        meta_cols = [c for c in cluster_df.columns
                     if c not in ("sample_id", "cluster", "embed_dim1", "embed_dim2")
                     and not pd.api.types.is_numeric_dtype(cluster_df[c])]
        for col in meta_cols:
            sub = cluster_df[["cluster", col]].dropna()
            if len(sub) < 10:
                continue
            ct = pd.crosstab(sub["cluster"], sub[col])
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            try:
                chi2, p, dof, _ = chi2_contingency(ct.values)
                # Cramer's V effect size
                n_chi = ct.values.sum()
                v = float(np.sqrt(chi2 / (n_chi * (min(ct.shape) - 1)))) if n_chi else float("nan")
                val_rows.append({
                    "test": f"cluster_vs_{col}",
                    "r": v,
                    "p_value": float(p),
                    "n_perm": 0,
                    "n_samples": int(n_chi),
                    "interpretation": "Cramer's V effect size; p from chi-squared test",
                })
            except Exception as e:
                logger.warning("chi-squared test failed for %s: %s", col, e)

    if val_rows:
        val_df = pd.DataFrame(val_rows)
        if any("p_value" in r and not np.isnan(r["p_value"]) for r in val_rows):
            val_df["p_adjusted"] = benjamini_hochberg(val_df["p_value"].values)
        val_path = output_dir / "corpus_validation.tsv"
        val_df.to_csv(val_path, sep="\t", index=False)
        out["corpus_validation"] = val_path

    # Gene-level corpus + optional clustering / per-category analyses.
    # Loaded if any of: --include-gene-level, --gene-level-clustering,
    # --gene-level-by-category was requested.
    need_gene_level = include_gene_level or gene_level_clustering or gene_level_by_category
    gene_corpus = None
    if need_gene_level and gene_paths:
        gene_corpus = pd.concat(
            [pd.read_csv(p, sep="\t") for p in gene_paths],
            ignore_index=True, sort=False,
        )
        gene_corpus_path = output_dir / "corpus_gene_signature.tsv"
        gene_corpus.to_csv(gene_corpus_path, sep="\t", index=False)
        out["corpus_gene_signature"] = gene_corpus_path
        logger.info("Wrote gene-level corpus: %d genes from %d genomes",
                    len(gene_corpus), gene_corpus["sample_id"].nunique())

    # Gene-level cross-genome clustering
    gene_cluster_result = None
    if gene_level_clustering and gene_corpus is not None:
        try:
            gene_cluster_result = cluster_corpus_genes(
                gene_corpus,
                use_umap=use_umap,
                hdbscan_min_cluster_size=None,  # heuristic per-corpus
                rng_seed=rng_seed,
                max_genes=gene_level_max_genes,
            )
            gc_df = pd.DataFrame({
                COL_GENE: gene_corpus[COL_GENE].values
                          if COL_GENE in gene_corpus.columns else range(len(gene_corpus)),
                "sample_id": gene_corpus["sample_id"].values,
                "cluster": gene_cluster_result["cluster"],
                "embed_dim1": gene_cluster_result["embedding"][:, 0],
                "embed_dim2": gene_cluster_result["embedding"][:, 1]
                              if gene_cluster_result["embedding"].shape[1] > 1
                              else np.zeros(len(gene_corpus)),
            })
            # Carry forward common annotation columns when present
            for c in ("KO", "KO_definition", "COG_ID", "COG_letter", "COG_category"):
                if c in gene_corpus.columns:
                    gc_df[c] = gene_corpus[c].values
            gc_path = output_dir / "corpus_gene_clusters.tsv"
            gc_df.to_csv(gc_path, sep="\t", index=False)
            out["corpus_gene_clusters"] = gc_path
            n_clusters_found = len(set(gene_cluster_result["cluster"])) - (
                1 if -1 in gene_cluster_result["cluster"] else 0
            )
            logger.info(
                "Gene-level clustering: %d clusters from %d genes (%s + %s)",
                n_clusters_found, len(gene_corpus),
                gene_cluster_result["method_dim"], gene_cluster_result["method_cluster"],
            )
        except ValueError as e:
            logger.warning("Gene-level clustering skipped: %s", e)

    # Per-functional-category gene clustering
    by_cat_df = None
    if gene_level_by_category and gene_corpus is not None:
        cat_col = gene_level_by_category
        # Accept friendly names
        if cat_col.upper() == "KO":
            cat_col = "KO"
        elif cat_col.upper() in ("COG", "COG_ID"):
            cat_col = "COG_ID"
        try:
            by_cat_df = cluster_corpus_genes_by_category(
                gene_corpus, cat_col,
                min_category_size=gene_level_min_category_size,
                rng_seed=rng_seed,
            )
            if not by_cat_df.empty:
                bc_path = output_dir / f"corpus_gene_clusters_by_{cat_col}.tsv"
                by_cat_df.to_csv(bc_path, sep="\t", index=False)
                out[f"corpus_gene_clusters_by_{cat_col}"] = bc_path
                # Build a per-category summary: how many sub-clusters did each get?
                summary_rows = []
                for cat, sub in by_cat_df.groupby("category"):
                    non_noise = [c for c in sub["sub_cluster"].unique() if c != -1]
                    summary_rows.append({
                        "category": cat,
                        "n_genes": int(sub["n_in_category"].iloc[0]),
                        "n_subclusters": int(len(non_noise)),
                        "n_noise_genes": int((sub["sub_cluster"] == -1).sum()),
                        "splits": int(len(non_noise) >= 2),
                    })
                summ_df = pd.DataFrame(summary_rows).sort_values(
                    ["splits", "n_subclusters", "n_genes"],
                    ascending=[False, False, False],
                )
                bs_path = output_dir / f"corpus_gene_clusters_by_{cat_col}_summary.tsv"
                summ_df.to_csv(bs_path, sep="\t", index=False)
                out[f"corpus_gene_clusters_by_{cat_col}_summary"] = bs_path
                logger.info(
                    "Per-%s clustering: %d categories tested, %d showed >=2 sub-clusters",
                    cat_col, len(summ_df), int(summ_df["splits"].sum()),
                )
        except Exception as e:
            logger.warning("Per-category gene clustering failed: %s", e, exc_info=True)

    # Cluster-driver analysis (per-cluster differential features)
    drivers_df = compute_corpus_cluster_drivers(
        corpus, feature_cols, cluster_result["cluster"], skip_noise=True,
    )
    if not drivers_df.empty:
        drv_path = output_dir / "corpus_cluster_drivers.tsv"
        drivers_df.to_csv(drv_path, sep="\t", index=False)
        out["corpus_cluster_drivers"] = drv_path
        n_sig = int(drivers_df["significant"].sum())
        logger.info(
            "Cluster drivers: %d / %d (cluster × feature) tests BH-significant; see %s",
            n_sig, len(drivers_df), drv_path,
        )

    # Render the five corpus figures
    try:
        from codonpipe.modules._corpus_figures import (
            render_multi_overlay_umap,
            render_cluster_signature_heatmap,
            render_cluster_drivers_forest,
            render_mantel_stratified,
            render_focus_genome_locator,
        )

        sample_ids = list(corpus["sample_id"].values)
        embed = cluster_result["embedding"]
        labels = cluster_result["cluster"]

        # Build the overlay frame (corpus features + any joined metadata)
        overlay_df = corpus.copy()
        if metadata_path is not None and Path(metadata_path).exists():
            meta = pd.read_csv(metadata_path, sep="\t")
            if "sample_id" in meta.columns:
                # Don't double-add columns that already exist in corpus
                new_cols = [c for c in meta.columns if c not in corpus.columns or c == "sample_id"]
                if len(new_cols) > 1:
                    overlay_df = overlay_df.merge(
                        meta[new_cols], on="sample_id", how="left",
                    )

        # 1. Multi-overlay UMAP grid
        png, svg = render_multi_overlay_umap(
            output_dir, embed, sample_ids, labels, overlay_df,
            method_dim=cluster_result["method_dim"].upper(),
            method_cluster=cluster_result["method_cluster"].upper(),
        )
        if png is not None:
            out["multi_overlay_png"] = png
            out["multi_overlay_svg"] = svg

        # 2. Cluster signature heatmap (only meaningful when geometry block has CLR-Δ)
        png, svg = render_cluster_signature_heatmap(
            output_dir, corpus, labels, skip_noise=True,
        )
        if png is not None:
            out["cluster_signature_png"] = png
            out["cluster_signature_svg"] = svg

        # 3. Per-cluster drivers forest plot
        if not drivers_df.empty:
            png, svg = render_cluster_drivers_forest(
                output_dir, drivers_df, n_per_cluster=8, significant_only=True,
            )
            if png is not None:
                out["cluster_drivers_png"] = png
                out["cluster_drivers_svg"] = svg

        # 4. Mantel-stratified scatter (when phylogeny was loaded)
        if phylogeny_path is not None and Path(phylogeny_path).exists():
            phylo_dist = read_phylogeny_distance_matrix(Path(phylogeny_path), sample_ids)
            if phylo_dist is not None:
                from scipy.spatial.distance import pdist as _pdist, squareform as _sq
                X = corpus[feature_cols].values
                Xz = _robust_zscore(X)
                sig_dist = _sq(_pdist(Xz, metric="euclidean"))
                # Pick a categorical metadata column for stratification
                strat_col = None
                if metadata_path is not None and Path(metadata_path).exists():
                    meta_full = pd.read_csv(metadata_path, sep="\t")
                    for candidate in ("phylum", "class", "order", "family", "habitat"):
                        if candidate in meta_full.columns:
                            strat_col = candidate
                            meta_for_mantel = meta_full
                            break
                else:
                    meta_for_mantel = None
                png, svg = render_mantel_stratified(
                    output_dir, sample_ids, sig_dist, phylo_dist,
                    metadata_df=meta_for_mantel if strat_col else None,
                    group_col=strat_col or "phylum",
                )
                if png is not None:
                    out["mantel_scatter_png"] = png
                    out["mantel_scatter_svg"] = svg

        # 5b. Gene-level UMAP figure
        if gene_cluster_result is not None and gene_corpus is not None:
            try:
                from codonpipe.modules._corpus_figures import render_gene_level_umap
                # Build host-metadata lookup for overlays
                host_meta = corpus[["sample_id"]].copy()
                for hcol in ("median_gc3", "grodon2_doubling_time_h",
                             "frac_in_optimized_set"):
                    if hcol in corpus.columns:
                        host_meta[hcol] = corpus[hcol].values
                if metadata_path is not None and Path(metadata_path).exists():
                    meta_full = pd.read_csv(metadata_path, sep="\t")
                    if "sample_id" in meta_full.columns:
                        for hcol in ("phylum", "class", "habitat"):
                            if hcol in meta_full.columns:
                                host_meta = host_meta.merge(
                                    meta_full[["sample_id", hcol]],
                                    on="sample_id", how="left",
                                )
                png, svg = render_gene_level_umap(
                    output_dir, gene_corpus, gene_cluster_result, host_meta,
                )
                if png is not None:
                    out["gene_level_umap_png"] = png
                    out["gene_level_umap_svg"] = svg
            except Exception as e:
                logger.warning("Gene-level UMAP figure failed: %s", e, exc_info=True)

        # 5c. Per-category diversity figure
        if by_cat_df is not None and not by_cat_df.empty:
            try:
                from codonpipe.modules._corpus_figures import render_category_diversity
                png, svg = render_category_diversity(
                    output_dir, by_cat_df,
                )
                if png is not None:
                    out["category_diversity_png"] = png
                    out["category_diversity_svg"] = svg
            except Exception as e:
                logger.warning("Category-diversity figure failed: %s", e, exc_info=True)

        # 6. Focus-genome locators
        if focus_genomes:
            for focus_id in focus_genomes:
                png, svg = render_focus_genome_locator(
                    output_dir, focus_id, embed, sample_ids, labels, corpus,
                    method_dim=cluster_result["method_dim"].upper(),
                )
                if png is not None:
                    safe = focus_id.replace("/", "_").replace(" ", "_")
                    out[f"focus_{safe}_png"] = png
                    out[f"focus_{safe}_svg"] = svg
    except Exception as e:
        logger.warning("Corpus figure rendering failed: %s", e, exc_info=True)

    logger.info("Corpus build complete. Outputs:")
    for k, v in out.items():
        logger.info("  %s: %s", k, v)
    return out


def _render_corpus_figure(corpus_df, cluster_df, cluster_result, output_dir) -> dict[str, Path]:
    """DEPRECATED. Kept for backward-compat; build_corpus now calls the
    five-figure renderers in _corpus_figures.py instead. This function
    still produces the original 3-panel summary figure if invoked directly.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # A: 2D embedding scatter, coloured by cluster
    embed = cluster_result["embedding"]
    labels = cluster_result["cluster"]
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = "#cccccc" if lab == -1 else cmap(i % 20)
        axA.scatter(embed[mask, 0], embed[mask, 1], s=20, alpha=0.7,
                    color=color, edgecolor="black", linewidth=0.2,
                    label=f"noise (n={mask.sum()})" if lab == -1 else f"c{lab} (n={mask.sum()})")
    axA.set_xlabel(f"{cluster_result['method_dim'].upper()} dim 1")
    axA.set_ylabel(f"{cluster_result['method_dim'].upper()} dim 2")
    axA.set_title(f"A. Corpus dim-reduction ({cluster_result['method_cluster']})")
    axA.legend(loc="best", fontsize=7, frameon=False)

    # B: scalar metrics per cluster (boxplots) — use ENC, GC3, doubling time
    scalar_cands = ["median_gc3", "median_enc", "grodon2_doubling_time_h",
                    "frac_in_optimized_set"]
    available = [c for c in scalar_cands if c in corpus_df.columns]
    if available and len(unique_labels) > 1:
        # Pick the first available, ranked by clinical relevance
        col = available[0]
        merged = corpus_df.merge(
            cluster_df[["sample_id", "cluster"]], on="sample_id", how="left",
        )
        groups = [merged.loc[merged["cluster"] == lab, col].dropna().values
                  for lab in unique_labels]
        positions = list(range(len(unique_labels)))
        bp = axB.boxplot(groups, positions=positions, showfliers=False, widths=0.6)
        axB.set_xticks(positions)
        axB.set_xticklabels([f"c{l}" if l != -1 else "noise" for l in unique_labels],
                            rotation=0, fontsize=8)
        axB.set_ylabel(col)
        axB.set_title(f"B. {col} by cluster")
    else:
        axB.text(0.5, 0.5, "Scalars unavailable", ha="center", va="center",
                 transform=axB.transAxes)

    # C: PCA scree
    pca_var = cluster_result.get("pca_explained")
    if pca_var is not None and len(pca_var) > 0:
        axC.bar(range(1, len(pca_var) + 1), pca_var, color="#1f77b4")
        axC.set_xlabel("PC")
        axC.set_ylabel("Var. fraction")
        axC.set_title("C. PCA scree (pre-clustering)")
        axC.set_xlim(0.5, len(pca_var) + 0.5)
    else:
        axC.text(0.5, 0.5, "PCA unavailable", ha="center", va="center",
                 transform=axC.transAxes)

    fig.suptitle(f"Cross-genome corpus  (n_genomes = {len(corpus_df)})",
                 fontsize=13, fontweight="bold")
    png_path = Path(output_dir) / "corpus_dimension_reduction.png"
    svg_path = Path(output_dir) / "corpus_dimension_reduction.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": png_path, "figure_svg": svg_path}
