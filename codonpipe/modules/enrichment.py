"""Pathway enrichment analysis for expression-classified gene sets.

Performs hypergeometric tests to identify KEGG pathways significantly enriched
among high-expression and low-expression genes, using KO accessions from
KofamScan as the annotation source.

KO-to-pathway mappings are fetched from the KEGG REST API and cached locally,
or loaded from a user-supplied file.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from codonpipe import __version__
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")

# KEGG REST endpoint for the KO-to-pathway link table
_KEGG_LINK_URL = "https://rest.kegg.jp/link/pathway/ko"
# KEGG REST endpoint for pathway names
_KEGG_PATHWAY_LIST_URL = "https://rest.kegg.jp/list/pathway/ko"


# ── KO-to-pathway mapping ───────────────────────────────────────────────────

def load_ko_pathway_map(
    user_file: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Load KO-to-pathway mapping.

    Priority:
        1. User-supplied TSV (two columns: KO<tab>pathway)
        2. Cached KEGG download
        3. Fresh KEGG REST API download (cached for reuse)

    Returns:
        Dict mapping KO accession (e.g. 'K00001') to a set of pathway IDs
        (e.g. {'ko00010', 'ko00071'}).
    """
    if user_file is not None:
        return _load_user_ko_map(user_file)

    if cache_dir is not None:
        cached = cache_dir / "kegg_ko_pathway.json"
        if cached.exists():
            logger.info("Loading cached KO-pathway map from %s", cached)
            with open(cached) as f:
                raw = json.load(f)
            # Filter out map* IDs that may exist in older cache files
            return {
                k: {pw for pw in v if not pw.startswith("map")}
                for k, v in raw.items()
            }

    # Download from KEGG REST API
    try:
        ko_map = _download_kegg_ko_map()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / "kegg_ko_pathway.json"
            with open(cached, "w") as f:
                json.dump({k: sorted(v) for k, v in ko_map.items()}, f)
            logger.info("Cached KO-pathway map to %s (%d KOs)", cached, len(ko_map))
        return ko_map
    except Exception as e:
        logger.warning(
            "Could not download KEGG KO-pathway mapping: %s. "
            "Pathway enrichment will be skipped. Supply --kegg-ko-pathway "
            "for offline use.",
            e,
        )
        return {}


def load_pathway_names(
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Load KEGG pathway ID -> pathway name mapping.

    Returns:
        Dict mapping pathway ID (e.g. 'ko00010') to name (e.g. 'Glycolysis').
    """
    if cache_dir is not None:
        cached = cache_dir / "kegg_pathway_names.json"
        if cached.exists():
            with open(cached) as f:
                return json.load(f)

    try:
        names = _download_kegg_pathway_names()
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir / "kegg_pathway_names.json", "w") as f:
                json.dump(names, f)
        return names
    except Exception as e:
        logger.warning("Could not load KEGG pathway names: %s", e)
        return {}


def _load_user_ko_map(path: Path) -> dict[str, set[str]]:
    """Parse a user-supplied KO-pathway TSV file.

    Expected format (no header): KO_accession<tab>pathway_id
    Lines starting with '#' are ignored.
    """
    ko_map: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                ko = parts[0].strip().replace("ko:", "")
                pathway = parts[1].strip().replace("path:", "")
                # Skip map* duplicates (same pathway as the ko* entry)
                if pathway.startswith("map"):
                    continue
                ko_map[ko].add(pathway)
    logger.info("Loaded user KO-pathway map: %d KOs from %s", len(ko_map), path)
    return dict(ko_map)


def _download_kegg_ko_map(max_retries: int = 3) -> dict[str, set[str]]:
    """Download KO-to-pathway link table from KEGG REST API.

    Retries up to *max_retries* times with exponential back-off on transient
    network errors (timeout, temporary server failure, etc.).
    """
    import time
    import urllib.error
    import urllib.request

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Downloading KO-pathway mapping from KEGG REST API (attempt %d/%d)...",
                        attempt, max_retries)
            req = urllib.request.Request(_KEGG_LINK_URL, headers={"User-Agent": f"CodonPipe/{__version__}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            break  # success
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("KEGG download attempt %d failed: %s. Retrying in %ds...",
                               attempt, exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"KEGG KO-pathway download failed after {max_retries} attempts: {exc}"
                ) from last_exc

    ko_map: dict[str, set[str]] = defaultdict(set)
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            ko = parts[0].strip().replace("ko:", "")
            pathway = parts[1].strip().replace("path:", "")
            # KEGG returns both ko* and map* IDs for each pathway; keep
            # only ko* to avoid duplicate entries (map* lacks name mappings
            # and represents the same pathway).
            if pathway.startswith("map"):
                continue
            ko_map[ko].add(pathway)

    logger.info("Downloaded %d KO-pathway associations", sum(len(v) for v in ko_map.values()))
    return dict(ko_map)


def _download_kegg_pathway_names(max_retries: int = 3) -> dict[str, str]:
    """Download pathway ID -> name mapping from KEGG REST API.

    Retries up to *max_retries* times with exponential back-off.
    """
    import time
    import urllib.error
    import urllib.request

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(_KEGG_PATHWAY_LIST_URL, headers={"User-Agent": f"CodonPipe/{__version__}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8")
            break  # success
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning("KEGG pathway names download attempt %d failed: %s. Retrying in %ds...",
                               attempt, exc, wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"KEGG pathway names download failed after {max_retries} attempts: {exc}"
                ) from last_exc

    names = {}
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = parts[0].strip().replace("path:", "")
            names[pid] = parts[1].strip()
    return names


# ── Hypergeometric enrichment test ───────────────────────────────────────────

def hypergeometric_enrichment(
    test_kos: set[str],
    background_kos: set[str],
    ko_pathway_map: dict[str, set[str]],
    pathway_names: dict[str, str] | None = None,
    fdr_threshold: float = 0.05,
) -> pd.DataFrame:
    """Test for pathway enrichment in a gene set using the hypergeometric test.

    For each pathway, the test asks: are genes annotated to this pathway
    over-represented in the test set compared to the background?

    The hypergeometric test is parameterized as:
        M = total annotated genes in background (with any pathway assignment)
        n = genes in background assigned to this pathway
        N = annotated genes in the test set
        k = genes in the test set assigned to this pathway

        p-value = P(X >= k) under Hypergeometric(M, n, N)

    Args:
        test_kos: Set of KO accessions in the test gene set (e.g. high-expression).
        background_kos: Set of KO accessions in the full genome (all annotated genes).
        ko_pathway_map: KO -> set of pathway IDs mapping.
        pathway_names: Optional pathway ID -> name mapping for readable output.
        fdr_threshold: Benjamini-Hochberg FDR threshold for significance.

    Returns:
        DataFrame with columns: pathway, pathway_name, k, n, N, M,
        fold_enrichment, p_value, fdr, significant.
        Sorted by p_value ascending.
    """
    if not test_kos or not background_kos or not ko_pathway_map:
        return pd.DataFrame()

    # Map KOs to pathways for background and test sets
    bg_pathway_kos: dict[str, set[str]] = defaultdict(set)
    for ko in background_kos:
        for pw in ko_pathway_map.get(ko, []):
            bg_pathway_kos[pw].add(ko)

    test_pathway_kos: dict[str, set[str]] = defaultdict(set)
    for ko in test_kos:
        for pw in ko_pathway_map.get(ko, []):
            test_pathway_kos[pw].add(ko)

    # Only test pathways that have at least one gene in the test set
    pathways_to_test = [pw for pw in test_pathway_kos if pw in bg_pathway_kos]
    if not pathways_to_test:
        return pd.DataFrame()

    # Population sizes: use all genes, not just those with pathway annotations
    M = len(background_kos)
    N = len(test_kos)

    if M == 0 or N == 0:
        return pd.DataFrame()

    rows = []
    filtered_count = 0
    for pw in pathways_to_test:
        n = len(bg_pathway_kos[pw])  # pathway size in background
        if n < 3:
            filtered_count += 1
            continue  # Skip pathways with fewer than 3 genes to avoid spurious enrichment
        k = len(test_pathway_kos[pw])  # hits in test set

        # Hypergeometric survival function: P(X >= k)
        # scipy parameterization: hypergeom.sf(k-1, M, n, N)
        pval = stats.hypergeom.sf(k - 1, M, n, N)

        expected = (n / M) * N if M > 0 else 0
        fold = k / expected if expected > 0 else np.inf
        # Cap fold enrichment to avoid inf values
        fold = min(fold, 100.0)

        pw_name = (pathway_names or {}).get(pw, "")
        rows.append({
            "pathway": pw,
            "pathway_name": pw_name,
            "k": k,
            "n": n,
            "N": N,
            "M": M,
            "fold_enrichment": round(fold, 3),
            "p_value": pval,
            "test_kos_in_pathway": ",".join(sorted(test_pathway_kos[pw])),
        })

    if filtered_count > 0:
        logger.debug(
            "Filtered %d pathways with fewer than 3 genes during enrichment analysis",
            filtered_count,
        )

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Benjamini-Hochberg FDR correction
    if len(result) > 0:
        result["fdr"] = benjamini_hochberg(result["p_value"].values)
        result["significant"] = result["fdr"] <= fdr_threshold
    else:
        result["fdr"] = pd.Series(dtype=float)
        result["significant"] = pd.Series(dtype=bool)

    return result


# ── Pipeline entry point ─────────────────────────────────────────────────────

def run_enrichment_analysis(
    expr_df: pd.DataFrame,
    kofam_df: pd.DataFrame,
    output_dir: Path,
    sample_id: str,
    ko_pathway_map: dict[str, set[str]] | None = None,
    pathway_names: dict[str, str] | None = None,
    kegg_ko_pathway_file: Path | None = None,
    fdr_threshold: float = 0.05,
    metrics: list[str] | None = None,
    output_subdir: str = "enrichment",
) -> dict[str, Path]:
    """Run pathway enrichment for high- and low-expression gene sets.

    Tests each expression metric independently. For each metric, genes
    classified as 'high' and 'low' are tested separately against the full
    genome as background.

    Args:
        expr_df: Combined expression table with gene, KO, and *_class columns.
        kofam_df: KofamScan parsed output with gene and KO columns.
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        ko_pathway_map: Pre-loaded KO->pathway mapping (avoids re-downloading).
        pathway_names: Pre-loaded pathway names.
        kegg_ko_pathway_file: User-supplied KO-pathway mapping file.
        fdr_threshold: FDR significance threshold.
        metrics: List of metric names to test. For each name ``m``, the
            expression table must contain a column ``{m}_class`` with values
            'high', 'medium', 'low'.  Defaults to ["CAI", "MELP", "Fop"].
        output_subdir: Subdirectory name under output_dir for results.
            Defaults to "enrichment".

    Returns:
        Dict of output file paths.
    """
    enrich_dir = output_dir / output_subdir
    enrich_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Load KO-pathway mapping if not provided
    cache_dir = output_dir / ".cache"
    if ko_pathway_map is None:
        ko_pathway_map = load_ko_pathway_map(
            user_file=kegg_ko_pathway_file, cache_dir=cache_dir,
        )
    if not ko_pathway_map:
        logger.warning(
            "No KO-pathway mapping available for %s; skipping enrichment.", sample_id,
        )
        return outputs

    if pathway_names is None:
        pathway_names = load_pathway_names(cache_dir=cache_dir)

    # Merge KO annotations onto expression data
    ko_col = _find_ko_column(kofam_df)
    gene_col = _find_gene_column(kofam_df)
    if ko_col is None or gene_col is None:
        logger.warning("Could not identify KO/gene columns in KofamScan output")
        return outputs

    # Build gene -> KO mapping. A gene may have multiple KO annotations;
    # keep all of them by joining with semicolons so enrichment sees every
    # pathway a gene participates in.
    ko_groups = kofam_df.groupby(gene_col)[ko_col].apply(
        lambda kos: ";".join(sorted(set(kos.dropna().astype(str))))
    )
    ko_lookup = ko_groups.to_dict()

    # Annotated expression table (gene -> KO, semicolon-separated if >1)
    expr_annotated = expr_df.copy()
    expr_annotated["KO"] = expr_annotated["gene"].map(ko_lookup)

    # Write the full annotated expression table with per-metric classes
    full_table_path = enrich_dir / f"{sample_id}_expression_by_tier.tsv"
    expr_annotated.to_csv(full_table_path, sep="\t", index=False)
    outputs["expression_by_tier"] = full_table_path
    logger.info(
        "Expression tier table: %d genes, %d with KO annotations",
        len(expr_annotated), expr_annotated["KO"].notna().sum(),
    )

    # Helper: explode semicolon-separated KO values into individual KOs
    def _explode_kos(series: pd.Series) -> set[str]:
        kos = set()
        for val in series.dropna():
            for ko in str(val).split(";"):
                ko = ko.strip()
                if ko:
                    kos.add(ko)
        return kos

    # Background: all genes with a KO annotation
    annotated_genes = expr_annotated.dropna(subset=["KO"])
    background_kos = _explode_kos(annotated_genes["KO"])

    # Per-metric enrichment
    if metrics is None:
        metrics = ["CAI", "MELP", "Fop"]
    test_metrics = [m for m in metrics if f"{m}_class" in expr_annotated.columns]

    for metric in test_metrics:
        class_col = f"{metric}_class"
        for tier in ["high", "low"]:
            tier_genes = annotated_genes[annotated_genes[class_col] == tier]
            tier_kos = _explode_kos(tier_genes["KO"])

            if len(tier_kos) < 2:
                logger.info(
                    "Too few annotated %s-%s genes (%d KOs) in %s for enrichment",
                    metric, tier, len(tier_kos), sample_id,
                )
                continue

            result = hypergeometric_enrichment(
                tier_kos, background_kos, ko_pathway_map,
                pathway_names=pathway_names, fdr_threshold=fdr_threshold,
            )

            out_path = enrich_dir / f"{sample_id}_{metric}_{tier}_enrichment.tsv"
            result.to_csv(out_path, sep="\t", index=False)
            outputs[f"enrichment_{metric}_{tier}"] = out_path

            n_sig = result["significant"].sum() if "significant" in result.columns else 0
            logger.info(
                "%s %s-%s enrichment: %d pathways tested, %d significant (FDR < %.2f)",
                sample_id, metric, tier, len(result), n_sig, fdr_threshold,
            )

    return outputs


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_ko_column(df: pd.DataFrame) -> str | None:
    """Find the KO accession column in a KofamScan DataFrame."""
    for col in ("KO", "ko", "KO_id", "ko_id", "accession"):
        if col in df.columns:
            return col
    # Fall back to first column containing K-number patterns
    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str)
        if sample.str.match(r"^K\d{5}$").any():
            return col
    return None


def _find_gene_column(df: pd.DataFrame) -> str | None:
    """Find the gene ID column in a KofamScan DataFrame."""
    from codonpipe.utils.io import find_gene_id_column
    return find_gene_id_column(df)


# ── Gene-level codon usage inefficiency report ────────────────────────────────

def generate_codon_inefficiency_report(
    mahal_clusters_path: Path,
    kofam_df: pd.DataFrame | None,
    enrichment_results: dict[str, pd.DataFrame],
    output_path: Path,
    mahal_summary_path: Path | None = None,
) -> Path | None:
    """Produce a ranked table of all genes ordered by codon usage inefficiency.

    Each gene is ranked by its Mahalanobis distance from the core cluster
    centroid (descending — most divergent first).  The table includes
    KO accession, and, for genes whose KO contributed to a significant
    enrichment result, the names of significantly enriched and significantly
    depleted KEGG pathways.

    Args:
        mahal_clusters_path: Path to ``{sample}_mahal_clusters.tsv``
            (gene, mahalanobis_distance, membership_score, in_optimized_set,
            is_ribosomal_protein).
        kofam_df: KofamScan annotation DataFrame (gene_name, KO, KO_definition).
        enrichment_results: Dict of enrichment DataFrames keyed like
            ``"mahal_enrichment_MELP_high"``, ``"mahal_enrichment_CAI_low"``,
            etc.  Each DataFrame must contain ``pathway``, ``pathway_name``,
            ``test_kos_in_pathway``, and ``significant`` columns.
        output_path: Where to write the TSV.
        mahal_summary_path: Optional path to ``{sample}_mahal_summary.tsv``
            used to normalize distances to the cluster threshold.

    Returns:
        output_path on success, None if no data was available.
    """
    if not mahal_clusters_path.exists():
        logger.warning("Mahalanobis clusters file not found: %s", mahal_clusters_path)
        return None

    clusters = pd.read_csv(mahal_clusters_path, sep="\t")
    if "gene" not in clusters.columns or "mahalanobis_distance" not in clusters.columns:
        logger.warning("mahal_clusters.tsv missing expected columns")
        return None

    # Normalize distance to cluster threshold if available
    threshold = None
    if mahal_summary_path is not None and mahal_summary_path.exists():
        try:
            summary = pd.read_csv(mahal_summary_path, sep="\t")
            if "mahalanobis_threshold" in summary.columns:
                threshold = float(summary["mahalanobis_threshold"].iloc[0])
        except Exception:
            pass

    if threshold and threshold > 0:
        clusters["distance_ratio"] = (
            clusters["mahalanobis_distance"] / threshold
        ).round(4)
    else:
        clusters["distance_ratio"] = np.nan

    # Merge KO accession and definition
    if kofam_df is not None and not kofam_df.empty:
        ko_col = _find_ko_column(kofam_df)
        gene_col = _find_gene_column(kofam_df)
        if ko_col and gene_col:
            ko_info = kofam_df[[gene_col, ko_col]].copy()
            if "KO_definition" in kofam_df.columns:
                ko_info = kofam_df[[gene_col, ko_col, "KO_definition"]].copy()
            ko_info = ko_info.rename(columns={gene_col: "gene", ko_col: "KO"})
            clusters = clusters.merge(ko_info, on="gene", how="left")
    if "KO" not in clusters.columns:
        clusters["KO"] = np.nan
    if "KO_definition" not in clusters.columns:
        clusters["KO_definition"] = np.nan

    # Build KO → significant pathway mappings from enrichment results.
    # "low" tiers represent pathways depleted in efficiently-biased genes
    # (i.e. over-represented among poorly-optimized genes).
    # "high" tiers represent pathways enriched among efficiently-biased genes.
    ko_to_depleted: dict[str, set[str]] = defaultdict(set)
    ko_to_enriched: dict[str, set[str]] = defaultdict(set)

    for key, enrich_df in enrichment_results.items():
        if not isinstance(enrich_df, pd.DataFrame) or enrich_df.empty:
            continue
        if "significant" not in enrich_df.columns:
            continue

        sig_rows = enrich_df[enrich_df["significant"] == True]  # noqa: E712
        if sig_rows.empty:
            continue

        # Determine whether this is a "low" or "high" tier enrichment
        key_lower = key.lower()
        is_low = "_low" in key_lower
        is_high = "_high" in key_lower
        if not is_low and not is_high:
            continue

        for _, row in sig_rows.iterrows():
            kos_str = row.get("test_kos_in_pathway", "")
            pw_name = row.get("pathway_name", row.get("pathway", ""))
            pw_id = row.get("pathway", "")
            label = f"{pw_name} ({pw_id})" if pw_name else pw_id

            if not kos_str or pd.isna(kos_str):
                continue
            for ko in str(kos_str).split(","):
                ko = ko.strip()
                if not ko:
                    continue
                if is_low:
                    ko_to_depleted[ko].add(label)
                if is_high:
                    ko_to_enriched[ko].add(label)

    # Map pathway annotations onto genes via their KO
    def _lookup_pathways(ko_val, mapping: dict[str, set[str]]) -> str:
        if pd.isna(ko_val):
            return ""
        hits = set()
        for ko in str(ko_val).split(";"):
            ko = ko.strip()
            if ko in mapping:
                hits.update(mapping[ko])
        return "; ".join(sorted(hits))

    clusters["pathways_significantly_depleted"] = clusters["KO"].apply(
        lambda ko: _lookup_pathways(ko, ko_to_depleted)
    )
    clusters["pathways_significantly_enriched"] = clusters["KO"].apply(
        lambda ko: _lookup_pathways(ko, ko_to_enriched)
    )

    # Rank by core-relative CAI (ascending — lowest CAI = most divergent
    # from core cluster preferences).  Fall back to Mahalanobis distance
    # descending if core_CAI is absent.
    if "core_CAI" in clusters.columns and clusters["core_CAI"].notna().any():
        clusters = clusters.sort_values("core_CAI", ascending=True).reset_index(drop=True)
    else:
        clusters = clusters.sort_values("mahalanobis_distance", ascending=False).reset_index(drop=True)
    clusters.index = clusters.index + 1
    clusters.index.name = "rank"

    # Select and order output columns
    out_cols = [
        "gene",
        "gene_length_codons",
        "core_CAI",
        "rare_codon_freq",
        "rare_codon_burden",
        "n_core_rare_codons",
        "n_genome_rare_codons",
        "mahalanobis_distance",
        "distance_ratio",
        "in_optimized_set",
        "is_ribosomal_protein",
        "KO",
        "KO_definition",
        "pathways_significantly_depleted",
        "pathways_significantly_enriched",
        "membership_score",
    ]
    out_cols = [c for c in out_cols if c in clusters.columns]
    clusters = clusters[out_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(output_path, sep="\t")
    logger.info(
        "Codon inefficiency report: %d genes ranked, %d with KO, "
        "%d linked to depleted pathways, %d linked to enriched pathways",
        len(clusters),
        clusters["KO"].notna().sum(),
        (clusters["pathways_significantly_depleted"] != "").sum(),
        (clusters["pathways_significantly_enriched"] != "").sum(),
    )
    return output_path
