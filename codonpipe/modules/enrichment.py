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
            return {k: set(v) for k, v in raw.items()}

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
    except Exception:
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
                ko_map[ko].add(pathway)
    logger.info("Loaded user KO-pathway map: %d KOs from %s", len(ko_map), path)
    return dict(ko_map)


def _download_kegg_ko_map() -> dict[str, set[str]]:
    """Download KO-to-pathway link table from KEGG REST API."""
    import urllib.request

    logger.info("Downloading KO-pathway mapping from KEGG REST API...")
    req = urllib.request.Request(_KEGG_LINK_URL, headers={"User-Agent": "CodonPipe/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    ko_map: dict[str, set[str]] = defaultdict(set)
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            ko = parts[0].strip().replace("ko:", "")
            pathway = parts[1].strip().replace("path:", "")
            ko_map[ko].add(pathway)

    logger.info("Downloaded %d KO-pathway associations", sum(len(v) for v in ko_map.values()))
    return dict(ko_map)


def _download_kegg_pathway_names() -> dict[str, str]:
    """Download pathway ID -> name mapping from KEGG REST API."""
    import urllib.request

    req = urllib.request.Request(_KEGG_PATHWAY_LIST_URL, headers={"User-Agent": "CodonPipe/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")

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
    for pw in pathways_to_test:
        n = len(bg_pathway_kos[pw])  # pathway size in background
        if n < 3:
            continue  # Skip pathways with fewer than 3 genes to avoid spurious enrichment
        k = len(test_pathway_kos[pw])  # hits in test set

        # Hypergeometric survival function: P(X >= k)
        # scipy parameterization: hypergeom.sf(k-1, M, n, N)
        pval = stats.hypergeom.sf(k - 1, M, n, N)

        expected = (n / M) * N if M > 0 else 0
        fold = k / expected if expected > 0 else np.inf

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

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Benjamini-Hochberg FDR correction
    n_tests = len(result)
    if n_tests > 0:
        ranked_pvals = result["p_value"].values
        bh_fdr = np.zeros(n_tests)
        for i in range(n_tests):
            bh_fdr[i] = ranked_pvals[i] * n_tests / (i + 1)
        # Enforce monotonicity (cumulative minimum from the right)
        for i in range(n_tests - 2, -1, -1):
            bh_fdr[i] = min(bh_fdr[i], bh_fdr[i + 1])
        bh_fdr = np.minimum(bh_fdr, 1.0)
        result["fdr"] = bh_fdr
        result["significant"] = result["fdr"] <= fdr_threshold
    else:
        result["fdr"] = []
        result["significant"] = []

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
) -> dict[str, Path]:
    """Run pathway enrichment for high- and low-expression gene sets.

    Tests each expression metric (CAI, MELP, Fop) independently. For each
    metric, genes classified as 'high' and 'low' are tested separately
    against the full genome as background.

    Args:
        expr_df: Combined expression table with gene, KO, and *_class columns.
        kofam_df: KofamScan parsed output with gene and KO columns.
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        ko_pathway_map: Pre-loaded KO->pathway mapping (avoids re-downloading).
        pathway_names: Pre-loaded pathway names.
        kegg_ko_pathway_file: User-supplied KO-pathway mapping file.
        fdr_threshold: FDR significance threshold.

    Returns:
        Dict of output file paths.
    """
    enrich_dir = output_dir / "enrichment"
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

    ko_lookup = dict(zip(kofam_df[gene_col], kofam_df[ko_col]))

    # Annotated expression table (gene -> KO)
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

    # Background: all genes with a KO annotation
    annotated_genes = expr_annotated.dropna(subset=["KO"])
    background_kos = set(annotated_genes["KO"].unique())

    # Per-metric enrichment
    metrics = [m for m in ["CAI", "MELP", "Fop"] if f"{m}_class" in expr_annotated.columns]

    for metric in metrics:
        class_col = f"{metric}_class"
        for tier in ["high", "low"]:
            tier_genes = annotated_genes[annotated_genes[class_col] == tier]
            tier_kos = set(tier_genes["KO"].unique())

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
    for col in ("gene_name", "gene", "gene_id", "query", "query_id"):
        if col in df.columns:
            return col
    return None
