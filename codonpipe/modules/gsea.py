"""Pre-ranked Gene Set Enrichment Analysis for codon usage data.

Implements the weighted Kolmogorov-Smirnov statistic from Subramanian et al.
(2003) with permutation-based significance testing.  Designed for microbial
genomes where annotation coverage is typically sparse.

Gene set sources
----------------
* **KEGG Modules** — compact, prokaryote-friendly functional units (primary)
* **KEGG Pathways** — broader metabolic maps (already in pipeline)
* **COG categories** — coarse functional classification (always available)

The primary ranking metric is core-relative CAI divergence (1 − core_CAI),
where core_CAI is computed against the distance-weighted RSCU of the
Mahalanobis core cluster.  Falls back to raw Mahalanobis distance when
core_CAI is unavailable.  Higher values = more divergent codon usage.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from codonpipe import __version__
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")

# ── Constants ────────────────────────────────────────────────────────────────

_KEGG_LINK_MODULE_URL = "https://rest.kegg.jp/link/module/ko"
_KEGG_LIST_MODULE_URL = "https://rest.kegg.jp/list/module"
_MIN_GENE_SET_SIZE = 5
_MAX_GENE_SET_SIZE = 500
_DEFAULT_PERMUTATIONS = 1000
_KEGG_MAX_RETRIES = 3
_KEGG_RETRY_DELAY = 5.0


# ── Pre-ranked GSEA algorithm ───────────────────────────────────────────────

def _enrichment_score(
    ranked_genes: np.ndarray,
    gene_set_mask: np.ndarray,
    weights: np.ndarray,
    exponent: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute the weighted running-sum enrichment score.

    Args:
        ranked_genes: Gene indices in rank order (not used directly; implied
            by ordering of gene_set_mask and weights).
        gene_set_mask: Boolean array, True where ranked gene is in the set.
        weights: Absolute ranking metric values (e.g. Mahalanobis distance)
            in the same order as ranked_genes.
        exponent: Weight exponent (1.0 = standard GSEA).

    Returns:
        (ES, running_sum) where ES is the maximum-deviation enrichment score
        and running_sum is the full running-sum curve.
    """
    n = len(gene_set_mask)
    n_hit = gene_set_mask.sum()
    if n_hit == 0 or n_hit == n:
        return 0.0, np.zeros(n)

    hit_weights = np.where(gene_set_mask, np.abs(weights) ** exponent, 0.0)
    nr = hit_weights.sum()
    if nr == 0:
        return 0.0, np.zeros(n)

    n_miss = n - n_hit
    hit_score = hit_weights / nr
    miss_score = np.where(~gene_set_mask, 1.0 / n_miss, 0.0)

    running_sum = np.cumsum(hit_score - miss_score)

    # ES = maximum deviation from zero (signed)
    max_pos = running_sum.max()
    max_neg = running_sum.min()
    es = max_pos if abs(max_pos) >= abs(max_neg) else max_neg

    return float(es), running_sum


def _permutation_es(
    gene_set_mask: np.ndarray,
    weights: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    exponent: float = 1.0,
) -> np.ndarray:
    """Generate null distribution of ES via gene-label permutation.

    Uses vectorised batch permutation: shuffles the gene-set mask across all
    permutations at once, then computes running-sum enrichment scores via
    cumulative sums on the full (n_perm × n_genes) matrix.  ~10× faster than
    the per-permutation loop for n_perm ≥ 1000.
    """
    n = len(gene_set_mask)
    n_hit = int(gene_set_mask.sum())
    n_miss = n - n_hit

    if n_hit == 0 or n_hit == n or n_miss == 0:
        return np.zeros(n_perm)

    abs_w = np.abs(weights) ** exponent

    # Build (n_perm × n) shuffled masks in one go
    idx = np.argsort(rng.random((n_perm, n)), axis=1)
    perm_masks = gene_set_mask[idx]  # (n_perm, n) boolean

    # Vectorised hit / miss scores per permutation
    hit_weights = np.where(perm_masks, abs_w[np.newaxis, :], 0.0)
    nr = hit_weights.sum(axis=1, keepdims=True)
    nr = np.where(nr == 0, 1.0, nr)  # avoid div-by-zero
    hit_score = hit_weights / nr
    miss_score = np.where(~perm_masks, 1.0 / n_miss, 0.0)

    running = np.cumsum(hit_score - miss_score, axis=1)

    max_pos = running.max(axis=1)
    max_neg = running.min(axis=1)
    null_es = np.where(np.abs(max_pos) >= np.abs(max_neg), max_pos, max_neg)

    return null_es


def run_preranked_gsea(
    ranked_df: pd.DataFrame,
    gene_sets: dict[str, set[str]],
    gene_col: str = "gene",
    metric_col: str = "mahalanobis_distance",
    n_perm: int = _DEFAULT_PERMUTATIONS,
    min_size: int = _MIN_GENE_SET_SIZE,
    max_size: int = _MAX_GENE_SET_SIZE,
    exponent: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Run pre-ranked GSEA on a gene list ranked by a continuous metric.

    Args:
        ranked_df: DataFrame with at least ``gene_col`` and ``metric_col``.
            Must be pre-sorted by ``metric_col`` descending (most divergent
            first).
        gene_sets: Mapping of gene-set name to set of gene identifiers
            (e.g. KO accessions).
        gene_col: Column containing gene identifiers that match gene_sets.
        metric_col: Column containing the continuous ranking metric.
        n_perm: Number of permutations for p-value estimation.
        min_size: Minimum gene set size after intersection with ranked list.
        max_size: Maximum gene set size.
        exponent: Weight exponent for the running-sum statistic.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: gene_set, size, es, nes, p_value, fdr,
        leading_edge_size, leading_edge_genes.
    """
    rng = np.random.default_rng(seed)
    genes = ranked_df[gene_col].values
    weights = ranked_df[metric_col].values.astype(float)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n = len(genes)

    rows = []
    all_null_pos = []
    all_null_neg = []

    for gs_name, gs_genes in gene_sets.items():
        # Intersect gene set with ranked list
        hits = gs_genes & set(genes)
        if len(hits) < min_size or len(hits) > max_size:
            continue

        mask = np.array([g in hits for g in genes], dtype=bool)
        es, running_sum = _enrichment_score(np.arange(n), mask, weights, exponent)

        # Permutation null
        null_es = _permutation_es(mask, weights, n_perm, rng, exponent)

        # Separate positive and negative null distributions for NES
        pos_null = null_es[null_es >= 0]
        neg_null = null_es[null_es < 0]

        # Nominal p-value
        if es >= 0:
            p_val = (np.sum(null_es >= es) + 1) / (n_perm + 1)
        else:
            p_val = (np.sum(null_es <= es) + 1) / (n_perm + 1)

        # Normalized enrichment score
        if es >= 0 and len(pos_null) > 0:
            nes = es / np.mean(pos_null) if np.mean(pos_null) > 0 else 0.0
        elif es < 0 and len(neg_null) > 0:
            nes = es / abs(np.mean(neg_null)) if np.mean(neg_null) != 0 else 0.0
        else:
            nes = 0.0

        # Leading edge: genes contributing to the ES before the peak
        if es >= 0:
            peak_idx = np.argmax(running_sum)
        else:
            peak_idx = np.argmin(running_sum)
        leading_edge = [genes[i] for i in range(peak_idx + 1) if mask[i]]

        all_null_pos.append(pos_null)
        all_null_neg.append(neg_null)

        rows.append({
            "gene_set": gs_name,
            "size": int(mask.sum()),
            "es": round(es, 6),
            "nes": round(nes, 4),
            "p_value": p_val,
            "leading_edge_size": len(leading_edge),
            "leading_edge_genes": ",".join(leading_edge),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "gene_set", "size", "es", "nes", "p_value", "fdr",
            "leading_edge_size", "leading_edge_genes",
        ])

    result = pd.DataFrame(rows)
    result["fdr"] = benjamini_hochberg(result["p_value"].values)
    result = result.sort_values("p_value").reset_index(drop=True)

    return result


# ── Gene set construction ────────────────────────────────────────────────────

def build_gene_sets_from_kofam(
    kofam_df: pd.DataFrame,
    ko_set_map: dict[str, set[str]],
    set_names: dict[str, str] | None = None,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build gene-level gene sets from KofamScan annotations and a KO→set map.

    Genes inherit set membership from their KO annotations.  A gene with
    multiple KOs (semicolon-separated) joins every corresponding set.

    Args:
        kofam_df: KofamScan results with gene_name and KO columns.
        ko_set_map: Mapping from KO accession to set of gene-set IDs.
        set_names: Optional mapping from gene-set ID to human-readable name.

    Returns:
        (gene_sets, display_names) where gene_sets maps gene-set ID to
        set of gene IDs, and display_names maps gene-set ID to a display
        label like "M00001 (Glycolysis)".
    """
    gene_sets: dict[str, set[str]] = defaultdict(set)

    gene_col = None
    for c in ("gene_name", "gene", "Gene"):
        if c in kofam_df.columns:
            gene_col = c
            break
    ko_col = "KO" if "KO" in kofam_df.columns else None
    if gene_col is None or ko_col is None:
        return {}, {}

    for _, row in kofam_df.iterrows():
        gene = row[gene_col]
        ko_val = row[ko_col]
        if pd.isna(ko_val):
            continue
        for ko in str(ko_val).split(";"):
            ko = ko.strip()
            if ko in ko_set_map:
                for gs_id in ko_set_map[ko]:
                    gene_sets[gs_id].add(gene)

    display_names = {}
    names = set_names or {}
    for gs_id in gene_sets:
        name = names.get(gs_id, "")
        display_names[gs_id] = f"{gs_id} ({name})" if name else gs_id

    return dict(gene_sets), display_names


def build_cog_gene_sets(
    cog_result_path: Path,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build gene sets from COG functional category assignments.

    Args:
        cog_result_path: Path to COGclassifier result TSV.

    Returns:
        (gene_sets, display_names) keyed by COG category letter.
    """
    _COG_NAMES = {
        "J": "Translation, ribosomal structure and biogenesis",
        "A": "RNA processing and modification",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "B": "Chromatin structure and dynamics",
        "D": "Cell cycle control, cell division",
        "Y": "Nuclear structure",
        "V": "Defense mechanisms",
        "T": "Signal transduction mechanisms",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "Z": "Cytoskeleton",
        "W": "Extracellular structures",
        "U": "Intracellular trafficking, secretion",
        "O": "Post-translational modification, protein turnover",
        "C": "Energy production and conversion",
        "G": "Carbohydrate transport and metabolism",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis",
        "R": "General function prediction only",
        "S": "Function unknown",
        "X": "Mobilome: prophages, transposons",
    }

    if not cog_result_path.exists():
        return {}, {}

    try:
        df = pd.read_csv(cog_result_path, sep="\t")
    except Exception:
        return {}, {}

    gene_col = None
    for c in ("QUERY_ID", "gene", "gene_name"):
        if c in df.columns:
            gene_col = c
            break
    cog_cat_col = None
    for c in ("COG_LETTER", "cog_letter", "COG_CATEGORY", "functional_category"):
        if c in df.columns:
            cog_cat_col = c
            break
    if gene_col is None or cog_cat_col is None:
        return {}, {}

    gene_sets: dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        gene = row[gene_col]
        cats = str(row[cog_cat_col]) if pd.notna(row[cog_cat_col]) else ""
        for letter in cats:
            if letter.isalpha():
                gene_sets[letter].add(gene)

    display_names = {
        k: f"COG:{k} ({_COG_NAMES.get(k, 'Unknown')})"
        for k in gene_sets
    }

    return dict(gene_sets), display_names


# ── KEGG Module fetching ─────────────────────────────────────────────────────

def _download_kegg_ko_module_map() -> dict[str, set[str]]:
    """Download KO-to-module mapping from KEGG REST API."""
    ko_module: dict[str, set[str]] = defaultdict(set)
    for attempt in range(1, _KEGG_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                _KEGG_LINK_MODULE_URL,
                headers={"User-Agent": f"CodonPipe/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp.read().decode("utf-8").splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        ko = parts[0].replace("ko:", "")
                        mod = parts[1].replace("md:", "")
                        ko_module[ko].add(mod)
            logger.info("Downloaded KO-module mapping: %d KOs → modules", len(ko_module))
            return dict(ko_module)
        except (urllib.error.URLError, OSError) as e:
            if attempt < _KEGG_MAX_RETRIES:
                logger.warning(
                    "KEGG module download attempt %d/%d failed: %s. Retrying in %.0fs.",
                    attempt, _KEGG_MAX_RETRIES, e, _KEGG_RETRY_DELAY * attempt,
                )
                time.sleep(_KEGG_RETRY_DELAY * attempt)
            else:
                logger.warning("KEGG module download failed after %d attempts: %s", _KEGG_MAX_RETRIES, e)
    return {}


def _download_kegg_module_names() -> dict[str, str]:
    """Download module ID → name mapping from KEGG."""
    names: dict[str, str] = {}
    for attempt in range(1, _KEGG_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                _KEGG_LIST_MODULE_URL,
                headers={"User-Agent": f"CodonPipe/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp.read().decode("utf-8").splitlines():
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        mod_id = parts[0].replace("md:", "")
                        names[mod_id] = parts[1]
            logger.info("Downloaded KEGG module names: %d modules", len(names))
            return names
        except (urllib.error.URLError, OSError) as e:
            if attempt < _KEGG_MAX_RETRIES:
                time.sleep(_KEGG_RETRY_DELAY * attempt)
            else:
                logger.warning("KEGG module names download failed: %s", e)
    return {}


def load_ko_module_map(
    user_file: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, set[str]]:
    """Load KO→module mapping, with caching and user-file override.

    Priority: user file > cached JSON > KEGG API download.
    """
    # User-supplied file
    if user_file is not None and user_file.exists():
        ko_mod: dict[str, set[str]] = defaultdict(set)
        with open(user_file) as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    ko_mod[parts[0]].add(parts[1])
        logger.info("Loaded KO-module mapping from user file: %d KOs", len(ko_mod))
        return dict(ko_mod)

    # Cached JSON
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "kegg_ko_module.json"
        if cache_path.exists():
            try:
                with open(cache_path) as fh:
                    raw = json.load(fh)
                mapping = {k: set(v) for k, v in raw.items()}
                logger.info("Loaded cached KO-module mapping: %d KOs", len(mapping))
                return mapping
            except Exception:
                pass

    # Download
    mapping = _download_kegg_ko_module_map()
    if mapping and cache_path is not None:
        try:
            serializable = {k: sorted(v) for k, v in mapping.items()}
            with open(cache_path, "w") as fh:
                json.dump(serializable, fh)
            logger.info("Cached KO-module mapping to %s", cache_path)
        except Exception:
            pass
    return mapping


def load_module_names(
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Load module ID → name mapping with caching."""
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "kegg_module_names.json"
        if cache_path.exists():
            try:
                with open(cache_path) as fh:
                    return json.load(fh)
            except Exception:
                pass

    names = _download_kegg_module_names()
    if names and cache_path is not None:
        try:
            with open(cache_path, "w") as fh:
                json.dump(names, fh)
        except Exception:
            pass
    return names


# ── Orchestrator for single-genome GSEA ──────────────────────────────────────

def run_gsea_analysis(
    mahal_clusters_path: Path,
    kofam_df: pd.DataFrame | None,
    output_dir: Path,
    sample_id: str,
    ko_pathway_map: dict[str, set[str]] | None = None,
    pathway_names: dict[str, str] | None = None,
    cog_result_path: Path | None = None,
    n_perm: int = _DEFAULT_PERMUTATIONS,
    min_size: int = _MIN_GENE_SET_SIZE,
    max_size: int = _MAX_GENE_SET_SIZE,
    cache_dir: Path | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run GSEA on a single genome using multiple gene-set sources.

    Args:
        mahal_clusters_path: Path to mahal_clusters.tsv with per-gene
            Mahalanobis distances.
        kofam_df: KofamScan annotation DataFrame.
        output_dir: Base output directory for this sample.
        sample_id: Sample identifier.
        ko_pathway_map: KO→pathway mapping (reused from enrichment module).
        pathway_names: Pathway ID→name mapping.
        cog_result_path: Path to COGclassifier result TSV.
        n_perm: Number of permutations.
        min_size: Minimum gene set size.
        max_size: Maximum gene set size.
        cache_dir: Directory for KEGG API cache.

    Returns:
        Dict with keys like "gsea_modules", "gsea_pathways", "gsea_cog"
        mapping to result DataFrames and output Paths.
    """
    results: dict[str, pd.DataFrame | Path] = {}

    if not mahal_clusters_path.exists():
        logger.warning("GSEA: mahal_clusters.tsv not found; skipping")
        return results

    clusters = pd.read_csv(mahal_clusters_path, sep="\t")
    if "gene" not in clusters.columns or "mahalanobis_distance" not in clusters.columns:
        logger.warning("GSEA: mahal_clusters.tsv missing expected columns; skipping")
        return results

    # Determine ranking metric.  Prefer core_CAI (1 − CAI gives a
    # divergence score where high = poorly adapted to core cluster).
    # Fall back to Mahalanobis distance when core_CAI is unavailable.
    if "core_CAI" in clusters.columns and clusters["core_CAI"].notna().any():
        clusters["core_CU_divergence"] = 1.0 - clusters["core_CAI"]
        _metric_col = "core_CU_divergence"
        logger.info("GSEA ranking metric: core_CU_divergence (1 − core_CAI)")
    else:
        _metric_col = "mahalanobis_distance"
        logger.info("GSEA ranking metric: mahalanobis_distance (core_CAI unavailable)")

    # Sort descending: most divergent genes first
    ranked = clusters.sort_values(_metric_col, ascending=False).reset_index(drop=True)

    gsea_dir = output_dir / "gsea"
    gsea_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = output_dir / ".cache"

    # For KO-based gene sets, we need to map genes to their KO first,
    # then use KO as the gene identifier for GSEA.  But since multiple genes
    # can share a KO, we work at the gene level: each gene inherits its
    # KO's set memberships directly.

    # ── KEGG Modules ─────────────────────────────────────────────────────
    if kofam_df is not None and not kofam_df.empty:
        try:
            ko_mod_map = load_ko_module_map(cache_dir=cache_dir)
            mod_names = load_module_names(cache_dir=cache_dir)

            if ko_mod_map:
                mod_gene_sets, mod_display = build_gene_sets_from_kofam(
                    kofam_df, ko_mod_map, mod_names,
                )
                if mod_gene_sets:
                    gsea_mod = run_preranked_gsea(
                        ranked, mod_gene_sets,
                        gene_col="gene", metric_col=_metric_col,
                        n_perm=n_perm, min_size=min_size, max_size=max_size,
                    )
                    # Replace gene_set IDs with display names
                    gsea_mod["gene_set_id"] = gsea_mod["gene_set"]
                    gsea_mod["gene_set"] = gsea_mod["gene_set_id"].map(
                        lambda x: mod_display.get(x, x)
                    )
                    out_path = gsea_dir / f"{sample_id}_gsea_modules.tsv"
                    gsea_mod.to_csv(out_path, sep="\t", index=False)
                    results["gsea_modules"] = gsea_mod
                    results["gsea_modules_path"] = out_path
                    n_sig = (gsea_mod["fdr"] < 0.25).sum()
                    logger.info(
                        "GSEA modules: %d sets tested, %d significant (FDR < 0.25)",
                        len(gsea_mod), n_sig,
                    )
        except Exception as e:
            logger.warning("GSEA KEGG Modules failed: %s", e, exc_info=True)

    # ── KEGG Pathways ────────────────────────────────────────────────────
    if kofam_df is not None and not kofam_df.empty and ko_pathway_map:
        try:
            pw_gene_sets, pw_display = build_gene_sets_from_kofam(
                kofam_df, ko_pathway_map, pathway_names,
            )
            if pw_gene_sets:
                gsea_pw = run_preranked_gsea(
                    ranked, pw_gene_sets,
                    gene_col="gene", metric_col=_metric_col,
                    n_perm=n_perm, min_size=min_size, max_size=max_size,
                )
                gsea_pw["gene_set_id"] = gsea_pw["gene_set"]
                gsea_pw["gene_set"] = gsea_pw["gene_set_id"].map(
                    lambda x: pw_display.get(x, x)
                )
                out_path = gsea_dir / f"{sample_id}_gsea_pathways.tsv"
                gsea_pw.to_csv(out_path, sep="\t", index=False)
                results["gsea_pathways"] = gsea_pw
                results["gsea_pathways_path"] = out_path
                n_sig = (gsea_pw["fdr"] < 0.25).sum()
                logger.info(
                    "GSEA pathways: %d sets tested, %d significant (FDR < 0.25)",
                    len(gsea_pw), n_sig,
                )
        except Exception as e:
            logger.warning("GSEA KEGG Pathways failed: %s", e, exc_info=True)

    # ── COG categories ───────────────────────────────────────────────────
    if cog_result_path is not None and cog_result_path.exists():
        try:
            cog_gene_sets, cog_display = build_cog_gene_sets(cog_result_path)
            if cog_gene_sets:
                gsea_cog = run_preranked_gsea(
                    ranked, cog_gene_sets,
                    gene_col="gene", metric_col=_metric_col,
                    n_perm=n_perm, min_size=min_size, max_size=max_size,
                )
                gsea_cog["gene_set_id"] = gsea_cog["gene_set"]
                gsea_cog["gene_set"] = gsea_cog["gene_set_id"].map(
                    lambda x: cog_display.get(x, x)
                )
                out_path = gsea_dir / f"{sample_id}_gsea_cog.tsv"
                gsea_cog.to_csv(out_path, sep="\t", index=False)
                results["gsea_cog"] = gsea_cog
                results["gsea_cog_path"] = out_path
                n_sig = (gsea_cog["fdr"] < 0.25).sum()
                logger.info(
                    "GSEA COG categories: %d sets tested, %d significant (FDR < 0.25)",
                    len(gsea_cog), n_sig,
                )
        except Exception as e:
            logger.warning("GSEA COG categories failed: %s", e, exc_info=True)

    return results


# ── Batch comparative GSEA ───────────────────────────────────────────────────

def compare_gsea_between_samples(
    sample_gsea: dict[str, pd.DataFrame],
    source: str = "modules",
) -> pd.DataFrame:
    """Compare NES values across samples for a given gene-set source.

    Produces a wide-format matrix: rows = gene sets, columns = sample NES.
    Useful for heatmaps and pairwise comparisons.

    Args:
        sample_gsea: Mapping from sample_id to GSEA result DataFrame.
        source: Label for the gene-set source (for logging).

    Returns:
        DataFrame with gene_set as index and one NES column per sample.
    """
    if not sample_gsea:
        return pd.DataFrame()

    frames = []
    for sid, df in sample_gsea.items():
        if df.empty:
            continue
        sub = df[["gene_set", "nes"]].copy()
        sub = sub.rename(columns={"nes": sid})
        sub = sub.set_index("gene_set")
        frames.append(sub)

    if not frames:
        return pd.DataFrame()

    combined = frames[0]
    for f in frames[1:]:
        combined = combined.join(f, how="outer")

    logger.info(
        "GSEA %s comparison: %d gene sets across %d samples",
        source, len(combined), len(sample_gsea),
    )
    return combined


def compare_gsea_between_conditions(
    sample_gsea: dict[str, pd.DataFrame],
    sample_conditions: dict[str, str],
    source: str = "modules",
) -> pd.DataFrame:
    """Statistical comparison of GSEA NES between conditions.

    For each gene set, performs Mann-Whitney U (2 conditions) or
    Kruskal-Wallis (3+) on the per-sample NES values.

    Args:
        sample_gsea: sample_id → GSEA DataFrame.
        sample_conditions: sample_id → condition label.
        source: Gene-set source label.

    Returns:
        DataFrame with gene_set, test, statistic, p_value, fdr,
        plus per-condition mean NES columns.
    """
    from scipy import stats as sp_stats

    nes_matrix = compare_gsea_between_samples(sample_gsea, source)
    if nes_matrix.empty:
        return pd.DataFrame()

    conditions = sorted(set(sample_conditions.values()))
    if len(conditions) < 2:
        return pd.DataFrame()

    # Group columns by condition
    cond_cols: dict[str, list[str]] = defaultdict(list)
    for sid, cond in sample_conditions.items():
        if sid in nes_matrix.columns:
            cond_cols[cond].append(sid)

    # Need at least 2 samples per condition for a meaningful test
    valid_conditions = [c for c in conditions if len(cond_cols[c]) >= 2]
    if len(valid_conditions) < 2:
        logger.info(
            "GSEA %s between-condition comparison: fewer than 2 conditions "
            "with >= 2 samples each; skipping", source,
        )
        return pd.DataFrame()

    rows = []
    for gs_name in nes_matrix.index:
        groups = []
        for cond in valid_conditions:
            vals = nes_matrix.loc[gs_name, cond_cols[cond]].dropna().values
            if len(vals) < 2:
                break
            groups.append(vals)

        if len(groups) < 2:
            continue

        if len(valid_conditions) == 2:
            stat, p_val = sp_stats.mannwhitneyu(
                groups[0], groups[1], alternative="two-sided",
            )
            test_name = "mann_whitney_u"
        else:
            stat, p_val = sp_stats.kruskal(*groups)
            test_name = "kruskal_wallis"

        row = {
            "gene_set": gs_name,
            "test": test_name,
            "statistic": round(stat, 4),
            "p_value": p_val,
        }
        for cond in valid_conditions:
            vals = nes_matrix.loc[gs_name, cond_cols[cond]].dropna().values
            row[f"mean_nes_{cond}"] = round(np.mean(vals), 4) if len(vals) > 0 else np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["fdr"] = benjamini_hochberg(result["p_value"].values)
    result["significant"] = result["fdr"] < 0.05
    return result.sort_values("p_value").reset_index(drop=True)
