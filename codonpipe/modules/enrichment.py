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
from codonpipe.utils.io import get_output_subdir
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


def _alias_pathway_prefixes(names: dict[str, str]) -> dict[str, str]:
    """Add cross-prefix aliases for KEGG pathway IDs in-place.

    KEGG REST is internally inconsistent about pathway-ID prefixes:
    ``/link/pathway/ko`` returns each KO once with ``path:ko#####`` and
    once with ``path:map#####``. The codonpipe link-file parser drops
    the ``map`` rows and keeps ``ko``. Meanwhile ``/list/pathway/ko``
    can return names keyed on ``map#####`` (older format) or on bare
    five-digit numerics (newer format), even though the URL ends in
    ``/ko``. Without aliasing, the names dict and the link map don't
    share keys — every join misses and pathway plots fall back to
    bare ko##### IDs.

    Strategy: for any key recognisable as a pathway ID, populate all
    three aliases (``ko#####``, ``map#####``, bare digits ``#####``)
    pointing at the same name. Module IDs (``M#####``) and other
    identifiers are unaffected because their format does not match
    any of the three pathway-ID shapes.
    """
    for key, value in list(names.items()):
        digits = None
        if key.startswith("ko") and len(key) == 7 and key[2:].isdigit():
            digits = key[2:]
        elif key.startswith("map") and len(key) == 8 and key[3:].isdigit():
            digits = key[3:]
        elif len(key) == 5 and key.isdigit():
            digits = key
        if digits is not None:
            names.setdefault("ko" + digits, value)
            names.setdefault("map" + digits, value)
            names.setdefault(digits, value)
    return names


def _load_user_names_tsv(path: Path) -> dict[str, str]:
    """Parse a user-supplied 2-column TSV of ``<id>\\t<name>``.

    Strips ``path:``, ``ko:``, and ``md:`` prefixes from the ID column so
    the file format is identical to what the KEGG REST API serves at
    ``/list/pathway/ko`` and ``/list/module``. Lines starting with ``#``
    and blank lines are ignored.

    Pathway-ID prefix aliasing:
        KEGG REST is internally inconsistent about pathway-ID prefixes.
        ``/link/pathway/ko`` returns each KO once with ``path:ko#####``
        and once with ``path:map#####``; codonpipe's link-file parser
        drops the ``map`` rows and keeps ``ko``. ``/list/pathway/ko``
        returns names keyed on ``map#####`` even though the URL ends
        in ``/ko``. Without aliasing, the names dict is keyed on
        ``map00010`` while enrichment looks up ``ko00010`` — every
        join misses and pathway plots fall back to bare ko##### IDs.
        For any 5-digit pathway ID that arrives under one prefix, we
        also store the alternate prefix so callers can look up either.
        Module IDs (``M#####``) and other identifiers are unaffected.
    """
    names: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                for prefix in ("path:", "ko:", "md:"):
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        break
                names[key] = parts[1].strip()
    # Apply pathway-ID aliasing once at the end so callers always get a
    # dict that resolves both ``ko#####`` and ``map#####`` lookups. This
    # is a no-op for module-name TSVs (M##### keys don't match either
    # branch).
    return _alias_pathway_prefixes(names)


def load_pathway_names(
    cache_dir: Path | None = None,
    user_file: Path | None = None,
) -> dict[str, str]:
    """Load KEGG pathway ID -> pathway name mapping.

    Resolution order:
        1. ``user_file`` if supplied and present — a TSV from
           ``https://rest.kegg.jp/list/pathway/ko``. This path lets a
           user fetch the file once on a node with internet access and
           reuse it on offline compute nodes.
        2. JSON cache under ``cache_dir`` (``kegg_pathway_names.json``).
        3. Fresh download from KEGG REST.

    When a user file is provided, the JSON cache is refreshed from it
    so subsequent runs in the same output directory pick up the names
    even without re-passing the flag.

    Returns:
        Dict mapping pathway ID (e.g. 'ko00010') to name (e.g. 'Glycolysis').
    """
    if user_file is not None and user_file.exists():
        names = _load_user_names_tsv(user_file)
        logger.info(
            "Loaded user pathway-names map: %d pathways from %s",
            len(names), user_file,
        )
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_dir / "kegg_pathway_names.json", "w") as fh:
                    json.dump(names, fh)
            except Exception:
                pass
        return names

    if cache_dir is not None:
        cached = cache_dir / "kegg_pathway_names.json"
        if cached.exists():
            with open(cached) as f:
                cached_names = json.load(f)
            # Heal stale caches written before the prefix-alias fix —
            # if the cache only contains map##### keys, lookups for
            # ko##### would still miss without this. Cheap to compute
            # and the result is rewritten so future runs don't redo it.
            healed = _alias_pathway_prefixes(dict(cached_names))
            if len(healed) > len(cached_names):
                try:
                    with open(cached, "w") as f:
                        json.dump(healed, f)
                    logger.info(
                        "Aliased %d ko##### ↔ map##### pathway-name "
                        "entries in stale cache at %s",
                        len(healed) - len(cached_names), cached,
                    )
                except Exception:
                    pass
            return healed

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
    # KEGG returns names keyed on map##### from /list/pathway/ko even
    # though the URL ends in /ko; alias them so ko##### lookups also
    # resolve. See _alias_pathway_prefixes for the full rationale.
    return _alias_pathway_prefixes(names)


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

    # Diagnostic: how many of the pathway IDs we'll be testing actually
    # resolve to a name in the supplied dict? Anything less than ~95% of
    # tested pathways means the names file used different ID formatting
    # than the link file (the classic cause: ko##### vs map##### vs
    # bare digits). Show three sample missing IDs and three sample names
    # so the user can compare formats and fix the mismatch upstream.
    if pathway_names:
        resolved = sum(1 for pw in pathways_to_test if pathway_names.get(pw))
        n_test = len(pathways_to_test)
        if resolved < n_test * 0.5:
            sample_missing = [
                pw for pw in pathways_to_test if not pathway_names.get(pw)
            ][:3]
            sample_have = list(pathway_names.keys())[:3]
            logger.warning(
                "Pathway-name coverage: %d/%d tested pathway IDs resolved "
                "to a name (%.0f%%). The names dict likely uses a "
                "different ID prefix than the link file. Example missing "
                "lookups: %s. Example keys present in names dict: %s.",
                resolved, n_test, 100 * resolved / n_test,
                sample_missing, sample_have,
            )
        else:
            logger.info(
                "Pathway-name coverage: %d/%d tested pathway IDs resolved "
                "to a name (%.0f%%)",
                resolved, n_test, 100 * resolved / n_test,
            )
    else:
        logger.warning(
            "No pathway-names dict supplied to enrichment; pathway plots "
            "will fall back to bare ko##### IDs",
        )

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
    kegg_pathway_names_file: Path | None = None,
    fdr_threshold: float = 0.05,
    metrics: list[str] | None = None,
    output_subdir: str = "enrichment",
    cache_dir: Path | None = None,
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
        cache_dir: Shared KEGG cache directory. If None, uses per-sample
            cache at output_dir / ".cache" for backward compatibility.

    Returns:
        Dict of output file paths.
    """
    enrich_dir = get_output_subdir(output_dir, "expression", output_subdir)
    outputs = {}

    # Load KO-pathway mapping if not provided
    if cache_dir is None:
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
        pathway_names = load_pathway_names(
            cache_dir=cache_dir, user_file=kegg_pathway_names_file,
        )

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

    # Collect all enrichment results
    all_results = []

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

            # Add metric and tier columns
            result["metric"] = metric
            result["tier"] = tier
            all_results.append(result)

            n_sig = result["significant"].sum() if "significant" in result.columns else 0
            logger.info(
                "%s %s-%s enrichment: %d pathways tested, %d significant (FDR < %.2f)",
                sample_id, metric, tier, len(result), n_sig, fdr_threshold,
            )

    # Consolidate all results into a single pathways.tsv file
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        # Reorder columns: metric and tier first
        cols = ["metric", "tier"] + [c for c in combined.columns if c not in ["metric", "tier"]]
        combined = combined[cols]

        pathways_path = enrich_dir / f"{sample_id}_pathways.tsv"
        combined.to_csv(pathways_path, sep="\t", index=False)
        outputs["pathways"] = pathways_path
        logger.info("Consolidated enrichment results to %s (%d pathway-tier combinations)",
                    pathways_path, len(combined))

        # Write per-metric-per-tier slices so downstream consumers load the
        # correct subset rather than the union. Previously every
        # ``enrichment_{metric}_{tier}`` key aliased to the same combined
        # file, which made every tier and every metric render identically.
        for metric in test_metrics:
            for tier in ["high", "low"]:
                slice_df = combined[
                    (combined["metric"] == metric) & (combined["tier"] == tier)
                ].reset_index(drop=True)
                if slice_df.empty:
                    continue
                slice_path = enrich_dir / f"{sample_id}_pathways_{metric}_{tier}.tsv"
                slice_df.to_csv(slice_path, sep="\t", index=False)
                outputs[f"enrichment_{metric}_{tier}"] = slice_path

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
            ko_cols = [gene_col, ko_col]
            if "KO_definition" in kofam_df.columns:
                ko_cols.append("KO_definition")
            ko_info = kofam_df[ko_cols].copy()
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
