"""Advanced codon usage analyses: COA, RSCU distance, neutrality, PR2, delta RSCU,
tRNA-codon correlation, COG enrichment, gene length vs bias, and ENC-ENC' difference.

All analyses are computed from data already available in the pipeline
(per-gene RSCU, ENC, ENCprime, expression tiers, GFF annotations, COG assignments).
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS,
    CODON_TABLE_11,
    RSCU_COLUMN_NAMES,
    RSCU_COL_TO_CODON,
    SENSE_CODONS,
    dna_to_rna,
    COL_EXPRESSION_CLASS,
    COL_ENC_DIFF,
    COL_ENCPRIME,
    COL_ENC,
    COL_GENE,
    COL_GC3,
    COL_CAI_CLASS,
    COL_MELP_CLASS,
    COL_FOP_CLASS,
)
from codonpipe.utils.io import find_gene_id_column, get_output_subdir
from codonpipe.utils.statistics import benjamini_hochberg

logger = logging.getLogger("codonpipe")


# Generous upper bound on the number of COA axes emitted for downstream
# Mahalanobis axis selection. The selector (mahal_clustering._select_n_axes)
# trims this to the variance-retained subset; emitting more than the historical
# hard cap of 4 ensures a real-but-minor selection axis (e.g. a translational-
# selection axis displaced to axis 3-5 in a high-GC genome) is not discarded
# before the selector ever sees it.
_COA_AXIS_EMIT_CAP = 15


def _broken_stick_pct(p: int) -> np.ndarray:
    """Expected % inertia per ranked axis under the broken-stick null model.

    Under the null that total inertia is partitioned at random (a stick of
    unit length broken at p-1 uniformly random points), the expected
    proportion held by the k-th largest piece is (1/p) * sum_{i=k}^{p} (1/i).
    Observed axes whose inertia exceeds this null are "real"; the first axis
    that falls below it marks the noise floor. Standard scree alternative
    (Frontier 1976; Jackson 1993; Legendre & Legendre 2012).
    """
    if p <= 0:
        return np.array([])
    inv = 1.0 / np.arange(1, p + 1)
    bs = np.array([inv[k:].sum() for k in range(p)])
    return bs / p * 100.0


# ─── Correspondence Analysis on RSCU ────────────────────────────────────────


def compute_coa_on_rscu(
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Correspondence Analysis on the per-gene RSCU matrix.

    Projects genes into a low-dimensional space where the major axes
    capture the dominant sources of codon usage variation. In most
    bacterial genomes, axis 1 correlates with expression level.

    Args:
        rscu_gene_df: Per-gene RSCU table (from compute_rscu_per_gene).
        expr_df: Optional expression table with gene and *_class columns.

    Returns:
        Dict with:
            - 'coa_coords': DataFrame (gene, Axis1..AxisN, plus expression tier
              columns if expr_df given). N is variance-retained (up to
              _COA_AXIS_EMIT_CAP), not a fixed 4; the downstream Mahalanobis
              step selects the working subset from these.
            - 'coa_codon_coords': DataFrame (codon, Axis1, Axis2)
            - 'coa_inertia': DataFrame (axis, eigenvalue, pct_inertia, cum_pct,
              broken_stick_pct)
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if len(rscu_cols) < 4 or len(rscu_gene_df) < 10:
        logger.warning("Too few genes/codons for COA (%d genes, %d codons)",
                       len(rscu_gene_df), len(rscu_cols))
        return {}

    X = rscu_gene_df[rscu_cols].values.copy()
    genes = rscu_gene_df["gene"].values

    # Replace NaN with 0 (genuine absence); zeros are meaningful RSCU values
    X = np.where(np.isnan(X), 0.0, X)

    # Drop any rows whose total row mass is so small that they would receive
    # an outsized 1/sqrt(row_mass) scaling in the row-coordinate computation
    # below. With epsilon=1e-8 in the denominator, a row sum below ~1e-3
    # would inflate that gene's coordinates by orders of magnitude relative
    # to genuine signals. count_codons + MIN_GENE_LENGTH normally prevents
    # this, but defensive filtering is cheap and stops a single malformed
    # gene from dominating the projection.
    row_total = X.sum(axis=1)
    valid_row = row_total > 1e-3
    n_dropped = int((~valid_row).sum())
    if n_dropped:
        logger.warning(
            "COA: dropped %d gene(s) with near-zero RSCU row sums "
            "(would otherwise dominate the row-coordinate projection)",
            n_dropped,
        )
        X = X[valid_row]
        genes = genes[valid_row]
        if len(X) < 10:
            logger.warning(
                "COA: only %d genes remain after row-mass filter; skipping COA", len(X),
            )
            return {}

    # For chi-squared standardization, add small constant to avoid zero row/column sums
    epsilon = 1e-8

    # Correspondence analysis via SVD on standardized residuals
    row_sums = X.sum(axis=1, keepdims=True) + epsilon
    col_sums = X.sum(axis=0, keepdims=True) + epsilon
    grand_total = X.sum() + epsilon

    # Expected frequencies
    E = (row_sums @ col_sums) / grand_total

    # Standardized residuals (chi-squared metric)
    row_masses = row_sums.ravel() / grand_total
    col_masses = col_sums.ravel() / grand_total

    # Residual matrix
    S = (X / grand_total - E / grand_total) / np.sqrt(E / grand_total)

    # SVD on the full standardized-residual matrix.
    U_full, sigma_full, Vt_full = np.linalg.svd(S, full_matrices=False)
    all_sigma = sigma_full.copy()
    rank = len(all_sigma)
    if rank < 2:
        return {}
    total_inertia = float(np.sum(all_sigma ** 2))

    # Variance-retained axis emission (replaces the old hard cap of 4). We emit
    # a generous, bounded set of axes here and let the downstream Mahalanobis
    # step (mahal_clustering._select_n_axes) pick the working dimensionality by
    # a 90%-inertia + broken-stick rule. Emitting up to _COA_AXIS_EMIT_CAP
    # ensures a genuine but minor codon-usage axis is available to that selector
    # instead of being truncated away a priori. Inertia percentages are computed
    # against the FULL spectrum (total_inertia), so they remain correct
    # regardless of how many axes are emitted.
    full_pct = (all_sigma ** 2) / total_inertia * 100 if total_inertia > 0 else np.zeros(rank)
    bs_pct = _broken_stick_pct(rank)
    n_axes = max(2, min(_COA_AXIS_EMIT_CAP, rank, min(S.shape) - 1))

    U = U_full[:, :n_axes]
    sigma = sigma_full[:n_axes]
    V = Vt_full[:n_axes, :].T

    eigenvalues = sigma ** 2
    pct_inertia = full_pct[:n_axes]

    # Row (gene) coordinates: scale by sqrt(row_masses).
    # SCALING-CONVENTION NOTE: this 1/sqrt(mass) rescaling is applied on top of
    # an S matrix already standardized as (P - E)/sqrt(E), so the resulting gene
    # coordinates are NOT identical to canonical CA principal coordinates from
    # vegan / ade4 / `ca` (absolute positions are stretched per-gene by
    # ~1/sqrt(row_mass)). Axis ORDERING and the inertia percentages above are
    # unaffected, and because RSCU row sums are near-constant across genes the
    # distortion is mild. CodonPipe uses these coordinates only as an internal,
    # self-consistent space for Mahalanobis clustering — do not compare the raw
    # coordinate values to external CA tools. Left as-is intentionally: changing
    # the scaling would shift every downstream cluster result (and any published
    # numbers derived from it), so it is flagged for a deliberate, versioned
    # migration rather than changed silently here.
    row_coords = np.diag(1.0 / np.sqrt(row_masses + 1e-30)) @ U * sigma[np.newaxis, :]
    # Column (codon) coordinates
    col_coords = np.diag(1.0 / np.sqrt(col_masses + 1e-30)) @ V * sigma[np.newaxis, :]

    # Build gene coordinate DataFrame
    axis_names = [f"Axis{i+1}" for i in range(n_axes)]
    coa_df = pd.DataFrame(row_coords, columns=axis_names)
    coa_df.insert(0, "gene", genes)

    # Merge expression tiers if available
    if expr_df is not None and COL_GENE in expr_df.columns:
        class_cols = [c for c in expr_df.columns if c.endswith("_class") and c != COL_EXPRESSION_CLASS]
        if class_cols:
            merge_cols = [COL_GENE] + class_cols
            available = [c for c in merge_cols if c in expr_df.columns]
            coa_df = coa_df.merge(expr_df[available], on=COL_GENE, how="left")

    # Codon coordinates
    codon_df = pd.DataFrame(col_coords[:, :2], columns=["Axis1", "Axis2"])
    codon_df.insert(0, "codon", rscu_cols)

    # Inertia summary
    inertia_df = pd.DataFrame({
        "axis": list(range(1, n_axes + 1)),
        "eigenvalue": eigenvalues,
        "pct_inertia": pct_inertia,
        "cum_pct": np.cumsum(pct_inertia),
        # Broken-stick null expectation per axis; the downstream selector keeps
        # only leading axes whose pct_inertia exceeds this.
        "broken_stick_pct": bs_pct[:n_axes],
    })

    return {
        "coa_coords": coa_df,
        "coa_codon_coords": codon_df,
        "coa_inertia": inertia_df,
    }


# ─── RSCU distance to reference profile ─────────────────────────────────────
#
# Previously called "S-value" in earlier versions of this pipeline. Renamed
# to avoid confusion with two unrelated published quantities:
#   1. The Sharp & Li (1987) S index (a CAI-style geometric mean of
#      relative adaptiveness weights).
#   2. The Carbone et al. (2003) S value, which is a chi-squared distance
#      *projected onto the first COA axis*. This module computes the raw
#      distance to a reference vector — not the COA-axis projection — so
#      it is NOT the Carbone S value either.
#
# What this function actually computes is a per-gene distance between the
# gene's RSCU vector and a fixed reference RSCU profile. Three metrics are
# supported: euclidean (default for backward compatibility), chi_squared
# (asymmetric: divides by the reference), and aitchison (Euclidean on
# CLR-transformed RSCU — the metric that respects the compositional
# constraints baked into RSCU). Aitchison is the principled choice for
# compositional data; use it when the result feeds into downstream
# distance-based statistics.


def compute_rscu_distance(
    rscu_gene_df: pd.DataFrame,
    rscu_rp: dict[str, float] | None,
    metric: str = "euclidean",
    rscu_ace: dict[str, float] | None = None,
    rscu_mahal_cluster: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-gene RSCU distance to a reference codon usage profile.

    Reference priority: Mahalanobis cluster > ACE consensus > ribosomal proteins.

    The Mahalanobis optimal cluster captures the codon usage of translationally
    optimised genes identified by data-driven clustering, making it the
    most biologically grounded reference. ACE consensus is genome-specific
    and composition-independent. Ribosomal proteins are the traditional
    fallback.

    Genes with low RSCU distance have codon usage similar to the reference
    (i.e. adapted toward translational optimization).

    Args:
        rscu_gene_df: Per-gene RSCU table.
        rscu_rp: Concatenated RSCU for ribosomal proteins (fallback reference).
        metric: 'euclidean' (default), 'chi_squared', or 'aitchison'.
            Aitchison is the compositional-data-correct choice (Euclidean
            on CLR-transformed RSCU, Aitchison 1986); use it whenever the
            output feeds into a downstream distance-based test. Euclidean
            and chi_squared treat each codon coordinate as independent,
            which is wrong in principle for RSCU (per-AA-family sum
            constraints) but is fine for ranking genes by similarity to
            a single reference.
        rscu_ace: ACE consensus RSCU dict.
        rscu_mahal_cluster: Mahalanobis optimal cluster RSCU dict (preferred reference).

    Returns:
        DataFrame with gene, RSCU_distance, RSCU_distance_reference,
        RSCU_distance_metric columns.
    """
    # Priority: Mahalanobis cluster > ACE consensus > RP
    if rscu_mahal_cluster is not None:
        ref = rscu_mahal_cluster
        ref_label = "mahal_cluster"
    elif rscu_ace is not None:
        ref = rscu_ace
        ref_label = "ace"
    else:
        ref = rscu_rp
        ref_label = "rp"

    empty_cols = [
        COL_GENE,
        "RSCU_distance",
        "RSCU_distance_reference",
        "RSCU_distance_metric",
    ]
    if ref is None:
        return pd.DataFrame(columns=empty_cols)

    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns and c in ref]
    if not rscu_cols:
        return pd.DataFrame(columns=empty_cols)

    ref_vec = np.array([ref[c] for c in rscu_cols], dtype=float)
    gene_mat = rscu_gene_df[rscu_cols].values.astype(float)

    metric_norm = (metric or "euclidean").lower()
    if metric_norm == "chi_squared":
        denom = np.where(ref_vec == 0, 1e-10, ref_vec)
        dists = np.sqrt(np.sum((gene_mat - ref_vec[np.newaxis, :]) ** 2 / denom[np.newaxis, :], axis=1))
    elif metric_norm == "aitchison":
        # CLR transform: log(x + ε) − mean(log(x + ε)) per row, then
        # Euclidean on the centered log-ratio vectors. ε floors zeros so
        # log(0) does not blow up. This is the textbook Aitchison distance
        # for compositional data and is the right choice when the RSCU
        # row sums to a constant per-AA-family (which it does).
        eps = 1e-6
        log_gene = np.log(np.clip(gene_mat, eps, None))
        clr_gene = log_gene - log_gene.mean(axis=1, keepdims=True)
        log_ref = np.log(np.clip(ref_vec, eps, None))
        clr_ref = log_ref - log_ref.mean()
        diff = clr_gene - clr_ref[np.newaxis, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
    else:
        # Plain Euclidean. Default for backward compatibility.
        dists = np.sqrt(np.sum((gene_mat - ref_vec[np.newaxis, :]) ** 2, axis=1))

    return pd.DataFrame({
        COL_GENE: rscu_gene_df[COL_GENE].values,
        "RSCU_distance": dists,
        "RSCU_distance_reference": ref_label,
        "RSCU_distance_metric": metric_norm,
    })


# Backward-compatibility alias for external code that imported the old name.
# The "S-value" name is misleading (this is NOT the Sharp & Li S index nor
# the Carbone et al. 2003 S value); kept solely so that downstream callers
# don't break. Prefer ``compute_rscu_distance`` in new code.
compute_s_value = compute_rscu_distance


# ─── ENC - ENC' difference ──────────────────────────────────────────────────


def compute_enc_diff(
    enc_df: pd.DataFrame,
    encprime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ENC - ENC' for each gene.

    Genes where ENC << ENC' are under codon selection pressure beyond
    what background nucleotide composition predicts.

    Args:
        enc_df: ENC table with gene, ENC, GC3.
        encprime_df: ENCprime table with gene and a score column.

    Returns:
        DataFrame with gene, ENC, ENCprime, ENC_diff, GC3.
    """
    if enc_df.empty or encprime_df.empty:
        return pd.DataFrame(columns=[COL_GENE, COL_ENC, COL_ENCPRIME, COL_ENC_DIFF, COL_GC3])

    # Identify the ENCprime score column
    score_candidates = [c for c in encprime_df.columns if c not in ("gene", "width")]
    if not score_candidates:
        logger.warning("ENCprime DataFrame has no score column; skipping ENC diff")
        return pd.DataFrame(columns=[COL_GENE, COL_ENC, COL_ENCPRIME, COL_ENC_DIFF, COL_GC3])
    score_col = score_candidates[0]

    merged = enc_df[[COL_GENE, COL_ENC, COL_GC3]].merge(
        encprime_df[[COL_GENE, score_col]].rename(columns={score_col: COL_ENCPRIME}),
        on=COL_GENE,
        how="inner",
    )
    merged[COL_ENC_DIFF] = merged[COL_ENC] - merged[COL_ENCPRIME]
    return merged


# ─── Neutrality plot (GC12 vs GC3) ──────────────────────────────────────────


def compute_gc12_gc3(ffn_path: Path, min_length: int = 240) -> pd.DataFrame:
    """Compute per-gene GC content at 1st+2nd vs 3rd codon positions.

    The Sueoka (1988) neutrality test: GC12 vs GC3 regression slope
    indicates relative contribution of mutation (slope ~1) vs selection
    (slope ~0).

    Returns:
        DataFrame with gene, GC1, GC2, GC12, GC3, length.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq).upper()
        if len(seq) < min_length:
            continue

        gc1 = gc2 = gc3 = total1 = total2 = total3 = 0
        for i in range(0, len(seq) - 2, 3):
            for pos_offset, pos_label in [(0, "1"), (1, "2"), (2, "3")]:
                base = seq[i + pos_offset]
                if base in "ACGT":
                    if pos_label == "1":
                        total1 += 1
                        if base in "GC":
                            gc1 += 1
                    elif pos_label == "2":
                        total2 += 1
                        if base in "GC":
                            gc2 += 1
                    else:
                        total3 += 1
                        if base in "GC":
                            gc3 += 1

        if total1 > 0 and total2 > 0 and total3 > 0:
            gc1_frac = gc1 / total1
            gc2_frac = gc2 / total2
            gc12_frac = (gc1 + gc2) / (total1 + total2)
            gc3_frac = gc3 / total3
            rows.append({
                COL_GENE: rec.id,
                "GC1": gc1_frac,
                "GC2": gc2_frac,
                "GC12": gc12_frac,
                COL_GC3: gc3_frac,
                "length": len(seq),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[COL_GENE, "GC1", "GC2", "GC12", COL_GC3, "length"]
    )


# ─── PR2 (Parity Rule 2) plot ───────────────────────────────────────────────


# Four-fold degenerate codon prefixes per Sueoka (1995). Only these sites
# leave the encoded amino acid invariant under any third-position substitution,
# so the PR2 bias statistic is a clean readout of mutational/replicative
# asymmetry. Any 2-fold or 3-fold site mixes selection on the amino acid into
# the third-position composition.
_FOURFOLD_PREFIXES = frozenset({
    "GC",  # Ala: GCU/GCC/GCA/GCG
    "GG",  # Gly: GGU/GGC/GGA/GGG
    "CC",  # Pro: CCU/CCC/CCA/CCG
    "AC",  # Thr: ACU/ACC/ACA/ACG
    "GU",  # Val: GUU/GUC/GUA/GUG
    "UC",  # Ser4: UCU/UCC/UCA/UCG (note: AGU/AGC are 2-fold, excluded)
    "CU",  # Leu4: CUU/CUC/CUA/CUG (note: UUA/UUG are 2-fold, excluded)
    "CG",  # Arg4: CGU/CGC/CGA/CGG (note: AGA/AGG are 2-fold, excluded)
})


def compute_pr2(ffn_path: Path, min_length: int = 240) -> pd.DataFrame:
    """Compute PR2 bias statistics at four-fold degenerate sites.

    PR2 (Sueoka 1995) plots A3/(A3+T3) vs G3/(G3+C3) using only third-position
    bases of four-fold degenerate codons (Ala, Gly, Pro, Thr, Val, Ser4, Leu4,
    Arg4). Restricting to 4-fold sites makes the third position effectively
    selection-free with respect to the amino acid, so any deviation from 0.5
    is read as replication-associated mutational asymmetry rather than
    translational selection.

    Earlier versions of this function counted *every* third-position base,
    which mixed in selection effects from 2-fold (Phe, Tyr, His, Gln, Asn,
    Lys, Asp, Glu, Cys, Ser2, Leu2, Arg2) and 3-fold (Ile) sites where
    third-position changes can be non-synonymous.

    Returns:
        DataFrame with gene, A3_ratio (A3/(A3+T3)), G3_ratio (G3/(G3+C3)),
        length, n_fourfold_codons.
    """
    rows = []
    for rec in SeqIO.parse(str(ffn_path), "fasta"):
        seq = str(rec.seq).upper().replace("T", "U")
        if len(seq) < min_length:
            continue

        base3_counts = Counter()
        n_fourfold = 0
        # Walk codons (frame 0). Only count the 3rd base when the first two
        # bases identify a four-fold degenerate family.
        for i in range(0, len(seq) - 2, 3):
            prefix = seq[i:i + 2]
            if prefix not in _FOURFOLD_PREFIXES:
                continue
            base = seq[i + 2]
            # Use DNA letters in the output to match the rest of the module's
            # conventions (count_codons returns RNA, but PR2 reports A/T/G/C).
            if base == "U":
                base = "T"
            if base in "ACGT":
                base3_counts[base] += 1
                n_fourfold += 1

        a3 = base3_counts.get("A", 0)
        t3 = base3_counts.get("T", 0)
        g3 = base3_counts.get("G", 0)
        c3 = base3_counts.get("C", 0)

        at_total = a3 + t3
        gc_total = g3 + c3

        # Need a meaningful number of 4-fold codons before the ratios are
        # interpretable. ~30 fourfold codons gives a reasonable SE on each
        # ratio for typical eubacterial gene lengths.
        if at_total > 0 and gc_total > 0 and n_fourfold >= 30:
            rows.append({
                COL_GENE: rec.id,
                "A3_ratio": a3 / at_total,
                "G3_ratio": g3 / gc_total,
                "length": len(seq),
                "n_fourfold_codons": n_fourfold,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=[COL_GENE, "A3_ratio", "G3_ratio", "length", "n_fourfold_codons"]
    )


# ─── Delta RSCU between expression tiers ─────────────────────────────────────


def compute_delta_rscu(
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    class_col: str = COL_EXPRESSION_CLASS,
    rscu_reference: dict[str, float] | None = None,
    reference_label: str = "genome_avg",
    test: bool = True,
) -> pd.DataFrame:
    """Per-codon delta-RSCU (high-expression vs reference) plus per-codon test.

    The raw delta is `mean(RSCU_high) - reference_RSCU`. RSCU is compositional
    within each AA family (per-family sums to the family size), so the raw
    deltas are not statistically independent across codons within a family;
    interpret them as a *ranking* of codon shifts, not as effect sizes that
    are comparable across families.

    To make the output statistically interpretable rather than purely
    descriptive, a per-codon Mann-Whitney U test of `RSCU(high)` vs
    `RSCU(rest)` is added when ``test=True``, with a global Benjamini-
    Hochberg FDR correction across all codons (the multiple-testing
    family is "any codon shifted in high-expression genes"). A CLR
    pretreatment of RSCU is also computed and included as
    ``delta_clr`` so callers who want a compositionally-correct
    "shift" magnitude have it without re-deriving CLR themselves.

    Args:
        rscu_gene_df: Per-gene RSCU table.
        expr_df: Expression table with gene and *_class columns.
        class_col: Which classification column to use.
        rscu_reference: Optional dict of reference RSCU values (e.g.
            Mahalanobis cluster RSCU).  When *None* the genome-wide
            average is used as the baseline.
        reference_label: Label used in the ``ref_rscu`` output column
            name (e.g. ``"genome_avg"`` or ``"mahal_cluster"``).
        test: If True (default), add per-codon Mann-Whitney U test
            (high vs rest) with global BH FDR. Set False for a purely
            descriptive table.

    Returns:
        DataFrame with codon, amino_acid, {reference_label}_rscu,
        high_expr_rscu, delta_rscu, delta_clr (CLR-space shift), and —
        when ``test=True`` — ``mw_p``, ``corrected_p`` (BH FDR over all
        codons), ``significant`` (BH < 0.05).
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if not rscu_cols:
        logger.warning("Delta RSCU: no RSCU columns found in gene DataFrame")
        return pd.DataFrame()
    if class_col not in expr_df.columns:
        logger.warning("Delta RSCU: classification column '%s' not found in expression data "
                       "(available: %s)", class_col, list(expr_df.columns))
        return pd.DataFrame()

    # Merge RSCU with expression tiers
    merged = rscu_gene_df.merge(
        expr_df[[COL_GENE, class_col]], on=COL_GENE, how="inner"
    )

    if rscu_reference is not None:
        ref_vals = pd.Series({c: rscu_reference.get(c, float("nan")) for c in rscu_cols})
    else:
        ref_vals = merged[rscu_cols].mean()

    high_mask = merged[class_col] == "high"
    rest_mask = ~high_mask

    if high_mask.sum() < 3:
        logger.warning("Fewer than 3 high-expression genes for delta RSCU")
        return pd.DataFrame()

    high_avg = merged.loc[high_mask, rscu_cols].mean()

    # CLR-space shift, computed on per-AA-family proportions. CLR removes
    # the per-family sum constraint so the "shift" is interpretable
    # cross-family. ε floors zeros for log stability.
    eps = 1e-6
    high_log = np.log(np.clip(high_avg.values, eps, None))
    high_clr = high_log - high_log.mean()
    ref_log = np.log(np.clip(ref_vals.values, eps, None))
    ref_clr = ref_log - ref_log.mean()
    delta_clr_vec = high_clr - ref_clr

    if test:
        try:
            from scipy.stats import mannwhitneyu
        except ImportError:
            mannwhitneyu = None
    else:
        mannwhitneyu = None

    rows = []
    for i, col in enumerate(rscu_cols):
        parts = col.split("-")
        aa = parts[0]
        codon = parts[-1]
        row = {
            "codon_col": col,
            "codon": codon,
            "amino_acid": aa,
            f"{reference_label}_rscu": round(float(ref_vals[col]), 4),
            "high_expr_rscu": round(float(high_avg[col]), 4),
            "delta_rscu": round(float(high_avg[col] - ref_vals[col]), 4),
            "delta_clr": round(float(delta_clr_vec[i]), 4),
        }
        if mannwhitneyu is not None and rest_mask.sum() >= 3:
            x = merged.loc[high_mask, col].dropna().values
            y = merged.loc[rest_mask, col].dropna().values
            if len(x) >= 3 and len(y) >= 3 and (np.std(x) > 0 or np.std(y) > 0):
                try:
                    _, p = mannwhitneyu(x, y, alternative="two-sided")
                    row["mw_p"] = float(p)
                except ValueError:
                    row["mw_p"] = float("nan")
            else:
                row["mw_p"] = float("nan")
        rows.append(row)

    out = pd.DataFrame(rows)

    # Global BH FDR across all codons in the panel. The multiple-testing
    # family is the whole codon table, not per-AA-family, because the
    # downstream interpretation is "any codon shifted under selection".
    if "mw_p" in out.columns and out["mw_p"].notna().any():
        pvals = out["mw_p"].fillna(1.0).values
        out["corrected_p"] = benjamini_hochberg(pvals)
        out["significant"] = out["corrected_p"] < 0.05

    return out


# ─── tRNA gene count vs codon frequency ──────────────────────────────────────
#
# The wobble decoding model below follows dos Reis, Wernisch & Savva (2003)
# / dos Reis, Savva & Wernisch (2004) for the bacterial tRNA-codon mapping.
# Position 34 of the anticodon (5'-most, pairs with codon position 3) governs
# which codons a single tRNA can read:
#   * U34 reads codon3 = A (Watson-Crick) and G (wobble, weight 0.561)
#   * C34 reads codon3 = G only (Watson-Crick, weight 1.0)
#   * G34 reads codon3 = C (Watson-Crick) and U (wobble, weight 0.561)
#   * A34 is universally modified to inosine (I34) in bacterial tRNAs that
#     have an A there; I34 reads U (1.0), C (0.72), and A (0.32)
# Restricting to direct reverse-complement (the previous behaviour) ignored
# wobble entirely, under-counting effective tRNA pool for ~half of all codons.

# 5'-most anticodon base → list of (codon-3rd-base-RNA, weight) pairs.
# Weights are dos Reis 2004's published tAI s-values.
_WOBBLE_DECODING: dict[str, list[tuple[str, float]]] = {
    "U": [("A", 1.0), ("G", 0.561)],
    "C": [("G", 1.0)],
    "G": [("C", 1.0), ("U", 0.561)],
    "A": [("U", 0.9999), ("C", 0.72), ("A", 0.32)],  # I34 (inosine) decoding
}


def _decoded_codons_with_weights(anticodon_dna: str) -> list[tuple[str, float]]:
    """Return list of (decoded_codon_RNA, wobble_weight) for an anticodon.

    The anticodon is read 5'→3' (as stored in tRNA gene annotations). The
    codon read by base-pairing has positions 1, 2, 3 = anticodon positions
    3, 2, 1 reverse-complemented. Position 1 of the codon (=anticodon
    position 3) and position 2 of the codon (=anticodon position 2) follow
    Watson-Crick pairing strictly; only position 3 of the codon (=anticodon
    position 1, "the wobble position") can wobble.

    Args:
        anticodon_dna: 3-letter DNA anticodon (5' to 3').

    Returns:
        List of (codon_rna, weight) tuples. Empty if anticodon is invalid.
    """
    ac = anticodon_dna.upper().replace("T", "U")
    if len(ac) != 3 or any(b not in "ACGU" for b in ac):
        return []

    complement = {"A": "U", "U": "A", "G": "C", "C": "G"}
    # Codon positions 1 and 2 are Watson-Crick complements of anticodon
    # positions 3 and 2 respectively.
    codon_pos1 = complement[ac[2]]
    codon_pos2 = complement[ac[1]]
    # Codon position 3 wobbles against anticodon position 1 (5' base).
    wobble_pairs = _WOBBLE_DECODING.get(ac[0], [])
    return [
        (codon_pos1 + codon_pos2 + base3, weight)
        for base3, weight in wobble_pairs
    ]


def extract_trna_counts_from_gff(gff_path: Path) -> pd.DataFrame:
    """Extract tRNA gene counts and wobble-decoded codon assignments from a GFF3 file.

    Parses tRNA features and expands each anticodon into the set of codons it
    decodes via Watson-Crick + wobble pairing (dos Reis et al. 2004). The
    output is in *long* format: one row per (anticodon, decoded codon) pair,
    so a single U34-bearing tRNA contributes two rows (the Watson-Crick
    codon at weight 1.0 and the G3 wobble codon at weight 0.561).

    Inosine handling: A34 is treated as I34 (universal in bacterial 4-fold
    decoding tRNAs), so anticodons starting with A decode three codons.

    Args:
        gff_path: Path to GFF3 annotation file.

    Returns:
        DataFrame with anticodon, codon (RNA), amino_acid, tRNA_copy_number,
        wobble_weight, effective_tRNA (= tRNA_copy_number * wobble_weight).
    """
    anticodon_pattern = re.compile(r"(?:anticodon|product)=.*?([ACGT]{3})", re.IGNORECASE)
    product_aa_pattern = re.compile(r"product=tRNA-(\w{3})", re.IGNORECASE)

    trna_counts: Counter = Counter()
    trna_aa: dict[str, str] = {}

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            # GFF3 has ## directives and FASTA section
            if line.startswith(">"):
                break

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type != "tRNA":
                continue

            attrs = parts[8]

            # Try to extract anticodon
            ac_match = anticodon_pattern.search(attrs)
            if ac_match:
                anticodon_dna = ac_match.group(1).upper()
                trna_counts[anticodon_dna] += 1

                # Try to get amino acid from product field
                aa_match = product_aa_pattern.search(attrs)
                if aa_match:
                    trna_aa[anticodon_dna] = aa_match.group(1)

    if not trna_counts:
        # Fallback: try aragorn-style annotation where product=tRNA-Xxx(anticodon)
        trna_counts, trna_aa = _parse_trna_fallback(gff_path)

    if not trna_counts:
        return pd.DataFrame(columns=[
            "anticodon", "codon", "amino_acid", "tRNA_copy_number",
            "wobble_weight", "effective_tRNA",
        ])

    rows = []
    for anticodon_dna, count in trna_counts.items():
        decoded = _decoded_codons_with_weights(anticodon_dna)
        if not decoded:
            # Couldn't parse anticodon — keep a minimal record so the gene
            # isn't lost from downstream summaries.
            codon_rna = dna_to_rna(_reverse_complement(anticodon_dna))
            aa = trna_aa.get(anticodon_dna, CODON_TABLE_11.get(codon_rna, "?"))
            rows.append({
                "anticodon": anticodon_dna,
                "codon": codon_rna,
                "amino_acid": aa,
                "tRNA_copy_number": count,
                "wobble_weight": 1.0,
                "effective_tRNA": float(count),
            })
            continue
        for codon_rna, weight in decoded:
            aa = trna_aa.get(anticodon_dna, CODON_TABLE_11.get(codon_rna, "?"))
            rows.append({
                "anticodon": anticodon_dna,
                "codon": codon_rna,
                "amino_acid": aa,
                "tRNA_copy_number": count,
                "wobble_weight": weight,
                "effective_tRNA": count * weight,
            })

    return pd.DataFrame(rows)


def _parse_trna_fallback(gff_path: Path) -> tuple[Counter, dict[str, str]]:
    """Fallback tRNA parser for various annotation formats."""
    # Match patterns like tRNA-Ala(TGC) or tRNA-Ala(tgc)
    pattern = re.compile(r"tRNA-(\w{3,4})\(([ACGTacgt]{3})\)")
    counts: Counter = Counter()
    aa_map: dict[str, str] = {}

    with open(gff_path) as fh:
        for line in fh:
            if line.startswith(("#", ">")):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            if parts[2] not in ("tRNA", "gene"):
                continue
            m = pattern.search(parts[8])
            if m:
                aa_abbr = m.group(1)
                anticodon = m.group(2).upper()
                counts[anticodon] += 1
                aa_map[anticodon] = aa_abbr

    return counts, aa_map


def compute_trna_codon_correlation(
    trna_df: pd.DataFrame,
    rscu_gene_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
    class_col: str = COL_EXPRESSION_CLASS,
) -> pd.DataFrame:
    """Correlate tRNA gene copy number with codon usage across genes.

    Returns the per-codon table AND attaches a Spearman summary on
    ``df.attrs["trna_correlation_summary"]`` with keys per gene set
    (``all_genes``, ``high_expr``, ``low_expr``): each value is a dict of
    ``{rho, p_value, n_codons}``. The companion function
    :func:`summarize_trna_codon_correlation` extracts that summary into
    a tidy DataFrame for serialization (TSV does not preserve ``attrs``).

    A strong positive Spearman ρ on the high-expression subset is the
    classical evidence of tRNA-codon co-adaptation (dos Reis et al. 2004).
    Spearman is preferred over Pearson because tRNA copy numbers are
    highly skewed and the relationship with RSCU is monotonic but not
    necessarily linear.

    Args:
        trna_df: tRNA count table (from extract_trna_counts_from_gff).
        rscu_gene_df: Per-gene RSCU table.
        expr_df: Expression table for subsetting high-expression genes.
        class_col: Expression classification column (defaults to
            ``expression_class``, which is ACE-MELP when available).

    Returns:
        DataFrame with codon, amino_acid, tRNA_copy_number, rscu_all_genes,
        rscu_high_expr (if available), rscu_low_expr (if available).
        Spearman summary attached as ``.attrs["trna_correlation_summary"]``.
    """
    rscu_cols = [c for c in RSCU_COLUMN_NAMES if c in rscu_gene_df.columns]
    if trna_df.empty or not rscu_cols:
        return pd.DataFrame()

    # Aggregate effective tRNA copies per codon (sum across wobble-decoded
    # entries). When the wobble-aware extractor was used, each tRNA may
    # appear in trna_df once per decoded codon with effective_tRNA =
    # tRNA_copy_number * wobble_weight; summing those yields the effective
    # decoding capacity for each codon. For backward compatibility, fall
    # back to the raw copy-number sum if the new column isn't present.
    if "effective_tRNA" in trna_df.columns:
        per_codon = trna_df.groupby("codon", as_index=False)["effective_tRNA"].sum()
        trna_map = dict(zip(per_codon["codon"], per_codon["effective_tRNA"]))
    else:
        per_codon = trna_df.groupby("codon", as_index=False)["tRNA_copy_number"].sum()
        trna_map = dict(zip(per_codon["codon"], per_codon["tRNA_copy_number"]))

    # Genome-wide average RSCU per codon
    genome_avg = rscu_gene_df[rscu_cols].mean()

    # High/low expression averages
    high_avg = low_avg = None
    if expr_df is not None and class_col in expr_df.columns:
        merged = rscu_gene_df.merge(expr_df[["gene", class_col]], on="gene", how="inner")
        high_mask = merged[class_col] == "high"
        low_mask = merged[class_col] == "low"
        if high_mask.sum() >= 3:
            high_avg = merged.loc[high_mask, rscu_cols].mean()
        if low_mask.sum() >= 3:
            low_avg = merged.loc[low_mask, rscu_cols].mean()

    rows = []
    for col in rscu_cols:
        if col not in RSCU_COL_TO_CODON:
            continue
        codon = RSCU_COL_TO_CODON[col]
        aa = col.split("-")[0] if "-" in col else col
        trna_n = trna_map.get(codon, 0)

        row = {
            "codon_col": col,
            "codon": codon,
            "amino_acid": aa,
            "tRNA_copy_number": trna_n,
            "rscu_all_genes": round(genome_avg[col], 4),
        }
        if high_avg is not None:
            row["rscu_high_expr"] = round(high_avg[col], 4)
        if low_avg is not None:
            row["rscu_low_expr"] = round(low_avg[col], 4)
        rows.append(row)

    out = pd.DataFrame(rows)

    # Spearman correlation across codons. Restricted to multi-codon AA families
    # because Met/Trp tRNAs vs RSCU=1 are uninformative singletons.
    summary: dict[str, dict] = {}
    if not out.empty:
        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None
        if spearmanr is not None:
            multi = out[~out["amino_acid"].isin(("Met", "Trp"))]
            for label, col in (
                ("all_genes", "rscu_all_genes"),
                ("high_expr", "rscu_high_expr"),
                ("low_expr", "rscu_low_expr"),
            ):
                if col in multi.columns:
                    sub = multi[["tRNA_copy_number", col]].dropna()
                    if len(sub) >= 6 and sub["tRNA_copy_number"].std() > 0:
                        rho, p = spearmanr(sub["tRNA_copy_number"], sub[col])
                        summary[label] = {
                            "rho": float(rho) if rho == rho else float("nan"),
                            "p_value": float(p) if p == p else float("nan"),
                            "n_codons": int(len(sub)),
                        }
    if summary:
        out.attrs["trna_correlation_summary"] = summary

    return out


def summarize_trna_codon_correlation(
    trna_corr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract the Spearman summary stashed on ``trna_corr_df.attrs`` into a
    tidy DataFrame suitable for ``to_csv``. Returns one row per gene set
    (``all_genes``, ``high_expr``, ``low_expr``) with ``rho``, ``p_value``,
    and ``n_codons``. Returns an empty frame if the summary is missing
    (e.g. when scipy was unavailable or the input was empty).
    """
    summary = trna_corr_df.attrs.get("trna_correlation_summary", {}) if hasattr(trna_corr_df, "attrs") else {}
    if not summary:
        return pd.DataFrame(columns=["gene_set", "rho", "p_value", "n_codons"])
    rows = [
        {
            "gene_set": label,
            "rho": vals.get("rho"),
            "p_value": vals.get("p_value"),
            "n_codons": vals.get("n_codons"),
        }
        for label, vals in summary.items()
    ]
    return pd.DataFrame(rows)


# ─── COG category enrichment in expression tiers ────────────────────────────


def compute_cog_enrichment(
    cog_result_tsv: Path,
    expr_df: pd.DataFrame,
    class_col: str = COL_EXPRESSION_CLASS,
) -> pd.DataFrame:
    """Test COG functional category enrichment in expression tiers.

    Uses Fisher's exact test per COG functional category to ask whether
    that category is enriched among high-expression (or low-expression) genes.

    Args:
        cog_result_tsv: COGclassifier result.tsv.
        expr_df: Expression table with gene and *_class columns.
        class_col: Expression classification column to use (defaults to
            ``expression_class``, which is ACE-MELP when available).

    Returns:
        DataFrame with COG_category, description, tier, n_tier, n_background,
        odds_ratio, p_value, fdr columns.
    """
    if not cog_result_tsv.exists() or expr_df.empty or class_col not in expr_df.columns:
        return pd.DataFrame()

    cog_df = pd.read_csv(cog_result_tsv, sep="\t")

    # Find COG functional category column
    func_col = None
    for candidate in ["FUNCTIONAL_CATEGORY", "functional_category", "COG_category",
                       "cog_category", "LETTER", "letter"]:
        if candidate in cog_df.columns:
            func_col = candidate
            break
    # Fallback: look for single-letter columns
    if func_col is None:
        for col in cog_df.columns:
            vals = cog_df[col].dropna().astype(str)
            if vals.str.match(r"^[A-Z]$").mean() > 0.5:
                func_col = col
                break

    if func_col is None:
        logger.warning("Cannot find COG functional category column")
        return pd.DataFrame()

    # Find query/gene ID column
    query_col = find_gene_id_column(cog_df, fallback_to_first=True)

    # COG category descriptions
    cog_descriptions = {
        "J": "Translation, ribosomal structure and biogenesis",
        "A": "RNA processing and modification",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "B": "Chromatin structure and dynamics",
        "D": "Cell cycle control, cell division, chromosome partitioning",
        "Y": "Nuclear structure",
        "V": "Defense mechanisms",
        "T": "Signal transduction mechanisms",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "Z": "Cytoskeleton",
        "W": "Extracellular structures",
        "U": "Intracellular trafficking, secretion, and vesicular transport",
        "O": "Posttranslational modification, protein turnover, chaperones",
        "X": "Mobilome: prophages, transposons",
        "C": "Energy production and conversion",
        "G": "Carbohydrate transport and metabolism",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis, transport and catabolism",
        "R": "General function prediction only",
        "S": "Function unknown",
    }

    # Merge COG with expression
    # Some genes may have multi-letter categories; explode them
    cog_slim = cog_df[[query_col, func_col]].copy()
    cog_slim.columns = ["gene", "cog_cat"]
    cog_slim["cog_cat"] = cog_slim["cog_cat"].astype(str)

    # Explode multi-letter categories (e.g. "KL" -> ["K", "L"])
    cog_slim["cog_cat"] = cog_slim["cog_cat"].apply(
        lambda s: [ch for ch in s if ch.isalpha()]
    )
    cog_exploded = cog_slim.explode("cog_cat").dropna(subset=["cog_cat"])
    cog_exploded = cog_exploded[cog_exploded["cog_cat"].str.len() > 0]

    merged = cog_exploded.merge(expr_df[["gene", class_col]], on="gene", how="inner")

    # All genes with COG annotations
    all_categories = sorted(merged["cog_cat"].unique())
    total_genes = len(merged["gene"].unique())

    rows = []
    for tier in ["high", "low"]:
        tier_genes = set(merged.loc[merged[class_col] == tier, "gene"].unique())
        non_tier_genes = set(merged["gene"].unique()) - tier_genes

        if len(tier_genes) < 3:
            continue

        for cat in all_categories:
            cat_genes = set(merged.loc[merged["cog_cat"] == cat, "gene"].unique())
            # 2x2 table:
            # tier & cat, tier & !cat
            # !tier & cat, !tier & !cat
            a = len(cat_genes & tier_genes)
            b = len(tier_genes - cat_genes)
            c = len(cat_genes - tier_genes)
            d = len(non_tier_genes - cat_genes)

            if a + c == 0:
                continue

            odds_ratio, pval = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")

            rows.append({
                "COG_category": cat,
                "description": cog_descriptions.get(cat, ""),
                "tier": tier,
                "n_in_tier": a,
                "n_total": a + c,
                "pct_in_tier": round(100 * a / len(tier_genes), 1) if len(tier_genes) > 0 else 0,
                "pct_in_genome": round(100 * (a + c) / total_genes, 1) if total_genes > 0 else 0,
                "odds_ratio": round(odds_ratio, 3),
                "p_value": pval,
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # BH FDR correction
    n_tests = len(result)
    if n_tests > 0:
        result["fdr"] = benjamini_hochberg(result["p_value"].values)
        result["significant"] = result["fdr"] <= 0.05

    return result


# ─── Gene length vs codon bias ──────────────────────────────────────────────


def compute_gene_length_bias(
    enc_df: pd.DataFrame,
    expr_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge gene length with codon bias metrics. Returns the per-gene table.

    Naming kept for backward compatibility. The companion function
    :func:`fit_gene_length_bias` performs the actual regression of each
    bias metric on log10(length); call it on this output if you need
    slope, R², and a length-effect p-value.

    Args:
        enc_df: ENC table with gene, length, ENC, GC3.
        expr_df: Optional expression table to add CAI/MELP/Fop scores.

    Returns:
        DataFrame with gene, length, ENC, GC3, and optionally CAI/MELP/Fop.
        This is the *data table* — no regression is fit here.
    """
    if enc_df.empty:
        return pd.DataFrame()

    result = enc_df[[COL_GENE, "length", COL_ENC, COL_GC3]].copy()

    if expr_df is not None and COL_GENE in expr_df.columns:
        score_cols = [c for c in ["MELP", "CAI", "Fop"] if c in expr_df.columns]
        if score_cols:
            result = result.merge(
                expr_df[[COL_GENE] + score_cols], on=COL_GENE, how="left"
            )

    return result


def fit_gene_length_bias(
    length_bias_df: pd.DataFrame,
    metrics: tuple[str, ...] = (COL_ENC, COL_GC3, "MELP", "CAI", "Fop"),
    min_n: int = 30,
) -> pd.DataFrame:
    """Fit a Spearman correlation and an OLS regression of each bias metric
    on log10(gene length).

    Spearman is the headline statistic because the relationship between
    length and codon bias is monotonic but rarely linear (short genes
    have noisy RSCU which depresses CAI/Fop in a non-linear way). The
    OLS slope and R² on log10(length) are reported as a secondary, more
    interpretable summary; do not over-interpret the linearity.

    Multiple-testing correction is applied across the panel of metrics
    (BH FDR, q < 0.05).

    Args:
        length_bias_df: Output of :func:`compute_gene_length_bias`.
        metrics: Bias metrics to test. Skipped if absent or fewer than
            ``min_n`` non-null values.
        min_n: Minimum number of genes required to fit.

    Returns:
        DataFrame with one row per metric: ``metric``, ``n``,
        ``spearman_rho``, ``spearman_p``, ``ols_slope_log10_length``,
        ``ols_intercept``, ``ols_r_squared``, ``corrected_p`` (BH FDR
        on the Spearman p-values), ``significant`` (BH FDR < 0.05).
    """
    if length_bias_df.empty or "length" not in length_bias_df.columns:
        return pd.DataFrame()

    try:
        from scipy.stats import spearmanr, linregress
    except ImportError:
        logger.warning("scipy unavailable; cannot fit gene length bias")
        return pd.DataFrame()

    log_len = np.log10(length_bias_df["length"].astype(float).clip(lower=1.0))

    rows = []
    for metric in metrics:
        if metric not in length_bias_df.columns:
            continue
        sub = pd.DataFrame({
            "log_length": log_len,
            metric: length_bias_df[metric],
        }).dropna()
        if len(sub) < min_n:
            continue
        if sub["log_length"].std() == 0 or sub[metric].std() == 0:
            continue
        rho, p_spearman = spearmanr(sub["log_length"], sub[metric])
        lin = linregress(sub["log_length"].values, sub[metric].values)
        rows.append({
            "metric": metric,
            "n": int(len(sub)),
            "spearman_rho": float(rho) if rho == rho else float("nan"),
            "spearman_p": float(p_spearman) if p_spearman == p_spearman else float("nan"),
            "ols_slope_log10_length": float(lin.slope),
            "ols_intercept": float(lin.intercept),
            "ols_r_squared": float(lin.rvalue ** 2),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["corrected_p"] = benjamini_hochberg(out["spearman_p"].values)
    out["significant"] = out["corrected_p"] < 0.05
    return out


# ─── Helper functions ────────────────────────────────────────────────────────


def _reverse_complement(dna: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    comp = {"A": "T", "T": "A", "G": "C", "C": "G",
            "a": "t", "t": "a", "g": "c", "c": "g"}
    return "".join(comp.get(b, b) for b in reversed(dna))


# ─── Master runner ───────────────────────────────────────────────────────────


def run_advanced_analyses(
    ffn_path: Path,
    output_dir: Path,
    sample_id: str,
    rscu_gene_df: pd.DataFrame,
    enc_df: pd.DataFrame,
    rscu_rp: dict[str, float] | None = None,
    expr_df: pd.DataFrame | None = None,
    encprime_df: pd.DataFrame | None = None,
    gff_path: Path | None = None,
    cog_result_tsv: Path | None = None,
    rscu_ace: dict[str, float] | None = None,
    rscu_mahal_cluster: dict[str, float] | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Run all advanced codon usage analyses.

    Args:
        ffn_path: Path to CDS nucleotide FASTA.
        output_dir: Base output directory.
        sample_id: Sample identifier.
        rscu_gene_df: Per-gene RSCU table.
        enc_df: ENC table with gene, ENC, GC3.
        rscu_rp: Concatenated RSCU for ribosomal proteins.
        expr_df: Expression table with gene, scores, *_class columns.
        encprime_df: ENCprime table.
        gff_path: GFF3 annotation file (for tRNA extraction).
        cog_result_tsv: COGclassifier result.tsv (for COG enrichment).
        rscu_ace: ACE consensus RSCU dict.
        rscu_mahal_cluster: Mahalanobis optimal cluster RSCU dict. When provided,
            RSCU distance uses this as the reference (highest priority).

    Returns:
        Dict of output DataFrames and file paths.
    """
    adv_dir = get_output_subdir(output_dir, "comparative", "advanced")
    adv_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, pd.DataFrame | Path] = {}

    # 1. COA on RSCU
    logger.info("Computing Correspondence Analysis on RSCU for %s", sample_id)
    coa_results = compute_coa_on_rscu(rscu_gene_df, expr_df)
    if coa_results:
        for key, df in coa_results.items():
            out_path = adv_dir / f"{sample_id}_{key}.tsv"
            df.to_csv(out_path, sep="\t", index=False)
            outputs[key] = df
            outputs[f"{key}_path"] = out_path

    # 2. RSCU distance (reference priority: Mahalanobis cluster > ACE consensus > RP)
    if rscu_mahal_cluster is not None:
        ref_label = "Mahalanobis cluster"
    elif rscu_ace is not None:
        ref_label = "ACE consensus"
    else:
        ref_label = "ribosomal proteins"
    logger.info("Computing RSCU distance to %s for %s", ref_label, sample_id)
    s_val_df = compute_rscu_distance(rscu_gene_df, rscu_rp, rscu_ace=rscu_ace,
                                      rscu_mahal_cluster=rscu_mahal_cluster)
    if not s_val_df.empty:
        out_path = adv_dir / f"{sample_id}_rscu_distance.tsv"
        s_val_df.to_csv(out_path, sep="\t", index=False)
        outputs["s_value"] = s_val_df
        outputs["s_value_path"] = out_path

    # 3. ENC - ENC' difference
    if encprime_df is not None and not encprime_df.empty:
        logger.info("Computing ENC - ENC' difference for %s", sample_id)
        enc_diff_df = compute_enc_diff(enc_df, encprime_df)
        if not enc_diff_df.empty:
            out_path = adv_dir / f"{sample_id}_enc_diff.tsv"
            enc_diff_df.to_csv(out_path, sep="\t", index=False)
            outputs["enc_diff"] = enc_diff_df
            outputs["enc_diff_path"] = out_path

    # 4. Neutrality plot data (GC12 vs GC3)
    logger.info("Computing GC12 vs GC3 (neutrality) for %s", sample_id)
    gc12_gc3_df = compute_gc12_gc3(ffn_path)
    if not gc12_gc3_df.empty:
        out_path = adv_dir / f"{sample_id}_gc12_gc3.tsv"
        gc12_gc3_df.to_csv(out_path, sep="\t", index=False)
        outputs["gc12_gc3"] = gc12_gc3_df
        outputs["gc12_gc3_path"] = out_path

    # 5. PR2 data
    logger.info("Computing PR2 bias statistics for %s", sample_id)
    pr2_df = compute_pr2(ffn_path)
    if not pr2_df.empty:
        out_path = adv_dir / f"{sample_id}_pr2.tsv"
        pr2_df.to_csv(out_path, sep="\t", index=False)
        outputs["pr2"] = pr2_df
        outputs["pr2_path"] = out_path

    # 6. Delta RSCU (genome-average baseline). Step 9d of the pipeline
    # overwrites this file with Mahalanobis-based tiers and adds
    # mahal_cluster baseline rows when Mahalanobis clustering is enabled.
    # Schema: metric, baseline, codon_col, codon, amino_acid, ref_rscu,
    # high_expr_rscu, delta_rscu.
    if expr_df is not None:
        logger.info("Computing delta RSCU (high-expression vs genome avg) for %s", sample_id)
        delta_class_cols = [COL_EXPRESSION_CLASS,
                           COL_CAI_CLASS, COL_MELP_CLASS, COL_FOP_CLASS]
        all_deltas = []

        for class_col in delta_class_cols:
            if class_col in expr_df.columns:
                delta_df = compute_delta_rscu(rscu_gene_df, expr_df, class_col)
                if not delta_df.empty:
                    metric = class_col.replace("_class", "")
                    # Normalize {reference_label}_rscu -> ref_rscu and tag
                    delta_df = delta_df.rename(columns={"genome_avg_rscu": "ref_rscu"})
                    delta_df["metric"] = metric
                    delta_df["baseline"] = "genome_avg"
                    all_deltas.append(delta_df)

        if all_deltas:
            combined_deltas = pd.concat(all_deltas, ignore_index=True)
            # Reorder columns: metric, baseline first
            lead = ["metric", "baseline"]
            cols = lead + [c for c in combined_deltas.columns if c not in lead]
            combined_deltas = combined_deltas[cols]

            out_path = adv_dir / f"{sample_id}_rscu_deltas.tsv"
            combined_deltas.to_csv(out_path, sep="\t", index=False)
            outputs["rscu_deltas"] = combined_deltas
            outputs["rscu_deltas_path"] = out_path
            logger.info("Consolidated delta RSCU to %s", out_path)

            # Back-compat per-metric keys (genome-avg baseline only at this step)
            for class_col in delta_class_cols:
                if class_col in expr_df.columns:
                    metric = class_col.replace("_class", "")
                    mask = combined_deltas["metric"] == metric
                    if mask.any():
                        outputs[f"delta_rscu_{metric}"] = (
                            combined_deltas[mask].drop(columns=["metric", "baseline"])
                        )
                        outputs[f"delta_rscu_{metric}_path"] = out_path

    # 7. tRNA gene copy number correlation
    if gff_path is not None and gff_path.exists():
        logger.info("Extracting tRNA gene counts from GFF for %s", sample_id)
        trna_df = extract_trna_counts_from_gff(gff_path)
        if not trna_df.empty:
            trna_out = adv_dir / f"{sample_id}_trna_counts.tsv"
            trna_df.to_csv(trna_out, sep="\t", index=False)
            outputs["trna_counts"] = trna_df
            outputs["trna_counts_path"] = trna_out

            trna_corr_df = compute_trna_codon_correlation(
                trna_df, rscu_gene_df, expr_df,
            )
            if not trna_corr_df.empty:
                corr_out = adv_dir / f"{sample_id}_trna_codon_correlation.tsv"
                trna_corr_df.to_csv(corr_out, sep="\t", index=False)
                outputs["trna_codon_correlation"] = trna_corr_df
                outputs["trna_codon_correlation_path"] = corr_out

                # Spearman summary lives on .attrs and would be lost in
                # to_csv; emit a separate small TSV so the correlation
                # the README documents is actually persisted to disk.
                summary_df = summarize_trna_codon_correlation(trna_corr_df)
                if not summary_df.empty:
                    summary_out = (
                        adv_dir / f"{sample_id}_trna_codon_correlation_summary.tsv"
                    )
                    summary_df.to_csv(summary_out, sep="\t", index=False)
                    outputs["trna_codon_correlation_summary"] = summary_df
                    outputs["trna_codon_correlation_summary_path"] = summary_out
        else:
            logger.info("No tRNA features found in GFF for %s", sample_id)
    else:
        logger.info("No GFF file available for %s; skipping tRNA analysis", sample_id)

    # 8. COG category enrichment
    if cog_result_tsv is not None and cog_result_tsv.exists() and expr_df is not None:
        logger.info("Computing COG category enrichment by expression tier for %s", sample_id)
        cog_enrich = compute_cog_enrichment(cog_result_tsv, expr_df)
        if not cog_enrich.empty:
            out_path = adv_dir / f"{sample_id}_cog_enrichment.tsv"
            cog_enrich.to_csv(out_path, sep="\t", index=False)
            outputs["cog_enrichment"] = cog_enrich
            outputs["cog_enrichment_path"] = out_path

    # 9. Gene length vs codon bias — per-gene table plus regression fit
    logger.info("Computing gene length vs codon bias for %s", sample_id)
    length_bias_df = compute_gene_length_bias(enc_df, expr_df)
    if not length_bias_df.empty:
        out_path = adv_dir / f"{sample_id}_gene_length_bias.tsv"
        length_bias_df.to_csv(out_path, sep="\t", index=False)
        outputs["gene_length_bias"] = length_bias_df
        outputs["gene_length_bias_path"] = out_path

        # Spearman + OLS on log10(length) per metric, BH FDR across the panel.
        # The function call is what makes this an "analysis" rather than just
        # a data merge — without it, callers had only the merged columns and
        # had to fit themselves.
        fit_df = fit_gene_length_bias(length_bias_df)
        if not fit_df.empty:
            fit_path = adv_dir / f"{sample_id}_gene_length_bias_fit.tsv"
            fit_df.to_csv(fit_path, sep="\t", index=False)
            outputs["gene_length_bias_fit"] = fit_df
            outputs["gene_length_bias_fit_path"] = fit_path

    n_analyses = sum(1 for k in outputs if not k.endswith("_path"))
    logger.info("Advanced analyses complete for %s: %d datasets produced", sample_id, n_analyses)
    return outputs
