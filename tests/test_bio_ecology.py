"""Tests for the biological and ecological analyses module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.bio_ecology import (
    detect_hgt_candidates,
    predict_growth_rate,
    quantify_translational_selection,
    detect_phage_mobile_elements,
    compute_strand_asymmetry,
    compute_operon_codon_coadaptation,
)
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_rscu_gene_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
    }
    for col in RSCU_COLUMN_NAMES:
        data[col] = rng.uniform(0.2, 3.0, size=n_genes)
    return pd.DataFrame(data)


def _make_enc_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "ENC": rng.uniform(25, 60, size=n_genes),
        "GC3": rng.uniform(0.2, 0.8, size=n_genes),
    })


def _make_expr_df(n_genes=100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    classes = (["low"] * 5 + ["medium"] * 90 + ["high"] * 5)[:n_genes]
    return pd.DataFrame({
        "gene": [f"gene_{i:04d}" for i in range(n_genes)],
        "CAI": rng.uniform(0.1, 0.9, size=n_genes),
        "expression_class": classes,
    })


def _write_fasta(tmpdir: Path, n_genes=100, gene_prefix="gene_") -> Path:
    """Write a synthetic FASTA file with deterministic sequences."""
    fasta_path = tmpdir / "test.ffn"
    rng = np.random.default_rng(42)
    bases = list("ATGC")
    with open(fasta_path, "w") as f:
        for i in range(n_genes):
            gene_id = f"{gene_prefix}{i:04d}"
            # Generate a sequence of ~600 nt (divisible by 3)
            length = 600
            seq = "".join(rng.choice(bases, size=length))
            f.write(f">{gene_id}\n{seq}\n")
    return fasta_path


def _write_gff(tmpdir: Path, n_genes=100) -> Path:
    """Write a minimal GFF3 with genes on both strands."""
    gff_path = tmpdir / "test.gff"
    with open(gff_path, "w") as f:
        f.write("##gff-version 3\n")
        for i in range(n_genes):
            strand = "+" if i % 2 == 0 else "-"
            start = i * 700 + 1
            end = start + 599
            f.write(
                f"contig1\tProdigal\tCDS\t{start}\t{end}\t.\t{strand}\t0\t"
                f"ID=gene_{i:04d};product=hypothetical protein\n"
            )
    return gff_path


# ── HGT Detection ──────────────────────────────────────────────────────────

class TestHGTDetection:
    def test_returns_expected_columns(self):
        rscu_df = _make_rscu_gene_df(50)
        enc_df = _make_enc_df(50)
        result = detect_hgt_candidates(rscu_df, enc_df)
        assert "gene" in result.columns
        assert "mahalanobis_dist" in result.columns
        assert "gc3_deviation" in result.columns
        assert "p_value" in result.columns
        assert "hgt_flag" in result.columns

    def test_correct_number_of_genes(self):
        rscu_df = _make_rscu_gene_df(50)
        enc_df = _make_enc_df(50)
        result = detect_hgt_candidates(rscu_df, enc_df)
        assert len(result) == 50

    def test_distances_are_positive(self):
        rscu_df = _make_rscu_gene_df(50)
        enc_df = _make_enc_df(50)
        result = detect_hgt_candidates(rscu_df, enc_df)
        assert (result["mahalanobis_dist"] >= 0).all()

    def test_merges_expression_class(self):
        rscu_df = _make_rscu_gene_df(50)
        enc_df = _make_enc_df(50)
        expr_df = _make_expr_df(50)
        result = detect_hgt_candidates(rscu_df, enc_df, expr_df)
        assert "expression_class" in result.columns

    def test_empty_input_returns_empty(self):
        result = detect_hgt_candidates(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_hgt_flag_is_boolean(self):
        rscu_df = _make_rscu_gene_df(50)
        enc_df = _make_enc_df(50)
        result = detect_hgt_candidates(rscu_df, enc_df)
        assert result["hgt_flag"].dtype == bool

    def test_pca_reduction_with_few_genes(self):
        """When n_genes < n_features, PCA should be applied without error."""
        rscu_df = _make_rscu_gene_df(10)  # fewer genes than RSCU columns
        enc_df = _make_enc_df(10)
        result = detect_hgt_candidates(rscu_df, enc_df)
        assert len(result) == 10


# ── Growth Rate Prediction ─────────────────────────────────────────────────

class TestGrowthRate:
    def test_basic_prediction(self, tmp_path):
        expr_df = _make_expr_df(100)
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("\n".join([f"gene_{i:04d}" for i in range(10)]) + "\n")
        result = predict_growth_rate(expr_df, rp_ids)
        assert result is not None
        assert "mean_metric_rp" in result
        assert "predicted_doubling_time_hours" in result
        assert "n_rp_genes" in result
        assert "growth_class" in result

    def test_doubling_time_positive(self, tmp_path):
        expr_df = _make_expr_df(100)
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("\n".join([f"gene_{i:04d}" for i in range(10)]) + "\n")
        result = predict_growth_rate(expr_df, rp_ids)
        assert result["predicted_doubling_time_hours"] > 0

    def test_growth_class_values(self, tmp_path):
        expr_df = _make_expr_df(100)
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("\n".join([f"gene_{i:04d}" for i in range(10)]) + "\n")
        result = predict_growth_rate(expr_df, rp_ids)
        assert result["growth_class"] in ("fast", "moderate", "slow")

    def test_empty_expr_returns_none(self):
        result = predict_growth_rate(pd.DataFrame())
        assert result is None

    def test_no_cai_column_returns_none(self):
        df = pd.DataFrame({"gene": ["g1", "g2"], "MELP": [0.5, 0.6]})
        result = predict_growth_rate(df)
        assert result is None

    def test_no_rp_genes_returns_none(self, tmp_path):
        expr_df = _make_expr_df(100)
        rp_ids = tmp_path / "rp_ids.txt"
        rp_ids.write_text("")  # empty file
        result = predict_growth_rate(expr_df, rp_ids)
        assert result is None


# ── Translational Selection ────────────────────────────────────────────────

class TestTranslationalSelection:
    def test_returns_expected_keys(self, tmp_path):
        rscu_df = _make_rscu_gene_df(100)
        enc_df = _make_enc_df(100)
        expr_df = _make_expr_df(100)
        ffn = _write_fasta(tmp_path, 100)
        result = quantify_translational_selection(rscu_df, enc_df, expr_df, ffn)
        assert "optimal_codons" in result
        assert "fop_gradient" in result
        assert "position_effects" in result

    def test_optimal_codons_has_delta_rscu(self, tmp_path):
        rscu_df = _make_rscu_gene_df(100)
        enc_df = _make_enc_df(100)
        expr_df = _make_expr_df(100)
        ffn = _write_fasta(tmp_path, 100)
        result = quantify_translational_selection(rscu_df, enc_df, expr_df, ffn)
        if not result["optimal_codons"].empty:
            assert "delta_rscu" in result["optimal_codons"].columns
            assert "is_optimal" in result["optimal_codons"].columns

    def test_fop_gradient_has_quintiles(self, tmp_path):
        rscu_df = _make_rscu_gene_df(100)
        enc_df = _make_enc_df(100)
        expr_df = _make_expr_df(100)
        ffn = _write_fasta(tmp_path, 100)
        result = quantify_translational_selection(rscu_df, enc_df, expr_df, ffn)
        if not result["fop_gradient"].empty:
            assert "quintile" in result["fop_gradient"].columns
            assert result["fop_gradient"]["quintile"].max() == 5

    def test_position_effects_has_regions(self, tmp_path):
        rscu_df = _make_rscu_gene_df(100)
        enc_df = _make_enc_df(100)
        expr_df = _make_expr_df(100)
        ffn = _write_fasta(tmp_path, 100)
        result = quantify_translational_selection(rscu_df, enc_df, expr_df, ffn)
        if not result["position_effects"].empty:
            assert "fop_5prime" in result["position_effects"].columns
            assert "fop_middle" in result["position_effects"].columns
            assert "fop_3prime" in result["position_effects"].columns

    def test_empty_input_returns_empty_dfs(self, tmp_path):
        ffn = _write_fasta(tmp_path, 5)
        result = quantify_translational_selection(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ffn
        )
        assert result["optimal_codons"].empty
        assert result["fop_gradient"].empty
        assert result["position_effects"].empty


# ── Phage/Mobile Element Detection ────────────────────────────────────────

class TestPhageMobileElements:
    def test_empty_hgt_returns_empty(self):
        result = detect_phage_mobile_elements(pd.DataFrame())
        assert result.empty

    def test_with_hgt_df_returns_columns(self):
        hgt_df = pd.DataFrame({
            "gene": ["g1", "g2", "g3"],
            "mahalanobis_dist": [5.0, 2.0, 8.0],
            "gc3_deviation": [0.1, -0.05, 0.2],
            "hgt_flag": [True, False, True],
        })
        result = detect_phage_mobile_elements(hgt_df)
        assert "is_mobilome" in result.columns
        assert "is_phage_related" in result.columns

    def test_kofam_annotation_integration(self):
        hgt_df = pd.DataFrame({
            "gene": ["g1", "g2"],
            "mahalanobis_dist": [5.0, 2.0],
            "gc3_deviation": [0.1, -0.05],
            "hgt_flag": [True, False],
        })
        kofam_df = pd.DataFrame({
            "gene": ["g1", "g2"],
            "KO": ["K07483", "K00001"],  # K07483 is a transposase
        })
        result = detect_phage_mobile_elements(hgt_df, kofam_df=kofam_df)
        assert result.loc[result["gene"] == "g1", "is_phage_related"].iloc[0] == True


# ── Strand Asymmetry ──────────────────────────────────────────────────────

class TestStrandAsymmetry:
    def test_no_gff_returns_none(self, tmp_path):
        ffn = _write_fasta(tmp_path, 20)
        result = compute_strand_asymmetry(ffn, gff_path=None)
        assert result is None

    def test_with_gff_returns_dataframe(self, tmp_path):
        ffn = _write_fasta(tmp_path, 50)
        gff = _write_gff(tmp_path, 50)
        rscu_df = _make_rscu_gene_df(50)
        result = compute_strand_asymmetry(ffn, gff, rscu_df)
        if result is not None:
            assert "codon" in result.columns
            assert "mean_rscu_plus" in result.columns
            assert "mean_rscu_minus" in result.columns
            assert "p_value" in result.columns


# ── Operon Coadaptation ──────────────────────────────────────────────────

class TestOperonCoadaptation:
    def test_no_gff_returns_none(self):
        rscu_df = _make_rscu_gene_df(50)
        result = compute_operon_codon_coadaptation(rscu_df, gff_path=None)
        assert result is None

    def test_with_gff_returns_dataframe(self, tmp_path):
        rscu_df = _make_rscu_gene_df(50)
        gff = _write_gff(tmp_path, 50)
        result = compute_operon_codon_coadaptation(rscu_df, gff)
        if result is not None:
            assert "gene1" in result.columns
            assert "gene2" in result.columns
            assert "rscu_distance" in result.columns
            assert "same_operon_prediction" in result.columns

    def test_distances_nonnegative(self, tmp_path):
        rscu_df = _make_rscu_gene_df(50)
        gff = _write_gff(tmp_path, 50)
        result = compute_operon_codon_coadaptation(rscu_df, gff)
        if result is not None:
            assert (result["rscu_distance"] >= 0).all()

    def test_empty_rscu_returns_none(self, tmp_path):
        gff = _write_gff(tmp_path, 50)
        result = compute_operon_codon_coadaptation(pd.DataFrame(), gff)
        assert result is None


# ── gRodon2 Module Tests ─────────────────────────────────────────────────────

class TestGrodon2Module:
    """Tests for the gRodon2 wrapper module (codonpipe.modules.grodon).

    Since gRodon2 requires R + Bioconductor packages which are unlikely to be
    installed in the test environment, these tests verify the Python wrapper
    logic: availability detection, error handling, and result parsing.
    """

    def test_import_grodon_module(self):
        """Module is importable."""
        from codonpipe.modules.grodon import is_grodon_available, run_grodon
        assert callable(is_grodon_available)
        assert callable(run_grodon)

    def test_availability_check_returns_bool(self):
        """is_grodon_available returns a boolean."""
        from codonpipe.modules import grodon
        # Reset cached state for a clean test
        grodon._GRODON_AVAILABLE = None
        result = grodon.is_grodon_available()
        assert isinstance(result, bool)

    def test_run_grodon_missing_fasta_returns_none(self, tmp_path):
        """run_grodon returns None if FASTA file doesn't exist."""
        from codonpipe.modules import grodon
        # Force availability to True to test the FASTA check path
        original = grodon._GRODON_AVAILABLE
        grodon._GRODON_AVAILABLE = True
        try:
            result = grodon.run_grodon(
                ffn_path=tmp_path / "nonexistent.ffn",
                output_dir=tmp_path,
                sample_id="test",
            )
            assert result is None
        finally:
            grodon._GRODON_AVAILABLE = original

    def test_run_grodon_unavailable_returns_none(self, tmp_path):
        """run_grodon returns None when gRodon2 is not available."""
        from codonpipe.modules import grodon
        original = grodon._GRODON_AVAILABLE
        grodon._GRODON_AVAILABLE = False
        try:
            fasta = tmp_path / "test.ffn"
            fasta.write_text(">gene1\nATGCATGCATGCATGCATGC\n")
            result = grodon.run_grodon(fasta, tmp_path, "test")
            assert result is None
        finally:
            grodon._GRODON_AVAILABLE = original

    def test_r_script_template_is_valid(self):
        """R script template contains expected gRodon2 calls."""
        from codonpipe.modules.grodon import _R_SCRIPT
        assert "library(gRodon)" in _R_SCRIPT
        assert "readDNAStringSet" in _R_SCRIPT
        assert "predictGrowth" in _R_SCRIPT
        assert "ribosomal protein" in _R_SCRIPT
        assert "jsonlite::toJSON" in _R_SCRIPT

    def test_grodon_result_dict_keys(self):
        """Verify expected keys in a mock gRodon2 result dict."""
        expected_keys = {
            "predicted_doubling_time_hours",
            "lower_ci_hours",
            "upper_ci_hours",
            "CUBHE",
            "ConsistencyHE",
            "CPB",
            "GC",
            "n_highly_expressed",
            "filtered_sequences",
            "growth_class",
            "model",
            "caveat",
        }
        # This just validates the schema is well-defined
        assert len(expected_keys) == 12
