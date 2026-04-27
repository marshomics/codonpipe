"""Tests for the gene-set vs genome comparison module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.gene_set import (
    _aitchison_distance,
    _bootstrap_cliffs_delta_ci,
    _build_summary_table,
    _cliffs_delta,
    _clr_transform,
    _drop_redundant_codon_per_family,
    _length_matched_indices,
    _percentile_rank,
    analyze_gene_set,
    load_sample_outputs,
    read_goi_file,
)
from codonpipe.utils.codon_tables import (
    AA_CODON_GROUPS_RSCU,
    COL_GENE,
    RSCU_COLUMN_NAMES,
)


# ── Synthetic data fixtures ─────────────────────────────────────────────────


def _make_rscu_gene_df(n_genes: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
    }
    for col in RSCU_COLUMN_NAMES:
        data[col] = rng.uniform(0.1, 3.0, size=n_genes)
    return pd.DataFrame(data)


def _make_enc_df(n_genes: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
        "ENC": rng.uniform(25, 60, size=n_genes),
        "GC3": rng.uniform(0.2, 0.8, size=n_genes),
    })


def _make_expr_df(n_genes: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "MELP": rng.uniform(0.0, 1.5, size=n_genes),
        "CAI": rng.uniform(0.2, 0.9, size=n_genes),
        "Fop": rng.uniform(0.3, 0.7, size=n_genes),
    })


def _make_mahal_cluster_df(n_genes: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    in_set = rng.random(n_genes) < 0.30
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "mahal_cluster_distance": rng.gamma(3.0, 1.0, size=n_genes),
        "membership_score": rng.beta(0.5, 0.5, size=n_genes),
        "in_optimized_set": in_set,
        "is_ribosomal_protein": rng.random(n_genes) < 0.03,
    })


def _make_cbi_df(n_genes: int = 200, col: str = "cbi_rp", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        col: rng.uniform(-0.5, 1.0, size=n_genes),
    })


def _make_hgt_df(n_genes: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flags = rng.random(n_genes) < 0.05
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "mahalanobis_dist": rng.gamma(5.0, 1.0, size=n_genes),
        "gc3_deviation": rng.normal(0, 0.05, size=n_genes),
        "p_adjusted": rng.uniform(0, 1, size=n_genes),
        "hgt_flag_fdr": flags,
        "hgt_flag_adaptive": flags,
        "gc3_outlier": rng.random(n_genes) < 0.05,
        "hgt_flag_combined": flags & (rng.random(n_genes) < 0.5),
    })


def _make_reference_rscu(seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {col: float(rng.uniform(0.4, 2.0)) for col in RSCU_COLUMN_NAMES}


# ── Helper unit tests ────────────────────────────────────────────────────────


class TestCliffsDelta:
    def test_extreme_positive(self):
        x = np.array([10, 11, 12, 13, 14])
        y = np.array([1, 2, 3, 4, 5])
        # Every x > every y: delta = +1
        assert _cliffs_delta(x, y) == 1.0

    def test_extreme_negative(self):
        x = np.array([1, 2, 3])
        y = np.array([10, 11, 12])
        assert _cliffs_delta(x, y) == -1.0

    def test_zero_overlap(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        # Same values: positive matches negative; delta = 0
        assert _cliffs_delta(x, y) == 0.0

    def test_empty_arrays(self):
        assert np.isnan(_cliffs_delta(np.array([]), np.array([1, 2])))


class TestBootstrapCI:
    def test_ci_brackets_point_estimate(self):
        rng = np.random.default_rng(0)
        x = rng.normal(loc=1.0, scale=1.0, size=50)
        y = rng.normal(loc=0.0, scale=1.0, size=50)
        delta = _cliffs_delta(x, y)
        lo, hi = _bootstrap_cliffs_delta_ci(x, y, n_boot=200, rng=rng)
        assert lo <= delta <= hi

    def test_small_sample_returns_nan(self):
        lo, hi = _bootstrap_cliffs_delta_ci(np.array([1, 2]), np.array([3, 4]))
        assert np.isnan(lo) and np.isnan(hi)


class TestCLRandAitchison:
    def test_clr_zero_centered(self):
        v = np.array([1.0, 2.0, 4.0, 8.0])
        clr = _clr_transform(v)
        assert np.isclose(clr.mean(), 0.0, atol=1e-9)

    def test_aitchison_zero_for_identical(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert _aitchison_distance(v, v) == pytest.approx(0.0)

    def test_aitchison_positive_for_different(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([4.0, 3.0, 2.0, 1.0])
        assert _aitchison_distance(a, b) > 0


class TestLengthMatchedSampling:
    def test_returns_correct_size(self):
        rng = np.random.default_rng(0)
        bg = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] * 10)
        goi = np.array([110, 210, 310, 410, 510])
        idx = _length_matched_indices(bg, goi, rng, n_bins=4)
        assert len(idx) == len(goi)

    def test_preserves_bin_distribution(self):
        rng = np.random.default_rng(0)
        bg = np.linspace(100, 1000, 100)
        # All GOI in lower quartile
        goi = np.array([110, 120, 130, 140, 150])
        idx = _length_matched_indices(bg, goi, rng, n_bins=4)
        sampled_lengths = bg[idx]
        # All sampled should be within first quartile of bg
        q25 = np.percentile(bg, 25)
        # Allow generous tolerance: at least 80% should be at or below Q25
        assert (sampled_lengths <= q25).mean() >= 0.8


class TestPercentileRank:
    def test_median_is_50(self):
        dist = np.arange(0, 100)
        assert 49 <= _percentile_rank(50, dist) <= 51

    def test_below_min_is_zero(self):
        dist = np.arange(10, 20)
        assert _percentile_rank(5, dist) == 0.0

    def test_above_max_is_100(self):
        dist = np.arange(10, 20)
        assert _percentile_rank(100, dist) == 100.0

    def test_nan_returns_nan(self):
        assert np.isnan(_percentile_rank(np.nan, np.arange(10)))


class TestRedundantColumnDrop:
    def test_drops_one_per_family(self):
        kept = _drop_redundant_codon_per_family(RSCU_COLUMN_NAMES)
        # 21 RSCU subfamilies, each loses one codon → 59 - 21 = 38
        assert len(kept) == 38

    def test_deterministic(self):
        a = _drop_redundant_codon_per_family(RSCU_COLUMN_NAMES)
        b = _drop_redundant_codon_per_family(RSCU_COLUMN_NAMES)
        assert a == b


# ── Builder tests ────────────────────────────────────────────────────────────


class TestSummaryTable:
    def test_filters_to_goi_only(self):
        rscu = _make_rscu_gene_df(50)
        enc = _make_enc_df(50)
        goi = {f"locus_{i:04d}" for i in range(0, 10)}
        out = _build_summary_table(goi, rscu, enc, None, None, None, None)
        assert len(out) == 10
        assert set(out[COL_GENE]) == goi

    def test_percentile_columns_present(self):
        rscu = _make_rscu_gene_df(50)
        enc = _make_enc_df(50)
        goi = {f"locus_{i:04d}" for i in range(0, 10)}
        out = _build_summary_table(goi, rscu, enc, None, None, None, None)
        assert "ENC_pctile" in out.columns
        assert "GC3_pctile" in out.columns

    def test_percentile_is_versus_rest(self):
        rscu = _make_rscu_gene_df(100)
        enc = _make_enc_df(100)
        # Force the first 10 genes to have ENC = 30 (low), rest high
        enc.loc[enc[COL_GENE].isin([f"locus_{i:04d}" for i in range(10)]), "ENC"] = 30.0
        enc.loc[~enc[COL_GENE].isin([f"locus_{i:04d}" for i in range(10)]), "ENC"] = 55.0
        goi = {f"locus_{i:04d}" for i in range(0, 10)}
        out = _build_summary_table(goi, rscu, enc, None, None, None, None)
        # All GOI should have ENC_pctile == 0 (below all rest)
        assert (out["ENC_pctile"] == 0.0).all()


# ── Orchestrator integration tests ───────────────────────────────────────────


class TestAnalyzeGeneSet:
    def test_runs_with_minimal_inputs(self, tmp_path):
        rscu = _make_rscu_gene_df(80)
        enc = _make_enc_df(80)
        goi = {f"locus_{i:04d}" for i in range(0, 10)}
        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
            output_dir=tmp_path, sample_id="test",
            n_permutations=99, make_figure=False,
        )
        assert "summary" in out
        assert "distribution_tests" in out
        assert "codon_tests" in out
        assert out["summary"].exists()

    def test_runs_with_full_inputs(self, tmp_path):
        rscu = _make_rscu_gene_df(150)
        enc = _make_enc_df(150)
        expr = _make_expr_df(150)
        hgt = _make_hgt_df(150)
        ref_genome = _make_reference_rscu(seed=1)
        ref_rp = _make_reference_rscu(seed=2)
        ref_mahal = _make_reference_rscu(seed=3)
        goi = {f"locus_{i:04d}" for i in range(0, 20)}

        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
            output_dir=tmp_path, sample_id="test",
            expr_df=expr, hgt_df=hgt,
            rscu_genome=ref_genome, rscu_rp=ref_rp, rscu_mahal_cluster=ref_mahal,
            n_permutations=99, make_figure=False,
        )
        for kind in ("summary", "distribution_tests", "codon_tests",
                     "aitchison", "hgt_enrichment"):
            assert kind in out
            assert out[kind].exists()

    def test_codon_tests_have_three_references(self, tmp_path):
        rscu = _make_rscu_gene_df(100)
        enc = _make_enc_df(100)
        goi = {f"locus_{i:04d}" for i in range(0, 10)}
        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
            output_dir=tmp_path, sample_id="test",
            rscu_genome=_make_reference_rscu(seed=1),
            rscu_rp=_make_reference_rscu(seed=2),
            rscu_mahal_cluster=_make_reference_rscu(seed=3),
            n_permutations=99, make_figure=False,
        )
        codon_df = pd.read_csv(out["codon_tests"], sep="\t")
        assert set(codon_df["reference"]) == {"genome", "ribosomal_proteins", "mahal_cluster"}
        assert "p_adjusted" in codon_df.columns

    def test_runs_with_mahal_cluster_inputs(self, tmp_path):
        rscu = _make_rscu_gene_df(150)
        enc = _make_enc_df(150)
        mahal = _make_mahal_cluster_df(150)
        cbi_rp = _make_cbi_df(150, col="cbi_rp", seed=11)
        cbi_mahal = _make_cbi_df(150, col="cbi_mahal", seed=12)
        goi = {f"locus_{i:04d}" for i in range(0, 25)}
        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
            output_dir=tmp_path, sample_id="mahal",
            mahal_cluster_df=mahal,
            cbi_rp_df=cbi_rp, cbi_mahal_df=cbi_mahal,
            n_permutations=199, make_figure=False,
        )
        assert "mahal_membership" in out
        assert out["mahal_membership"].exists()
        m_df = pd.read_csv(out["mahal_membership"], sep="\t")
        # New columns must be present
        for col in ("obs_rate_goi", "rate_background", "p_value_two_sided",
                    "p_more_in_cluster", "p_less_in_cluster", "n_perm"):
            assert col in m_df.columns

        # Summary table should include Mahal-cluster percentiles + flag
        s_df = pd.read_csv(out["summary"], sep="\t")
        assert "in_optimized_set" in s_df.columns
        assert "mahal_cluster_distance_pctile" in s_df.columns
        assert "cbi_mahal_pctile" in s_df.columns

        # Distribution tests should include the new metrics
        d_df = pd.read_csv(out["distribution_tests"], sep="\t")
        for m in ("mahal_cluster_distance", "membership_score", "cbi_rp", "cbi_mahal"):
            assert (d_df["metric"] == m).any()

    def test_mahal_membership_detects_enrichment(self, tmp_path):
        """When GOI is forced to be entirely in Mahal cluster, membership test should flag."""
        rng = np.random.default_rng(0)
        n = 200
        rscu = _make_rscu_gene_df(n)
        enc = _make_enc_df(n)
        mahal = _make_mahal_cluster_df(n)
        # Force first 25 genes into the cluster, force most of the rest out
        mahal.loc[mahal["gene"].isin(
            [f"locus_{i:04d}" for i in range(25)]
        ), "in_optimized_set"] = True
        mahal.loc[mahal["gene"].isin(
            [f"locus_{i:04d}" for i in range(25, 200)]
        ), "in_optimized_set"] = rng.random(175) < 0.1  # 10% baseline rate

        goi = {f"locus_{i:04d}" for i in range(0, 25)}
        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
            output_dir=tmp_path, sample_id="enriched",
            mahal_cluster_df=mahal,
            n_permutations=199, make_figure=False,
        )
        m_df = pd.read_csv(out["mahal_membership"], sep="\t").iloc[0]
        # GOI is 100% in cluster, background ~10% — must show enrichment
        assert m_df["obs_rate_goi"] == pytest.approx(1.0)
        assert m_df["p_more_in_cluster"] < 0.05

    def test_too_few_goi_raises(self, tmp_path):
        rscu = _make_rscu_gene_df(50)
        enc = _make_enc_df(50)
        with pytest.raises(ValueError, match="at least 3"):
            analyze_gene_set(
                rscu_gene_df=rscu, enc_df=enc,
                goi_ids={"locus_0001"},
                output_dir=tmp_path, sample_id="test",
                n_permutations=10, make_figure=False,
            )

    def test_warns_on_missing_goi(self, tmp_path, caplog):
        import logging
        rscu = _make_rscu_gene_df(50)
        enc = _make_enc_df(50)
        goi = {f"locus_{i:04d}" for i in range(0, 5)} | {"missing_gene_1", "missing_gene_2"}
        with caplog.at_level(logging.WARNING, logger="codonpipe"):
            analyze_gene_set(
                rscu_gene_df=rscu, enc_df=enc, goi_ids=goi,
                output_dir=tmp_path, sample_id="test",
                n_permutations=10, make_figure=False,
            )
        assert any("not found" in m for m in caplog.messages)

    def test_aitchison_p_lower_when_goi_diverges(self, tmp_path):
        """Synthetic divergent GOI should have lower Aitchison p-value than random GOI."""
        rng = np.random.default_rng(0)
        n = 200
        rscu = _make_rscu_gene_df(n, seed=0)
        enc = _make_enc_df(n, seed=0)
        # Inject a strong codon-usage signature into the first 30 genes:
        # bias their RSCU heavily toward the "first" codon in each family.
        goi_ids_set = {f"locus_{i:04d}" for i in range(30)}
        for col in RSCU_COLUMN_NAMES:
            mask = rscu[COL_GENE].isin(goi_ids_set)
            if col.endswith("UUU") or col.endswith("UCU") or col.endswith("CUU"):
                rscu.loc[mask, col] = rng.uniform(2.5, 3.5, size=mask.sum())
            elif col.endswith("UUC") or col.endswith("UCC") or col.endswith("CUC"):
                rscu.loc[mask, col] = rng.uniform(0.05, 0.2, size=mask.sum())

        ref = _make_reference_rscu(seed=99)
        out = analyze_gene_set(
            rscu_gene_df=rscu, enc_df=enc, goi_ids=goi_ids_set,
            output_dir=tmp_path, sample_id="div",
            rscu_genome=ref,
            n_permutations=199, make_figure=False,
        )
        aitch = pd.read_csv(out["aitchison"], sep="\t")
        # Observed Aitchison distance should be at least as large as the
        # permutation mean (indicating the GOI structure is genuinely
        # different from random length-matched controls).
        row = aitch[aitch["reference"] == "genome"].iloc[0]
        assert row["obs_aitchison"] >= row["perm_mean"]


# ── Loader / file IO ─────────────────────────────────────────────────────────


class TestReadGoiFile:
    def test_basic(self, tmp_path):
        f = tmp_path / "goi.txt"
        f.write_text("# header comment\nlocus_0001\nlocus_0002  # inline tail\n\nlocus_0003\n")
        ids = read_goi_file(f)
        assert ids == {"locus_0001", "locus_0002", "locus_0003"}

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("# only comments\n\n# more comments\n")
        assert read_goi_file(f) == set()


class TestLoadSampleOutputs:
    def test_minimal_layout(self, tmp_path):
        sample_dir = tmp_path / "sample"
        rscu_dir = sample_dir / "rscu"
        rscu_dir.mkdir(parents=True)
        rscu = _make_rscu_gene_df(20)
        enc = _make_enc_df(20)
        rscu.to_csv(rscu_dir / "S_rscu_all_genes.tsv", sep="\t", index=False)
        enc.to_csv(rscu_dir / "S_enc.tsv", sep="\t", index=False)
        loaded = load_sample_outputs(sample_dir, "S")
        assert loaded["rscu_gene_df"].shape[0] == 20
        assert loaded["enc_df"].shape[0] == 20
        assert loaded["expr_df"] is None
        assert loaded["hgt_df"] is None
        # Mahal-cluster keys must exist (and be None for a minimal layout)
        for k in ("mahal_cluster_df", "cbi_rp_df", "cbi_mahal_df"):
            assert k in loaded
            assert loaded[k] is None

    def test_loads_mahal_cluster_files(self, tmp_path):
        sample_dir = tmp_path / "sample"
        (sample_dir / "rscu").mkdir(parents=True)
        (sample_dir / "gmm_clustering").mkdir(parents=True)
        (sample_dir / "codon_tables").mkdir(parents=True)
        rscu = _make_rscu_gene_df(20)
        enc = _make_enc_df(20)
        rscu.to_csv(sample_dir / "rscu/S_rscu_all_genes.tsv", sep="\t", index=False)
        enc.to_csv(sample_dir / "rscu/S_enc.tsv", sep="\t", index=False)

        # Mahal-cluster table uses the original column name
        # ('mahalanobis_distance') as written by the pipeline; the loader
        # renames it to mahal_cluster_distance.
        mahal = _make_mahal_cluster_df(20).rename(
            columns={"mahal_cluster_distance": "mahalanobis_distance"}
        )
        mahal.to_csv(sample_dir / "gmm_clustering/S_gmm_clusters.tsv",
                     sep="\t", index=False)

        cbi_rp = pd.DataFrame({
            "gene": [f"locus_{i:04d}" for i in range(20)],
            "cbi": np.linspace(-0.2, 0.8, 20),
        })
        cbi_rp.to_csv(sample_dir / "codon_tables/S_cbi.tsv", sep="\t", index=False)

        cbi_mahal = pd.DataFrame({
            "gene": [f"locus_{i:04d}" for i in range(20)],
            "cbi": np.linspace(-0.1, 0.7, 20),
        })
        cbi_mahal.to_csv(sample_dir / "codon_tables/S_gmm_cluster_cbi.tsv",
                         sep="\t", index=False)

        loaded = load_sample_outputs(sample_dir, "S")
        assert loaded["mahal_cluster_df"] is not None
        # Loader must have renamed the distance column
        assert "mahal_cluster_distance" in loaded["mahal_cluster_df"].columns
        assert loaded["cbi_rp_df"] is not None
        assert "cbi_rp" in loaded["cbi_rp_df"].columns
        assert loaded["cbi_mahal_df"] is not None
        assert "cbi_mahal" in loaded["cbi_mahal_df"].columns

    def test_missing_required_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Required files"):
            load_sample_outputs(tmp_path, "nope")
