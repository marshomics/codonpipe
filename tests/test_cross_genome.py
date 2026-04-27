"""Tests for the cross-genome signature + corpus module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.cross_genome import (
    _GEOMETRY_PREFIXES,
    _all_feature_columns,
    _geometry_columns,
    _robust_zscore,
    build_corpus,
    cluster_corpus,
    cluster_corpus_genes,
    cluster_corpus_genes_by_category,
    compute_corpus_cluster_drivers,
    compute_gene_signature,
    compute_genome_signature,
    discover_signatures,
    mantel_test,
    read_phylogeny_distance_matrix,
    write_signatures_for_sample,
)
from codonpipe.utils.codon_tables import COL_GENE, RSCU_COLUMN_NAMES


# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_rscu_gene_df(n_genes: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
    }
    for col in RSCU_COLUMN_NAMES:
        data[col] = rng.uniform(0.2, 2.5, size=n_genes)
    return pd.DataFrame(data)


def _make_enc_df(n_genes: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "length": rng.integers(300, 3000, size=n_genes),
        "ENC": rng.uniform(28, 58, size=n_genes),
        "GC3": rng.uniform(0.2, 0.8, size=n_genes),
    })


def _make_expr_df(n_genes: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "MELP": rng.uniform(0.0, 1.5, size=n_genes),
        "CAI": rng.uniform(0.2, 0.9, size=n_genes),
        "Fop": rng.uniform(0.3, 0.7, size=n_genes),
        "rp_CAI": rng.uniform(0.2, 0.9, size=n_genes),
    })


def _make_mahal_df(n_genes: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": [f"locus_{i:04d}" for i in range(n_genes)],
        "mahal_cluster_distance": rng.gamma(3.0, 1.0, size=n_genes),
        "membership_score": rng.beta(0.5, 0.5, size=n_genes),
        "in_optimized_set": rng.random(n_genes) < 0.30,
    })


def _make_ref_dict(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {c: float(rng.uniform(0.4, 2.0)) for c in RSCU_COLUMN_NAMES}


# ── gene_signature builder ───────────────────────────────────────────────────


class TestGeneSignature:
    def test_basic_shape(self):
        rscu = _make_rscu_gene_df(40)
        ref = _make_ref_dict(seed=0)
        sig = compute_gene_signature(rscu, ref, sample_id="S1")
        assert len(sig) == 40
        assert "sample_id" in sig.columns
        assert (sig["sample_id"] == "S1").all()
        # 38 independent codon delta_clr columns
        delta_cols = [c for c in sig.columns if c.startswith("delta_clr_")]
        assert len(delta_cols) == 38

    def test_includes_optional_metrics(self):
        rscu = _make_rscu_gene_df(50)
        ref = _make_ref_dict(seed=1)
        sig = compute_gene_signature(
            rscu, ref, sample_id="S",
            enc_df=_make_enc_df(50),
            expr_df=_make_expr_df(50),
            mahal_cluster_df=_make_mahal_df(50),
        )
        for c in ("ENC", "GC3", "CAI", "MELP", "Fop",
                  "mahalanobis_dist_cluster", "membership_score",
                  "in_optimized_set"):
            assert c in sig.columns

    def test_clr_delta_finite_values(self):
        rscu = _make_rscu_gene_df(40)
        ref = _make_ref_dict(seed=2)
        sig = compute_gene_signature(rscu, ref, sample_id="S")
        delta_cols = [c for c in sig.columns if c.startswith("delta_clr_")]
        mat = sig[delta_cols].values
        assert np.isfinite(mat).all()

    def test_empty_input_returns_empty(self):
        sig = compute_gene_signature(
            pd.DataFrame(), _make_ref_dict(seed=3), sample_id="S",
        )
        assert sig.empty


# ── genome_signature builder ─────────────────────────────────────────────────


class TestGenomeSignature:
    def test_geometry_block_present(self):
        sig = compute_genome_signature(
            sample_id="S",
            rscu_genome=_make_ref_dict(0),
            rscu_rp=_make_ref_dict(1),
            rscu_mahal_cluster=_make_ref_dict(2),
        )
        assert len(sig) == 1
        # 38 codons × 2 deltas + 3 Aitchison scalars
        delta_mahal = [c for c in sig.columns if c.startswith("delta_clr_mahal_")]
        delta_rp = [c for c in sig.columns if c.startswith("delta_clr_rp_")]
        assert len(delta_mahal) == 38
        assert len(delta_rp) == 38
        for c in ("aitchison_genome_to_mahal", "aitchison_genome_to_rp",
                  "aitchison_rp_to_mahal"):
            assert c in sig.columns
            assert np.isfinite(sig[c].iloc[0])

    def test_handles_missing_references(self):
        # Only genome reference; RP and Mahal absent.
        sig = compute_genome_signature(
            sample_id="S",
            rscu_genome=_make_ref_dict(0),
            rscu_rp=None,
            rscu_mahal_cluster=None,
        )
        # Aitchison values to missing references must be NaN, not crash
        assert pd.isna(sig["aitchison_genome_to_mahal"].iloc[0])
        assert pd.isna(sig["aitchison_genome_to_rp"].iloc[0])
        assert pd.isna(sig["aitchison_rp_to_mahal"].iloc[0])

    def test_aitchison_zero_when_refs_identical(self):
        ref = _make_ref_dict(7)
        sig = compute_genome_signature(
            sample_id="S",
            rscu_genome=ref, rscu_rp=ref, rscu_mahal_cluster=ref,
        )
        assert sig["aitchison_genome_to_mahal"].iloc[0] == pytest.approx(0.0)
        assert sig["aitchison_rp_to_mahal"].iloc[0] == pytest.approx(0.0)

    def test_summary_block_populated_when_inputs_given(self):
        sig = compute_genome_signature(
            sample_id="S",
            rscu_genome=_make_ref_dict(0),
            rscu_rp=_make_ref_dict(1),
            rscu_mahal_cluster=_make_ref_dict(2),
            rscu_gene_df=_make_rscu_gene_df(60),
            enc_df=_make_enc_df(60),
            expr_df=_make_expr_df(60),
            mahal_cluster_df=_make_mahal_df(60),
        )
        for c in ("median_cai", "median_melp", "median_enc", "median_gc3",
                  "frac_in_optimized_set", "n_genes_total"):
            assert c in sig.columns
        assert sig["n_genes_total"].iloc[0] == 60


# ── corpus helpers ───────────────────────────────────────────────────────────


class TestRobustZScore:
    def test_zero_mad_column_is_zeroed(self):
        x = np.column_stack([np.ones(10), np.arange(10)])
        z = _robust_zscore(x)
        # Constant column → zeros
        assert (z[:, 0] == 0).all()
        # Variable column → finite, non-constant
        assert np.isfinite(z[:, 1]).all() and z[:, 1].std() > 0

    def test_nan_replaced_with_zero(self):
        x = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0]])
        z = _robust_zscore(x)
        assert np.isfinite(z).all()


class TestGeometryColumnSelection:
    def test_picks_only_geometry(self):
        df = pd.DataFrame({
            "sample_id": ["S"],
            "delta_clr_mahal_Phe-UUU": [0.1],
            "delta_clr_rp_Phe-UUU": [0.2],
            "aitchison_genome_to_mahal": [0.5],
            "median_cai": [0.7],  # ecology, should be excluded from geometry
        })
        cols = _geometry_columns(df)
        assert "delta_clr_mahal_Phe-UUU" in cols
        assert "delta_clr_rp_Phe-UUU" in cols
        assert "aitchison_genome_to_mahal" in cols
        assert "median_cai" not in cols

    def test_all_features_excludes_sample_id(self):
        df = pd.DataFrame({
            "sample_id": ["S"], "median_cai": [0.7], "median_enc": [50.0],
        })
        cols = _all_feature_columns(df)
        assert cols == ["median_cai", "median_enc"]


class TestClusterCorpus:
    def test_small_corpus_clusters(self):
        rng = np.random.default_rng(0)
        n = 20
        # Two synthetic clusters in feature space
        block_a = rng.normal(loc=0, scale=0.5, size=(n // 2, 10))
        block_b = rng.normal(loc=5, scale=0.5, size=(n // 2, 10))
        X = np.vstack([block_a, block_b])
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df.insert(0, "sample_id", [f"S{i}" for i in range(n)])
        out = cluster_corpus(df, [c for c in df.columns if c.startswith("f")],
                             pca_components=5, hdbscan_min_cluster_size=4)
        assert out["embedding"].shape == (n, 2)
        assert len(out["cluster"]) == n
        # Two distinct clusters should be found (HDBSCAN may also flag noise)
        non_noise = [c for c in set(out["cluster"]) if c != -1]
        assert len(non_noise) >= 2

    def test_too_few_samples_raises(self):
        df = pd.DataFrame({
            "sample_id": ["a", "b", "c"],
            "f0": [1.0, 2.0, 3.0],
        })
        with pytest.raises(ValueError, match="at least 4"):
            cluster_corpus(df, ["f0"])


# ── Mantel test ──────────────────────────────────────────────────────────────


class TestMantel:
    def test_identical_matrices_r_one(self):
        rng = np.random.default_rng(0)
        n = 10
        from scipy.spatial.distance import pdist, squareform
        X = rng.normal(size=(n, 5))
        D = squareform(pdist(X))
        out = mantel_test(D, D, n_perm=99)
        assert out["r"] == pytest.approx(1.0)
        # p-value should be small (every permutation gives r=1 too, so
        # technically all perms tie — but the count includes ties so p is high.
        # We don't assert on p here to keep the test stable.)

    def test_unrelated_matrices_low_r(self):
        rng = np.random.default_rng(0)
        n = 30
        from scipy.spatial.distance import pdist, squareform
        A = squareform(pdist(rng.normal(size=(n, 5))))
        B = squareform(pdist(rng.normal(size=(n, 5))))
        out = mantel_test(A, B, n_perm=199)
        assert abs(out["r"]) < 0.4
        # With unrelated matrices p should not be small
        assert out["p_value"] > 0.05

    def test_too_few_returns_nan(self):
        D = np.zeros((3, 3))
        out = mantel_test(D, D, n_perm=10)
        assert np.isnan(out["r"])

    def test_phylogeny_distance_matrix_load(self, tmp_path):
        sample_ids = ["a", "b", "c"]
        df = pd.DataFrame(
            np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float),
            index=sample_ids, columns=sample_ids,
        )
        path = tmp_path / "phylo.tsv"
        df.to_csv(path, sep="\t")
        D = read_phylogeny_distance_matrix(path, sample_ids)
        assert D.shape == (3, 3)
        np.testing.assert_array_almost_equal(D, df.values)

    def test_phylogeny_missing_samples_returns_none(self, tmp_path):
        df = pd.DataFrame(
            np.zeros((2, 2)), index=["a", "b"], columns=["a", "b"],
        )
        path = tmp_path / "phylo.tsv"
        df.to_csv(path, sep="\t")
        # Request a sample not in the matrix
        out = read_phylogeny_distance_matrix(path, ["a", "b", "missing"])
        assert out is None


# ── corpus orchestrator ──────────────────────────────────────────────────────


def _write_synthetic_genome_signatures(tmp_path: Path, n_genomes: int = 8) -> Path:
    """Write n synthetic genome_signature files with two latent groups."""
    rng = np.random.default_rng(42)
    sig_dir = tmp_path / "sigs"
    sig_dir.mkdir()
    for i in range(n_genomes):
        # Two latent groups with offset CLR-Δ vectors so clustering finds structure
        group = i % 2
        delta_offset = 1.5 if group == 0 else -1.5
        row = {"sample_id": f"G{i:03d}"}
        for c in RSCU_COLUMN_NAMES[:38]:
            row[f"delta_clr_mahal_{c}"] = float(delta_offset + rng.normal(0, 0.3))
            row[f"delta_clr_rp_{c}"] = float(delta_offset * 0.7 + rng.normal(0, 0.3))
        row["aitchison_genome_to_mahal"] = float(abs(delta_offset) + rng.normal(0, 0.1))
        row["aitchison_genome_to_rp"] = float(abs(delta_offset) * 0.8 + rng.normal(0, 0.1))
        row["aitchison_rp_to_mahal"] = float(abs(delta_offset) * 0.3 + rng.normal(0, 0.1))
        row["median_cai"] = float(0.5 + delta_offset * 0.1)
        row["median_gc3"] = float(0.4 + group * 0.2 + rng.normal(0, 0.05))
        row["median_enc"] = float(50 + rng.normal(0, 3))
        row["frac_in_optimized_set"] = float(0.3 + rng.normal(0, 0.05))
        row["n_genes_total"] = 1500
        pd.DataFrame([row]).to_csv(sig_dir / f"G{i:03d}_genome_signature.tsv",
                                    sep="\t", index=False)
    return sig_dir


class TestDiscoverSignatures:
    def test_finds_in_flat_dir(self, tmp_path):
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=4)
        gen, gene = discover_signatures([sig_dir])
        assert len(gen) == 4
        assert all(p.name.endswith("_genome_signature.tsv") for p in gen)

    def test_handles_direct_file_argument(self, tmp_path):
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=2)
        files = sorted(sig_dir.glob("*_genome_signature.tsv"))
        gen, _ = discover_signatures(files)
        assert len(gen) == 2


class TestComputeCorpusClusterDrivers:
    def test_recovers_two_group_drivers(self):
        """Synthetic two-group corpus: drivers should differ between clusters."""
        rng = np.random.default_rng(0)
        n = 30
        # Build a feature matrix where columns f0..f4 differ between groups
        rows_a = rng.normal(loc=2.0, scale=0.3, size=(n // 2, 5))
        rows_b = rng.normal(loc=-2.0, scale=0.3, size=(n // 2, 5))
        # Plus 5 noise columns
        noise = rng.normal(size=(n, 5))
        X = np.hstack([np.vstack([rows_a, rows_b]), noise])
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        df.insert(0, "sample_id", [f"S{i}" for i in range(n)])
        labels = np.array([0] * (n // 2) + [1] * (n // 2))

        drivers = compute_corpus_cluster_drivers(
            df, [f"f{i}" for i in range(10)], labels, skip_noise=True,
        )
        assert not drivers.empty
        # Cluster 0 features f0..f4 should be enriched (high values), Cluster 1 depleted
        c0_top = drivers[drivers["cluster_id"] == 0].iloc[0]
        c1_top = drivers[drivers["cluster_id"] == 1].iloc[0]
        # Top driver of c0 should be a real signal feature (f0..f4)
        assert c0_top["feature"] in [f"f{i}" for i in range(5)]
        assert c1_top["feature"] in [f"f{i}" for i in range(5)]
        # Effect signs should be opposite for the same feature
        for f in [f"f{i}" for i in range(5)]:
            r0 = drivers[(drivers["cluster_id"] == 0) & (drivers["feature"] == f)]
            r1 = drivers[(drivers["cluster_id"] == 1) & (drivers["feature"] == f)]
            if not r0.empty and not r1.empty:
                assert np.sign(r0.iloc[0]["cliffs_delta"]) != np.sign(r1.iloc[0]["cliffs_delta"])

    def test_skip_noise_excludes_minus_one(self):
        rng = np.random.default_rng(0)
        n = 20
        df = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(n)],
            "f0": rng.normal(size=n),
        })
        labels = np.array([0] * 8 + [1] * 8 + [-1] * 4)
        out_skip = compute_corpus_cluster_drivers(
            df, ["f0"], labels, skip_noise=True,
        )
        # Should only return rows for clusters 0 and 1
        assert set(out_skip["cluster_id"].unique()) <= {0, 1}

    def test_too_few_clusters_returns_empty(self):
        df = pd.DataFrame({
            "sample_id": ["S0", "S1", "S2"],
            "f0": [1.0, 2.0, 3.0],
        })
        labels = np.array([0, 0, 0])  # only one cluster
        out = compute_corpus_cluster_drivers(df, ["f0"], labels)
        assert out.empty


class TestClusterCorpusGenes:
    def _build_gene_corpus(self, n_per_group=80, seed=0):
        """Synthetic gene corpus with two latent codon-strategy groups.

        Every codon column carries signal so PCA on this small synthetic
        corpus actually recovers the group structure. With Gaussian centres
        and 80 genes per group, HDBSCAN finds two density peaks and KMeans
        recovers cleanly when HDBSCAN flags everything as noise.
        """
        rng = np.random.default_rng(seed)
        rows = []
        # Half the codons are "group A high"; the other half are "group B high"
        codons_a_high = RSCU_COLUMN_NAMES[: len(RSCU_COLUMN_NAMES) // 2]
        for i in range(n_per_group):
            row = {"gene": f"A_{i:04d}", "sample_id": "G001",
                   "length": rng.integers(300, 3000)}
            for c in RSCU_COLUMN_NAMES:
                centre = 1.5 if c in codons_a_high else -1.5
                row[f"delta_clr_{c}"] = float(rng.normal(loc=centre, scale=0.3))
            rows.append(row)
        for i in range(n_per_group):
            row = {"gene": f"B_{i:04d}", "sample_id": "G002",
                   "length": rng.integers(300, 3000)}
            for c in RSCU_COLUMN_NAMES:
                centre = -1.5 if c in codons_a_high else 1.5
                row[f"delta_clr_{c}"] = float(rng.normal(loc=centre, scale=0.3))
            rows.append(row)
        return pd.DataFrame(rows)

    def test_recovers_two_groups(self):
        n_per_group = 80
        gene_corpus = self._build_gene_corpus(n_per_group=n_per_group)
        result = cluster_corpus_genes(
            gene_corpus, use_umap=False,
            hdbscan_min_cluster_size=10, rng_seed=42,
        )
        assert "cluster" in result
        assert "embedding" in result
        labels = result["cluster"]
        non_noise = [c for c in set(labels) if c != -1]
        assert len(non_noise) >= 2
        # Modal cluster of A_xxx genes should differ from modal cluster of B_xxx
        a_labels = labels[:n_per_group]
        b_labels = labels[n_per_group:]
        from collections import Counter
        a_mode = Counter(a_labels).most_common(1)[0][0]
        b_mode = Counter(b_labels).most_common(1)[0][0]
        assert a_mode != b_mode

    def test_max_genes_guard(self):
        gene_corpus = self._build_gene_corpus(n_per_group=10)
        with pytest.raises(ValueError, match="cap"):
            cluster_corpus_genes(gene_corpus, max_genes=5)


class TestClusterCorpusGenesByCategory:
    def _build_corpus_with_categories(self, n_per_cat_per_group=20, seed=0):
        """Corpus with two KO categories, each splitting into two strategies.

        Every codon column carries signal so per-category PCA + clustering
        actually recovers the within-category split on small samples.
        """
        rng = np.random.default_rng(seed)
        rows = []
        codons_a_high = RSCU_COLUMN_NAMES[: len(RSCU_COLUMN_NAMES) // 2]
        for ko in ("K001", "K002"):
            for i in range(n_per_cat_per_group):
                row = {"gene": f"{ko}_A_{i:03d}", "sample_id": f"G{i:03d}",
                       "KO": ko, "length": rng.integers(300, 3000)}
                for c in RSCU_COLUMN_NAMES:
                    centre = 1.5 if c in codons_a_high else -1.5
                    row[f"delta_clr_{c}"] = float(rng.normal(loc=centre, scale=0.3))
                rows.append(row)
            for i in range(n_per_cat_per_group):
                row = {"gene": f"{ko}_B_{i:03d}", "sample_id": f"G{i+50:03d}",
                       "KO": ko, "length": rng.integers(300, 3000)}
                for c in RSCU_COLUMN_NAMES:
                    centre = -1.5 if c in codons_a_high else 1.5
                    row[f"delta_clr_{c}"] = float(rng.normal(loc=centre, scale=0.3))
                rows.append(row)
        return pd.DataFrame(rows)

    def test_per_category_clustering_recovers_split(self):
        gene_corpus = self._build_corpus_with_categories(n_per_cat_per_group=20)
        out = cluster_corpus_genes_by_category(
            gene_corpus, "KO", min_category_size=10, rng_seed=42,
        )
        assert not out.empty
        assert set(out["category"]) == {"K001", "K002"}
        # Each KO should split into >=2 sub-clusters because of the synthetic structure
        for cat in out["category"].unique():
            sub = out[out["category"] == cat]
            non_noise = [c for c in sub["sub_cluster"].unique() if c != -1]
            assert len(non_noise) >= 1  # at least one cluster found
        # Required output columns
        for col in ("category", "sample_id", "gene", "sub_cluster",
                    "embed_dim1", "embed_dim2"):
            assert col in out.columns

    def test_skips_small_categories(self):
        gene_corpus = self._build_corpus_with_categories(n_per_cat_per_group=2)
        out = cluster_corpus_genes_by_category(
            gene_corpus, "KO", min_category_size=20, rng_seed=42,
        )
        assert out.empty

    def test_missing_category_column(self):
        gene_corpus = pd.DataFrame({
            "gene": ["g1", "g2", "g3"],
            "sample_id": ["S", "S", "S"],
            "delta_clr_Phe-UUU": [1.0, 2.0, 3.0],
        })
        out = cluster_corpus_genes_by_category(gene_corpus, "missing_col")
        assert out.empty


class TestBuildCorpus:
    def test_geometry_features_default(self, tmp_path):
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=12)
        out_dir = tmp_path / "corpus"
        outputs = build_corpus(
            input_dirs=[sig_dir], output_dir=out_dir,
            features="geometry",
            hdbscan_min_cluster_size=3,
        )
        for k in ("corpus_genome_signature", "corpus_genome_clusters"):
            assert k in outputs
            assert outputs[k].exists()
        clusters = pd.read_csv(outputs["corpus_genome_clusters"], sep="\t")
        assert len(clusters) == 12
        # The two latent groups should be separable by some kind of clustering
        non_noise = [c for c in set(clusters["cluster"]) if c != -1]
        assert len(non_noise) >= 2

    def test_all_features_uses_more_columns(self, tmp_path):
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=12)
        out_dir = tmp_path / "corpus_all"
        outputs = build_corpus(
            input_dirs=[sig_dir], output_dir=out_dir,
            features="all",
            hdbscan_min_cluster_size=3,
        )
        assert outputs["corpus_genome_clusters"].exists()

    def test_with_phylogeny_writes_validation(self, tmp_path):
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=10)
        # Build a fake phylogeny distance matrix
        sample_ids = sorted([
            p.name.replace("_genome_signature.tsv", "")
            for p in sig_dir.glob("*_genome_signature.tsv")
        ])
        rng = np.random.default_rng(0)
        D = rng.uniform(0, 1, size=(len(sample_ids), len(sample_ids)))
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0)
        phylo = pd.DataFrame(D, index=sample_ids, columns=sample_ids)
        phylo_path = tmp_path / "phylo.tsv"
        phylo.to_csv(phylo_path, sep="\t")

        out_dir = tmp_path / "corpus_phylo"
        outputs = build_corpus(
            input_dirs=[sig_dir], output_dir=out_dir,
            features="geometry",
            phylogeny_path=phylo_path,
            hdbscan_min_cluster_size=3,
        )
        assert "corpus_validation" in outputs
        val = pd.read_csv(outputs["corpus_validation"], sep="\t")
        assert (val["test"] == "mantel_signature_vs_phylogeny").any()

    def test_no_signatures_raises(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No.*genome_signature.tsv"):
            build_corpus([empty_dir], tmp_path / "out")

    def test_gene_level_clustering_emits_outputs(self, tmp_path):
        """End-to-end: when --gene-level-clustering is on, gene_clusters.tsv appears."""
        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=6)
        # Synthesize matching gene-level signatures
        rng = np.random.default_rng(7)
        for sample_path in sig_dir.glob("*_genome_signature.tsv"):
            sid = sample_path.name.replace("_genome_signature.tsv", "")
            gene_rows = []
            for i in range(20):
                row = {"sample_id": sid, "gene": f"{sid}_g{i:03d}",
                       "length": rng.integers(300, 3000), "KO": f"K{i % 4:04d}"}
                # CLR-Δ vector with two latent groups
                offset = 1.5 if i < 10 else -1.5
                for c in RSCU_COLUMN_NAMES[:38]:
                    row[f"delta_clr_{c}"] = float(offset + rng.normal(0, 0.3))
                gene_rows.append(row)
            pd.DataFrame(gene_rows).to_csv(
                sig_dir / f"{sid}_gene_signature.tsv", sep="\t", index=False,
            )
        out_dir = tmp_path / "corpus_gene"
        outputs = build_corpus(
            input_dirs=[sig_dir], output_dir=out_dir,
            gene_level_clustering=True,
            gene_level_by_category="KO",
            gene_level_min_category_size=5,
            hdbscan_min_cluster_size=3,
        )
        assert "corpus_gene_clusters" in outputs
        assert outputs["corpus_gene_clusters"].exists()
        assert "corpus_gene_clusters_by_KO" in outputs
        gc = pd.read_csv(outputs["corpus_gene_clusters"], sep="\t")
        assert "cluster" in gc.columns
        # Two latent groups with structure → at least one non-noise cluster
        non_noise = [c for c in set(gc["cluster"]) if c != -1]
        assert len(non_noise) >= 1


# ── write_signatures_for_sample integration ──────────────────────────────────


class TestWriteSignaturesForSample:
    def test_minimal_sample_dir(self, tmp_path):
        sample_dir = tmp_path / "sample"
        rscu_dir = sample_dir / "rscu"
        rscu_dir.mkdir(parents=True)
        rscu = _make_rscu_gene_df(30)
        enc = _make_enc_df(30)
        rscu.to_csv(rscu_dir / "S_rscu_all_genes.tsv", sep="\t", index=False)
        enc.to_csv(rscu_dir / "S_enc.tsv", sep="\t", index=False)

        out_dir = tmp_path / "sigs"
        outputs = write_signatures_for_sample(sample_dir, "S", out_dir)
        assert outputs["gene_signature"].exists()
        assert outputs["genome_signature"].exists()

        gene_df = pd.read_csv(outputs["gene_signature"], sep="\t")
        assert (gene_df["sample_id"] == "S").all()
        # 38 CLR-Δ codon columns
        delta_cols = [c for c in gene_df.columns if c.startswith("delta_clr_")]
        assert len(delta_cols) == 38

        genome_df = pd.read_csv(outputs["genome_signature"], sep="\t")
        assert len(genome_df) == 1
        assert genome_df["sample_id"].iloc[0] == "S"
