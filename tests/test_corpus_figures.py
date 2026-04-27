"""Smoke tests for the corpus figure renderers.

Verify each function produces non-empty PNG/SVG output when given valid
inputs, and gracefully returns (None, None) for edge cases. The figures'
exact pixel content isn't checked — that's not what unit tests are for.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules._corpus_figures import (
    render_cluster_drivers_forest,
    render_cluster_signature_heatmap,
    render_focus_genome_locator,
    render_mantel_stratified,
    render_multi_overlay_umap,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_corpus(n_genomes: int = 24, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Synthetic corpus DataFrame + 2-D embedding + cluster labels with two groups."""
    rng = np.random.default_rng(seed)
    sample_ids = [f"G{i:03d}" for i in range(n_genomes)]
    half = n_genomes // 2
    # CLR-Δ blocks: group A high, group B low for the first 5 codons
    cols_mahal = [f"delta_clr_mahal_codon_{i}" for i in range(38)]
    cols_rp = [f"delta_clr_rp_codon_{i}" for i in range(38)]
    delta_a = np.hstack([
        rng.normal(loc=2.0, scale=0.4, size=(half, 5)),
        rng.normal(loc=0.0, scale=0.3, size=(half, 33)),
    ])
    delta_b = np.hstack([
        rng.normal(loc=-2.0, scale=0.4, size=(n_genomes - half, 5)),
        rng.normal(loc=0.0, scale=0.3, size=(n_genomes - half, 33)),
    ])
    delta_mat = np.vstack([delta_a, delta_b])

    rows = []
    for i, sid in enumerate(sample_ids):
        row = {"sample_id": sid}
        for j, c in enumerate(cols_mahal):
            row[c] = float(delta_mat[i, j])
        for j, c in enumerate(cols_rp):
            row[c] = float(delta_mat[i, j] * 0.7)
        row["aitchison_genome_to_mahal"] = float(rng.uniform(0.5, 5.0))
        row["aitchison_genome_to_rp"] = float(rng.uniform(0.5, 5.0))
        row["aitchison_rp_to_mahal"] = float(rng.uniform(0.1, 2.0))
        row["median_gc3"] = 0.4 + (i / n_genomes) * 0.2
        row["median_cai"] = float(rng.uniform(0.5, 0.85))
        row["frac_in_optimized_set"] = float(rng.uniform(0.1, 0.5))
        row["grodon2_doubling_time_h"] = float(rng.uniform(1, 12))
        row["hgt_candidate_frac"] = float(rng.uniform(0.05, 0.3))
        rows.append(row)
    corpus_df = pd.DataFrame(rows)

    # 2D embedding: clear separation along axis 1
    embedding = np.column_stack([
        np.concatenate([rng.normal(loc=-3, scale=0.5, size=half),
                        rng.normal(loc=3, scale=0.5, size=n_genomes - half)]),
        rng.normal(scale=0.5, size=n_genomes),
    ])
    cluster_labels = np.array([0] * half + [1] * (n_genomes - half))
    return corpus_df, embedding, cluster_labels


# ── tests ────────────────────────────────────────────────────────────────────


class TestMultiOverlay:
    def test_produces_files(self, tmp_path):
        corpus, embed, labels = _make_corpus(20)
        png, svg = render_multi_overlay_umap(
            tmp_path, embed, list(corpus["sample_id"]), labels, corpus,
        )
        assert png is not None and png.exists() and png.stat().st_size > 1000
        assert svg is not None and svg.exists()

    def test_handles_missing_overlays(self, tmp_path):
        # Strip all optional columns; only sample_id remains
        corpus, embed, labels = _make_corpus(15)
        sub = corpus[["sample_id"]].copy()
        png, svg = render_multi_overlay_umap(
            tmp_path, embed, list(corpus["sample_id"]), labels, sub,
        )
        # cluster overlay alone is enough to produce a figure
        assert png is not None
        assert png.exists()


class TestClusterSignatureHeatmap:
    def test_produces_files(self, tmp_path):
        corpus, _, labels = _make_corpus(30)
        png, svg = render_cluster_signature_heatmap(tmp_path, corpus, labels)
        assert png is not None and png.exists() and png.stat().st_size > 1000

    def test_returns_none_without_clr_columns(self, tmp_path):
        # Strip CLR-Δ columns
        corpus, _, labels = _make_corpus(20)
        no_clr = corpus[["sample_id", "median_cai"]].copy()
        png, svg = render_cluster_signature_heatmap(tmp_path, no_clr, labels)
        assert png is None and svg is None


class TestClusterDriversForest:
    def test_produces_files(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 12
        drivers = pd.DataFrame({
            "cluster_id": [0] * (n // 2) + [1] * (n // 2),
            "feature": [f"feat_{i}" for i in range(n)],
            "median_in": rng.uniform(size=n),
            "median_out": rng.uniform(size=n),
            "U_statistic": rng.uniform(0, 100, size=n),
            "p_value": rng.uniform(0, 0.05, size=n),
            "cliffs_delta": rng.uniform(-1, 1, size=n),
            "abs_effect": np.abs(rng.uniform(-1, 1, size=n)),
            "p_adjusted": rng.uniform(0, 0.05, size=n),
            "significant": [True] * n,
        })
        png, svg = render_cluster_drivers_forest(tmp_path, drivers)
        assert png is not None and png.exists()

    def test_empty_drivers_returns_none(self, tmp_path):
        png, svg = render_cluster_drivers_forest(tmp_path, pd.DataFrame())
        assert png is None and svg is None


class TestMantelStratified:
    def test_produces_files(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 20
        sample_ids = [f"G{i:03d}" for i in range(n)]
        sig = rng.uniform(0, 5, size=(n, n))
        sig = (sig + sig.T) / 2
        np.fill_diagonal(sig, 0)
        phy = rng.uniform(0, 5, size=(n, n))
        phy = (phy + phy.T) / 2
        np.fill_diagonal(phy, 0)
        meta = pd.DataFrame({
            "sample_id": sample_ids,
            "phylum": ["A"] * (n // 2) + ["B"] * (n // 2),
        })
        png, svg = render_mantel_stratified(
            tmp_path, sample_ids, sig, phy, metadata_df=meta, group_col="phylum",
        )
        assert png is not None and png.exists()

    def test_too_few_samples_returns_none(self, tmp_path):
        png, svg = render_mantel_stratified(
            tmp_path, ["a", "b", "c"], np.zeros((3, 3)), np.zeros((3, 3)),
        )
        assert png is None and svg is None


class TestFocusGenomeLocator:
    def test_produces_files(self, tmp_path):
        corpus, embed, labels = _make_corpus(20)
        sample_ids = list(corpus["sample_id"])
        png, svg = render_focus_genome_locator(
            tmp_path, sample_ids[3], embed, sample_ids, labels, corpus,
        )
        assert png is not None and png.exists() and png.stat().st_size > 1000

    def test_unknown_sample_returns_none(self, tmp_path):
        corpus, embed, labels = _make_corpus(15)
        sample_ids = list(corpus["sample_id"])
        png, svg = render_focus_genome_locator(
            tmp_path, "nonexistent_id", embed, sample_ids, labels, corpus,
        )
        assert png is None and svg is None


class TestBuildCorpusIntegration:
    """End-to-end: build_corpus should emit the new figure files when feasible."""

    def test_emits_new_figures(self, tmp_path):
        from codonpipe.modules.cross_genome import build_corpus
        # Reuse the synthetic-genomes helper from test_cross_genome
        from tests.test_cross_genome import _write_synthetic_genome_signatures

        sig_dir = _write_synthetic_genome_signatures(tmp_path, n_genomes=14)
        out_dir = tmp_path / "corpus"
        outputs = build_corpus(
            input_dirs=[sig_dir], output_dir=out_dir,
            features="geometry",
            hdbscan_min_cluster_size=3,
            focus_genomes=["G000"],
        )
        # Five figure outputs should be present (or at least the first three)
        assert "multi_overlay_png" in outputs
        # Cluster signature heatmap requires CLR-Δ columns + >=2 clusters
        assert "cluster_signature_png" in outputs
        # Drivers should be computed (synthetic two-group data)
        assert "corpus_cluster_drivers" in outputs
        # Focus-genome locator
        assert "focus_G000_png" in outputs
