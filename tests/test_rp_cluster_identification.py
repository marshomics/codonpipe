"""RP multi-cluster identification + stability-validated anchor selection.

When a genome's ribosomal-protein genes form two or more distinct populations
in COA space, the bootstrap path must (a) decide whether each candidate split
is a real cluster or sampling noise, and (b) anchor the optimized core on a
single, biologically sensible cluster rather than pooling all RP genes.

Validation uses clusterwise Jaccard bootstrap (Hennig 2007): a candidate is a
genuine cluster only if its mean maximum-Jaccard to the best-matching bootstrap
cluster is >= 0.75; below 0.60 it is treated as dissolved (noise). These tests
lock in that two well-separated RP populations both validate, an outlier pocket
dissolves, and the anchor is chosen by RSCU similarity (count as tie-break).
"""

import numpy as np

from codonpipe.modules import cluster_stability as CS


def _two_rp_populations(n_a=40, n_b=24, seed=0):
    """Two tight, well-separated RP clusters in a 3-D COA-like space."""
    rng = np.random.default_rng(seed)
    A = rng.normal([1.2, 1.2, 0.0], 0.05, (n_a, 3))
    B = rng.normal([-1.2, -1.0, 0.3], 0.05, (n_b, 3))
    X = np.vstack([A, B])
    ids = [f"a{i}" for i in range(n_a)] + [f"b{i}" for i in range(n_b)]
    return X, ids, set(f"a{i}" for i in range(n_a)), set(f"b{i}" for i in range(n_b))


def _one_population_plus_outliers(n_core=45, n_out=8, seed=1):
    """A single tight RP cluster plus a few scattered outliers (no 2nd cluster)."""
    rng = np.random.default_rng(seed)
    core = rng.normal([1.0, 1.0, 0.0], 0.05, (n_core, 3))
    out = rng.uniform(-2.0, 2.0, (n_out, 3))
    X = np.vstack([core, out])
    ids = [f"c{i}" for i in range(n_core)] + [f"o{i}" for i in range(n_out)]
    return X, ids


class TestClusterValidation:
    def test_two_real_clusters_both_validate(self):
        X, ids, A, B = _two_rp_populations()
        v = CS._validate_rp_subclusters(X, ids, n_boot=60, seed=0)
        stable = [c for c in v if c["stability_label"] == "stable"]
        assert len(stable) == 2, f"expected 2 stable clusters, got {[c['stability_label'] for c in v]}"
        assert all(c["mean_jaccard"] >= CS._CLUSTER_STABLE_JACCARD for c in stable)

    def test_outliers_dissolve_single_population(self):
        X, ids = _one_population_plus_outliers()
        v = CS._validate_rp_subclusters(X, ids, n_boot=60, seed=0)
        stable = [c for c in v if c["stability_label"] == "stable"]
        # The core is one stable cluster; any outlier pocket must not validate
        # as a second genuine cluster.
        assert len(stable) <= 1, f"outliers spuriously validated: {[(c['n'], c['mean_jaccard']) for c in v]}"

    def test_jaccard_index_basic(self):
        assert CS._jaccard_index({1, 2, 3}, {2, 3, 4}) == 2 / 4
        assert CS._jaccard_index(set(), set()) == 0.0
        assert CS._jaccard_index({1}, {1}) == 1.0


class TestAnchorSelection:
    def test_anchor_prefers_rscu_consensus_over_count(self, monkeypatch):
        # Two stable clusters: a SMALLER one closer to the RP consensus and a
        # LARGER one further away. Anchor must be the smaller, closer cluster.
        small = {"gene_ids": [f"s{i}" for i in range(20)], "n": 20, "stability_label": "stable"}
        large = {"gene_ids": [f"l{i}" for i in range(40)], "n": 40, "stability_label": "stable"}

        # Stub the RSCU computation so 'small' has higher cosine to consensus.
        import pandas as pd

        def fake_cluster_rscu(ffn, ids, gene_weights=None):
            tag = next(iter(ids))
            val = 1.0 if tag.startswith("s") else 0.0
            return pd.Series({"Phe-UUU": val, "Phe-UUC": 1.0 - val})

        monkeypatch.setattr(CS, "_compute_cluster_rscu", fake_cluster_rscu)
        monkeypatch.setattr(CS, "RSCU_COLUMN_NAMES", ["Phe-UUU", "Phe-UUC"])
        rp_rscu = pd.DataFrame({"Phe-UUU": [1.0], "Phe-UUC": [0.0]})

        class _P:
            def exists(self):
                return True

        anchor, reason = CS._select_rp_anchor([large, small], _P(), rp_rscu)
        assert anchor is small, "anchor should be the RSCU-consensus-closest cluster, not the largest"
        assert reason == "rscu_consensus_similarity"

    def test_anchor_fallback_largest_when_no_rscu(self):
        small = {"gene_ids": [f"s{i}" for i in range(20)], "n": 20, "stability_label": "stable"}
        large = {"gene_ids": [f"l{i}" for i in range(40)], "n": 40, "stability_label": "stable"}
        anchor, reason = CS._select_rp_anchor([large, small], None, None)
        assert anchor is large
        assert reason == "largest_stable_cluster_fallback"

    def test_dissolved_clusters_excluded_from_anchor(self):
        stable = {"gene_ids": [f"s{i}" for i in range(30)], "n": 30, "stability_label": "stable"}
        dead = {"gene_ids": [f"d{i}" for i in range(25)], "n": 25, "stability_label": "dissolved"}
        anchor, reason = CS._select_rp_anchor([dead, stable], None, None)
        assert anchor is stable
        assert reason == "single_validated_cluster"


class TestRscuRescue:
    """Borderline RP genes whose codon usage is as ribosomal-like as the
    weakest core member are rescued, but only within a distance sanity cap so
    compositionally-RP-but-geometrically-distant genes are not pulled in."""

    def _setup(self, monkeypatch, tmp_path):
        import pandas as pd
        from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES
        cols = list(RSCU_COLUMN_NAMES[:6])

        # RP consensus vector (all-ones over the used columns).
        rp_rscu_df = pd.DataFrame([{c: 1.0 for c in cols} for _ in range(5)])

        # Per-gene RSCU: core genes + an RP-like-but-far gene + an RP-unlike gene.
        rows = {
            # core RP genes, clearly RP-like
            "core1": {c: 1.0 for c in cols},
            "core2": {c: 0.95 for c in cols},
            "core3": {c: 0.90 for c in cols},   # weakest core member sets the bar
            # borderline RP gene: as RP-like as core3, just past the boundary
            "border": {c: 0.92 for c in cols},
            # RP-like in composition but geometrically far away
            "far": {c: 0.99 for c in cols},
            # RP gene whose codon usage has drifted (low cosine)
            "drift": {c: (1.0 if i == 0 else -1.0) for i, c in enumerate(cols)},
        }
        gene_rscu = pd.DataFrame(rows).T
        monkeypatch.setattr(CS, "_compute_per_gene_rscu",
                            lambda ffn, ids: gene_rscu.loc[[g for g in ids if g in gene_rscu.index]])

        ffn = tmp_path / "x.ffn"
        ffn.write_text(">core1\nATG\n")  # only needs to exist
        return rp_rscu_df, ffn

    def test_rescues_borderline_within_distance_cap(self, monkeypatch, tmp_path):
        rp_rscu_df, ffn = self._setup(monkeypatch, tmp_path)
        core = {"core1", "core2", "core3"}
        rp_ids = {"core1", "core2", "core3", "border", "far", "drift"}
        gene_ids = ["core1", "core2", "core3", "border", "far", "drift"]
        # boundary ~3.5; border just past it, far well beyond 2x cap, drift near.
        dist = [1.0, 2.0, 3.0, 4.0, 20.0, 3.5]
        rescued = CS._rescue_rp_by_rscu(core, rp_ids, ffn, rp_rscu_df,
                                        distances=dist, gene_ids=gene_ids, boundary=3.5)
        assert "border" in rescued          # RP-like + within distance cap
        assert "far" not in rescued         # RP-like but beyond the distance cap
        assert "drift" not in rescued       # close but not RP-like

    def test_distance_cap_excludes_far_rp_like_genes(self, monkeypatch, tmp_path):
        rp_rscu_df, ffn = self._setup(monkeypatch, tmp_path)
        core = {"core1", "core2", "core3"}
        rp_ids = {"core1", "core2", "core3", "far"}
        gene_ids = ["core1", "core2", "core3", "far"]
        dist = [1.0, 2.0, 3.0, 20.0]
        rescued = CS._rescue_rp_by_rscu(core, rp_ids, ffn, rp_rscu_df,
                                        distances=dist, gene_ids=gene_ids, boundary=3.5)
        assert rescued == set()

    def test_rescue_is_rp_only(self, monkeypatch, tmp_path):
        # A non-RP gene that happens to be RP-like must never be rescued, because
        # the candidate loop iterates only over rp_gene_ids.
        rp_rscu_df, ffn = self._setup(monkeypatch, tmp_path)
        core = {"core1", "core2", "core3"}
        rp_ids = {"core1", "core2", "core3"}     # 'border' is NOT an RP gene here
        gene_ids = ["core1", "core2", "core3", "border"]
        dist = [1.0, 2.0, 3.0, 4.0]
        rescued = CS._rescue_rp_by_rscu(core, rp_ids, ffn, rp_rscu_df,
                                        distances=dist, gene_ids=gene_ids, boundary=3.5)
        assert "border" not in rescued

    def test_no_ffn_returns_empty(self, monkeypatch, tmp_path):
        import pandas as pd
        rp_rscu_df, _ = self._setup(monkeypatch, tmp_path)
        rescued = CS._rescue_rp_by_rscu({"core1"}, {"core1", "border"}, None, rp_rscu_df)
        assert rescued == set()
