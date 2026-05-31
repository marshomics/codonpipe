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
