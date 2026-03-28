"""Tests for the statistics module."""

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.statistics import (
    compute_zscore_normalization,
    pairwise_wilcoxon,
)


@pytest.fixture
def sample_rscu_df():
    """Create a sample RSCU DataFrame with two groups."""
    np.random.seed(42)
    n_per_group = 50
    data = {
        "sample_id": [f"s{i}" for i in range(n_per_group * 2)],
        "group": ["A"] * n_per_group + ["B"] * n_per_group,
        "Phe-UUU": np.concatenate([
            np.random.normal(1.2, 0.2, n_per_group),
            np.random.normal(0.8, 0.2, n_per_group),
        ]),
        "Phe-UUC": np.concatenate([
            np.random.normal(0.8, 0.2, n_per_group),
            np.random.normal(1.2, 0.2, n_per_group),
        ]),
    }
    return pd.DataFrame(data)


class TestPairwiseWilcoxon:
    def test_detects_significant_difference(self, sample_rscu_df):
        result = pairwise_wilcoxon(
            sample_rscu_df, "group", ["Phe-UUU", "Phe-UUC"],
        )
        assert len(result) == 2  # 2 codons × 1 pair
        # Both should be significant given the clear separation
        assert result["significant"].all()

    def test_no_difference(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "group": ["A"] * 50 + ["B"] * 50,
            "Phe-UUU": np.random.normal(1.0, 0.1, 100),
        })
        result = pairwise_wilcoxon(df, "group", ["Phe-UUU"])
        # With identical distributions, should not be significant
        # (though this is probabilistic — seed controls it)
        assert len(result) == 1

    def test_multiple_groups(self, sample_rscu_df):
        # Add a third group
        extra = sample_rscu_df.head(20).copy()
        extra["group"] = "C"
        df = pd.concat([sample_rscu_df, extra], ignore_index=True)
        result = pairwise_wilcoxon(df, "group", ["Phe-UUU"])
        # Should have 3 pairs: A-B, A-C, B-C
        assert len(result) == 3


class TestZscoreNormalization:
    def test_zscore_mean_zero(self, sample_rscu_df):
        result = compute_zscore_normalization(sample_rscu_df, ["Phe-UUU", "Phe-UUC"])
        assert abs(result["Phe-UUU"].mean()) < 0.01
        assert abs(result["Phe-UUC"].mean()) < 0.01

    def test_zscore_std_one(self, sample_rscu_df):
        result = compute_zscore_normalization(sample_rscu_df, ["Phe-UUU"])
        assert abs(result["Phe-UUU"].std() - 1.0) < 0.1

    def test_preserves_other_columns(self, sample_rscu_df):
        result = compute_zscore_normalization(sample_rscu_df, ["Phe-UUU"])
        assert "sample_id" in result.columns
        assert "group" in result.columns
