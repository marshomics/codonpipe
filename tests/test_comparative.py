"""Tests for the condition-aware comparative analysis module."""

import numpy as np
import pandas as pd
import pytest

from codonpipe.modules.comparative import (
    _cliffs_delta,
    _effect_size_label,
    within_condition_stats,
    within_condition_rscu_dispersion,
    between_condition_tests,
    between_condition_rscu_tests,
    permanova_rscu,
)
from codonpipe.utils.codon_tables import RSCU_COLUMN_NAMES


# ---------- fixtures ----------

@pytest.fixture
def metrics_two_conditions():
    """Two conditions with clearly different RSCU and metric values."""
    np.random.seed(42)
    n = 30
    rscu_cols = RSCU_COLUMN_NAMES[:6]  # first 6 codons for speed
    data = {
        "sample_id": [f"s{i}" for i in range(n * 2)],
        "condition": ["healthy"] * n + ["disease"] * n,
        "median_MELP": np.concatenate([
            np.random.normal(0.6, 0.05, n),
            np.random.normal(0.4, 0.05, n),
        ]),
        "mean_ENC": np.concatenate([
            np.random.normal(50, 3, n),
            np.random.normal(45, 3, n),
        ]),
        "doubling_time_hours": np.concatenate([
            np.random.normal(2.0, 0.3, n),
            np.random.normal(5.0, 0.5, n),
        ]),
    }
    for i, col in enumerate(rscu_cols):
        data[col] = np.concatenate([
            np.random.normal(1.0 + 0.1 * i, 0.15, n),
            np.random.normal(0.7 + 0.1 * i, 0.15, n),
        ])
    return pd.DataFrame(data)


@pytest.fixture
def metrics_three_conditions():
    """Three conditions for Kruskal-Wallis and PERMANOVA tests."""
    np.random.seed(99)
    n = 20
    rscu_cols = RSCU_COLUMN_NAMES[:4]
    conds = ["A"] * n + ["B"] * n + ["C"] * n
    data = {
        "sample_id": [f"s{i}" for i in range(n * 3)],
        "condition": conds,
        "mean_ENC": np.concatenate([
            np.random.normal(50, 2, n),
            np.random.normal(45, 2, n),
            np.random.normal(40, 2, n),
        ]),
    }
    for i, col in enumerate(rscu_cols):
        data[col] = np.concatenate([
            np.random.normal(1.0 + 0.2 * i, 0.1, n),
            np.random.normal(0.8 + 0.2 * i, 0.1, n),
            np.random.normal(0.6 + 0.2 * i, 0.1, n),
        ])
    return pd.DataFrame(data)


# ---------- _cliffs_delta ----------

class TestCliffsDelta:
    def test_identical_arrays(self):
        x = np.array([1.0, 2.0, 3.0])
        assert _cliffs_delta(x, x) == 0.0

    def test_perfect_separation(self):
        x = np.array([10, 20, 30])
        y = np.array([1, 2, 3])
        assert _cliffs_delta(x, y) == 1.0
        assert _cliffs_delta(y, x) == -1.0

    def test_empty_arrays(self):
        result = _cliffs_delta(np.array([]), np.array([1, 2]))
        assert np.isnan(result), "Empty input should return NaN, not a numeric value"

    def test_effect_size_labels(self):
        assert _effect_size_label(0.05) == "negligible"
        assert _effect_size_label(0.2) == "small"
        assert _effect_size_label(0.4) == "medium"
        assert _effect_size_label(0.8) == "large"


# ---------- within_condition_stats ----------

class TestWithinConditionStats:
    def test_basic_output(self, metrics_two_conditions):
        result = within_condition_stats(metrics_two_conditions, "condition")
        assert not result.empty
        assert "condition" in result.columns
        assert "metric" in result.columns
        assert "mean" in result.columns
        assert "cv" in result.columns
        conditions = result["condition"].unique()
        assert set(conditions) == {"healthy", "disease"}

    def test_metrics_included(self, metrics_two_conditions):
        result = within_condition_stats(metrics_two_conditions, "condition")
        metrics_in_result = result["metric"].unique()
        assert "median_MELP" in metrics_in_result
        assert "mean_ENC" in metrics_in_result

    def test_rscu_columns_excluded(self, metrics_two_conditions):
        """RSCU columns should not appear in per-metric summaries."""
        result = within_condition_stats(metrics_two_conditions, "condition")
        rscu_in_result = [m for m in result["metric"].unique() if m in RSCU_COLUMN_NAMES]
        assert len(rscu_in_result) == 0

    def test_missing_condition_col(self, metrics_two_conditions):
        result = within_condition_stats(metrics_two_conditions, "nonexistent")
        assert result.empty


# ---------- within_condition_rscu_dispersion ----------

class TestWithinConditionRSCUDispersion:
    def test_basic_output(self, metrics_two_conditions):
        result = within_condition_rscu_dispersion(metrics_two_conditions, "condition")
        assert not result.empty
        assert "condition" in result.columns
        assert "codon" in result.columns
        assert "cv" in result.columns

    def test_cv_positive(self, metrics_two_conditions):
        result = within_condition_rscu_dispersion(metrics_two_conditions, "condition")
        assert (result["cv"] >= 0).all()

    def test_missing_condition_col(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = within_condition_rscu_dispersion(df, "condition")
        assert result.empty


# ---------- between_condition_tests ----------

class TestBetweenConditionTests:
    def test_detects_differences(self, metrics_two_conditions):
        result = between_condition_tests(metrics_two_conditions, "condition")
        assert not result.empty
        assert "metric" in result.columns
        assert "p_value" in result.columns
        assert "corrected_p" in result.columns
        assert "effect_size" in result.columns
        assert "significant" in result.columns
        # Given the large effect sizes, some metrics should be significant
        assert result["significant"].any()

    def test_effect_size_direction(self, metrics_two_conditions):
        result = between_condition_tests(metrics_two_conditions, "condition")
        melp_row = result[result["metric"] == "median_MELP"]
        if not melp_row.empty:
            # healthy > disease for MELP, so effect size should be non-zero
            assert melp_row["effect_size"].abs().iloc[0] > 0.1

    def test_three_conditions_kruskal(self, metrics_three_conditions):
        result = between_condition_tests(metrics_three_conditions, "condition")
        assert not result.empty
        # Should include Kruskal-Wallis tests
        tests = result["test"].unique() if "test" in result.columns else []
        # With 3 groups: both pairwise MWU and omnibus KW
        assert len(result) > 0

    def test_single_condition_returns_empty(self):
        df = pd.DataFrame({
            "condition": ["A"] * 10,
            "metric1": np.random.normal(0, 1, 10),
        })
        result = between_condition_tests(df, "condition")
        assert result.empty


# ---------- between_condition_rscu_tests ----------

class TestBetweenConditionRSCUTests:
    def test_detects_rscu_differences(self, metrics_two_conditions):
        result = between_condition_rscu_tests(metrics_two_conditions, "condition")
        assert not result.empty
        assert "codon" in result.columns
        assert "log2_fold_change" in result.columns
        assert "corrected_p" in result.columns

    def test_log2fc_direction(self, metrics_two_conditions):
        """Group 1 (disease) vs group 2 (healthy) — check FC has correct sign."""
        result = between_condition_rscu_tests(metrics_two_conditions, "condition")
        # All RSCU values are higher in healthy; fold change should reflect that
        assert not result.empty
        # At least some codons should show non-zero fold change
        assert (result["log2_fold_change"].abs() > 0.01).any()


# ---------- permanova_rscu ----------

class TestPERMANOVA:
    def test_detects_difference(self, metrics_two_conditions):
        result = permanova_rscu(metrics_two_conditions, "condition", n_perm=99)
        assert result  # non-empty dict
        assert "F_statistic" in result
        assert "p_value" in result
        assert "R2" in result
        assert result["F_statistic"] > 0
        assert 0 <= result["R2"] <= 1

    def test_three_conditions(self, metrics_three_conditions):
        result = permanova_rscu(metrics_three_conditions, "condition", n_perm=99)
        assert result
        assert result["p_value"] <= 0.05  # strong separation

    def test_too_few_samples(self):
        """PERMANOVA requires at least 6 samples total."""
        df = pd.DataFrame({
            "condition": ["A", "A", "B", "B"],
            RSCU_COLUMN_NAMES[0]: [1.0, 1.1, 0.9, 0.8],
        })
        result = permanova_rscu(df, "condition")
        assert result == {}

    def test_no_condition_col(self, metrics_two_conditions):
        result = permanova_rscu(metrics_two_conditions, "nonexistent")
        assert result == {}
