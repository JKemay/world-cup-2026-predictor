"""Tests for footy/evaluate/backtest.py — scoring rules, score(), naive_baseline()."""

import numpy as np
import pandas as pd
import pytest

from footy.evaluate.backtest import (
    actual_outcome,
    bootstrap_ci,
    naive_baseline,
    paired_bootstrap,
    per_match_rps,
    rps,
    score,
)

# ---------------------------------------------------------------------------
# actual_outcome
# ---------------------------------------------------------------------------

class TestActualOutcome:
    def test_home_win(self):
        assert actual_outcome(3, 1) == 0

    def test_draw(self):
        assert actual_outcome(1, 1) == 1
        assert actual_outcome(0, 0) == 1

    def test_away_win(self):
        assert actual_outcome(0, 2) == 2

    def test_one_goal_difference(self):
        assert actual_outcome(2, 1) == 0
        assert actual_outcome(1, 2) == 2

    def test_large_scoreline(self):
        assert actual_outcome(7, 0) == 0
        assert actual_outcome(0, 7) == 2


# ---------------------------------------------------------------------------
# rps — Ranked Probability Score
# ---------------------------------------------------------------------------

class TestRPS:
    def test_perfect_home_prediction_gives_zero(self):
        """A perfectly confident correct prediction should give RPS = 0."""
        r = rps(np.array([1.0, 0.0, 0.0]), outcome=0)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_perfect_draw_prediction_gives_zero(self):
        r = rps(np.array([0.0, 1.0, 0.0]), outcome=1)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_perfect_away_prediction_gives_zero(self):
        r = rps(np.array([0.0, 0.0, 1.0]), outcome=2)
        assert r == pytest.approx(0.0, abs=1e-9)

    def test_wrong_confident_home_when_away_wins(self):
        """Confidently predicting home when away wins is near-maximum error."""
        r = rps(np.array([1.0, 0.0, 0.0]), outcome=2)
        # Maximum possible RPS for 3 outcomes = (1^2 + 1^2)/2 = 1.0
        assert r == pytest.approx(1.0, abs=1e-9)

    def test_moderate_prediction_between_zero_and_one(self):
        """An uncertain prediction should give RPS strictly between 0 and 1."""
        r = rps(np.array([0.5, 0.3, 0.2]), outcome=0)
        assert 0.0 < r < 1.0

    def test_closer_prediction_lower_rps(self):
        """Predicting closer to the actual outcome gives lower RPS."""
        # Actual: home win (0)
        good = rps(np.array([0.7, 0.2, 0.1]), outcome=0)
        bad = rps(np.array([0.1, 0.2, 0.7]), outcome=0)
        assert good < bad

    def test_non_negative(self):
        """RPS is always >= 0."""
        probs = np.array([0.4, 0.35, 0.25])
        for outcome in [0, 1, 2]:
            assert rps(probs, outcome) >= 0.0

    def test_uniform_prediction_rps_value(self):
        """With uniform probs [1/3, 1/3, 1/3], RPS for home win = 1/3."""
        p = np.array([1 / 3, 1 / 3, 1 / 3])
        r = rps(p, outcome=0)
        # cumulative: [1/3, 2/3, 1], vs [1, 1, 1] → ((2/3)^2 + (1/3)^2) / 2 = 5/18
        expected = ((1 - 1/3)**2 + (1 - 2/3)**2) / 2
        assert r == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def _make_perfect(self):
        preds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        actuals = np.array([0, 1, 2])
        return preds, actuals

    def _make_all_wrong(self):
        preds = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        actuals = np.array([0, 1, 2])
        return preds, actuals

    def test_returns_required_keys(self):
        preds, actuals = self._make_perfect()
        s = score(preds, actuals)
        for key in ("n", "log_loss", "rps", "accuracy"):
            assert key in s

    def test_n_matches_input_length(self):
        preds = np.tile([1/3, 1/3, 1/3], (5, 1))
        actuals = np.array([0, 1, 2, 0, 1])
        s = score(preds, actuals)
        assert s["n"] == 5

    def test_accuracy_in_unit_interval(self):
        preds = np.tile([0.4, 0.3, 0.3], (4, 1))
        actuals = np.array([0, 0, 1, 2])
        s = score(preds, actuals)
        assert 0.0 <= s["accuracy"] <= 1.0

    def test_rps_in_unit_interval(self):
        preds = np.tile([0.4, 0.3, 0.3], (4, 1))
        actuals = np.array([0, 1, 2, 0])
        s = score(preds, actuals)
        assert 0.0 <= s["rps"] <= 1.0

    def test_perfect_accuracy_equals_one(self):
        preds, actuals = self._make_perfect()
        s = score(preds, actuals)
        assert s["accuracy"] == pytest.approx(1.0)

    def test_all_wrong_accuracy_equals_zero(self):
        preds, actuals = self._make_all_wrong()
        s = score(preds, actuals)
        assert s["accuracy"] == pytest.approx(0.0)

    def test_log_loss_positive(self):
        preds = np.tile([1/3, 1/3, 1/3], (3, 1))
        actuals = np.array([0, 1, 2])
        s = score(preds, actuals)
        assert s["log_loss"] > 0.0


# ---------------------------------------------------------------------------
# naive_baseline
# ---------------------------------------------------------------------------

class TestNaiveBaseline:
    @pytest.fixture
    def tiny_matches(self):
        return pd.DataFrame({
            "home":       ["A", "B", "C", "A", "B"],
            "away":       ["B", "C", "A", "C", "A"],
            "home_goals": [2, 0, 1, 3, 1],
            "away_goals": [1, 0, 1, 0, 2],
        })

    def test_returns_two_arrays(self, tiny_matches):
        preds, actuals = naive_baseline(tiny_matches)
        assert isinstance(preds, np.ndarray)
        assert isinstance(actuals, np.ndarray)

    def test_preds_shape(self, tiny_matches):
        preds, actuals = naive_baseline(tiny_matches)
        assert preds.shape == (len(tiny_matches), 3)
        assert actuals.shape == (len(tiny_matches),)

    def test_each_row_sums_to_one(self, tiny_matches):
        preds, _ = naive_baseline(tiny_matches)
        row_sums = preds.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_all_rows_identical(self, tiny_matches):
        """naive_baseline emits the same base-rate vector for every match."""
        preds, _ = naive_baseline(tiny_matches)
        for i in range(1, len(preds)):
            np.testing.assert_array_equal(preds[0], preds[i])

    def test_probs_non_negative(self, tiny_matches):
        preds, _ = naive_baseline(tiny_matches)
        assert (preds >= 0).all()

    def test_actuals_are_valid_outcomes(self, tiny_matches):
        _, actuals = naive_baseline(tiny_matches)
        assert set(actuals).issubset({0, 1, 2})


# ---------------------------------------------------------------------------
# per_match_rps
# ---------------------------------------------------------------------------

class TestPerMatchRPS:
    @pytest.fixture
    def sample_preds_actuals(self):
        rng = np.random.default_rng(42)
        n = 20
        raw = rng.dirichlet(np.ones(3), size=n)
        preds = raw / raw.sum(axis=1, keepdims=True)
        actuals = rng.integers(0, 3, size=n)
        return preds, actuals

    def test_length_equals_n(self, sample_preds_actuals):
        preds, actuals = sample_preds_actuals
        result = per_match_rps(preds, actuals)
        assert len(result) == len(actuals)

    def test_each_value_in_unit_interval(self, sample_preds_actuals):
        preds, actuals = sample_preds_actuals
        result = per_match_rps(preds, actuals)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_mean_equals_score_rps(self, sample_preds_actuals):
        preds, actuals = sample_preds_actuals
        result = per_match_rps(preds, actuals)
        s = score(preds, actuals)
        assert float(result.mean()) == pytest.approx(s["rps"], rel=1e-9)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    @pytest.fixture
    def known_values(self):
        rng = np.random.default_rng(7)
        return rng.uniform(0.1, 0.4, size=50)

    def test_mean_is_correct(self, known_values):
        ci = bootstrap_ci(known_values, n_boot=5_000, seed=0)
        assert ci["mean"] == pytest.approx(float(known_values.mean()), rel=1e-9)

    def test_lo_le_mean_le_hi(self, known_values):
        ci = bootstrap_ci(known_values, n_boot=5_000, seed=0)
        assert ci["lo"] <= ci["mean"] <= ci["hi"]

    def test_wider_ci_at_higher_confidence(self, known_values):
        ci95 = bootstrap_ci(known_values, n_boot=5_000, ci=0.95, seed=0)
        ci99 = bootstrap_ci(known_values, n_boot=5_000, ci=0.99, seed=0)
        # 99% CI must be at least as wide as the 95% CI
        assert ci99["lo"] <= ci95["lo"]
        assert ci99["hi"] >= ci95["hi"]

    def test_deterministic_with_same_seed(self, known_values):
        ci_a = bootstrap_ci(known_values, n_boot=1_000, seed=99)
        ci_b = bootstrap_ci(known_values, n_boot=1_000, seed=99)
        assert ci_a == ci_b

    def test_different_seeds_may_differ(self, known_values):
        ci_a = bootstrap_ci(known_values, n_boot=1_000, seed=0)
        ci_b = bootstrap_ci(known_values, n_boot=1_000, seed=1)
        # mean is always identical; only lo/hi may shift
        assert ci_a["mean"] == ci_b["mean"]


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------

class TestPairedBootstrap:
    def test_identical_inputs_mean_diff_zero(self):
        rng = np.random.default_rng(5)
        arr = rng.uniform(0.1, 0.5, size=30)
        result = paired_bootstrap(arr, arr, n_boot=2_000, seed=0)
        assert result["mean_diff"] == pytest.approx(0.0, abs=1e-12)

    def test_identical_inputs_p_a_better_in_unit_interval(self):
        rng = np.random.default_rng(5)
        arr = rng.uniform(0.1, 0.5, size=30)
        result = paired_bootstrap(arr, arr, n_boot=2_000, seed=0)
        assert 0.0 <= result["p_a_better"] <= 1.0

    def test_identical_inputs_deterministic(self):
        rng = np.random.default_rng(5)
        arr = rng.uniform(0.1, 0.5, size=30)
        r1 = paired_bootstrap(arr, arr, n_boot=500, seed=42)
        r2 = paired_bootstrap(arr, arr, n_boot=500, seed=42)
        assert r1 == r2

    def test_a_uniformly_smaller_p_a_better_one(self):
        """When rps_a is strictly lower for every match, P(a better) == 1.0."""
        rng = np.random.default_rng(3)
        rps_b = rng.uniform(0.3, 0.5, size=40)
        rps_a = rps_b - 0.1  # a is always 0.1 better
        result = paired_bootstrap(rps_a, rps_b, n_boot=2_000, seed=0)
        assert result["mean_diff"] == pytest.approx(-0.1, rel=1e-9)
        assert result["p_a_better"] == pytest.approx(1.0)

    def test_a_uniformly_larger_p_a_better_zero(self):
        """When rps_a is strictly higher for every match, P(a better) == 0.0."""
        rng = np.random.default_rng(3)
        rps_b = rng.uniform(0.1, 0.3, size=40)
        rps_a = rps_b + 0.1
        result = paired_bootstrap(rps_a, rps_b, n_boot=2_000, seed=0)
        assert result["p_a_better"] == pytest.approx(0.0)

    def test_ci_straddles_zero_when_no_real_difference(self):
        """When the two series are independent noise, the CI should include 0."""
        rng = np.random.default_rng(17)
        a = rng.uniform(0.2, 0.4, size=100)
        b = rng.uniform(0.2, 0.4, size=100)
        result = paired_bootstrap(a, b, n_boot=5_000, seed=0)
        assert result["lo"] < 0 < result["hi"]
