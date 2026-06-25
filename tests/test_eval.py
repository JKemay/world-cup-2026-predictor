"""Tests for footy/evaluate/backtest.py — scoring rules, score(), naive_baseline()."""

import numpy as np
import pandas as pd
import pytest

from footy.evaluate.backtest import actual_outcome, naive_baseline, rps, score

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
