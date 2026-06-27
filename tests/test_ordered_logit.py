"""Tests for footy/ratings/ordered_logit.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from footy.ratings.ordered_logit import build_features, fit_ordered_logit, predict_wdl_ol


def make_toy_matches() -> pd.DataFrame:
    """20 rows with varying teams and score diffs to give the model real signal."""
    # 10 home wins, 5 draws, 5 away wins across two different pairings so
    # elo_diff varies across rows and the model can learn draw-proneness.
    return pd.DataFrame({
        "match_id": [f"id_{i}" for i in range(20)],
        "home": (["TeamA"] * 10 + ["TeamC"] * 10),
        "away": (["TeamB"] * 10 + ["TeamD"] * 10),
        "home_goals": [2, 2, 1, 3, 2, 1, 1, 1, 0, 2, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
        "away_goals": [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2],
        "season_id": ["s1"] * 20,
    })


class FakeEloVarying:
    """Elo object with two team pairs at different rating gaps."""

    pre_match_ratings: dict = {}
    ratings: dict[str, float] = {
        "TeamA": 1550.0,  # strong home
        "TeamB": 1450.0,  # weaker away → big elo_diff
        "TeamC": 1500.0,  # even matchup
        "TeamD": 1500.0,  # even matchup → elo_diff ≈ 0, abs_diff ≈ 0
    }



class TestOrderedLogit:
    def _fit(self):
        matches = make_toy_matches()
        elo = FakeEloVarying()
        fifa_z = {"TeamA": 0.8, "TeamB": -0.5, "TeamC": 0.0, "TeamD": 0.0}
        X, y = build_features(matches, elo, fifa_z)
        model = fit_ordered_logit(X, y)
        return model, elo, fifa_z

    # ------------------------------------------------------------------
    # Shape / probability axioms
    # ------------------------------------------------------------------

    def test_shape_sums_to_one(self):
        model, _elo, _fifa_z = self._fit()
        x = np.array([[0.2, 0.3, 0.1]])
        p = predict_wdl_ol(model, x)
        assert p.shape == (3,)
        assert abs(p.sum() - 1.0) < 1e-8
        assert (p >= 0).all()

    def test_all_probs_positive(self):
        """Clipping ensures no exactly-zero probabilities."""
        model, _elo, _fifa_z = self._fit()
        x = np.array([[0.0, 0.0, 0.0]])
        p = predict_wdl_ol(model, x)
        assert (p > 0).all()

    # ------------------------------------------------------------------
    # Semantic / monotonicity checks
    # ------------------------------------------------------------------

    def test_draw_proneness(self):
        """Even matchup should have higher draw probability than lopsided one."""
        model, _elo, _fifa_z = self._fit()
        x_even = np.array([[0.0, 0.0, 0.0]])   # elo_diff=0, fifa_diff=0, abs=0
        x_lopsided = np.array([[1.5, 0.5, 1.5]])  # strong home favourite
        p_even = predict_wdl_ol(model, x_even)
        p_lop = predict_wdl_ol(model, x_lopsided)
        assert p_even[1] > p_lop[1], "Draw prob should be higher for even matchup"

    def test_monotone_home_win(self):
        """Increasing elo_diff should increase P(home win)."""
        model, _elo, _fifa_z = self._fit()
        x_low = np.array([[0.0, 0.0, 0.0]])
        x_high = np.array([[2.0, 0.0, 2.0]])
        p_low = predict_wdl_ol(model, x_low)
        p_high = predict_wdl_ol(model, x_high)
        assert p_high[0] > p_low[0]

    # ------------------------------------------------------------------
    # Correctness of class-column reindexing
    # ------------------------------------------------------------------

    def test_class_column_reindex(self):
        """Output always maps to [home, draw, away] regardless of model.classes_ order."""
        model, _elo, _fifa_z = self._fit()
        clf = model["clf"]
        x = np.array([[0.1, 0.0, 0.1]])
        p = predict_wdl_ol(model, x)
        assert p.shape == (3,)
        assert (p > 0).all()
        # classes_ must be a permutation of {0,1,2}
        assert set(clf.classes_) == {0, 1, 2}

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_determinism(self):
        model, _elo, _fifa_z = self._fit()
        x = np.array([[0.3, 0.1, 0.3]])
        p1 = predict_wdl_ol(model, x)
        p2 = predict_wdl_ol(model, x)
        np.testing.assert_array_equal(p1, p2)

    # ------------------------------------------------------------------
    # build_features
    # ------------------------------------------------------------------

    _FIFA_Z = {"TeamA": 0.8, "TeamB": -0.5, "TeamC": 0.0, "TeamD": 0.0}

    def test_build_features_shape(self):
        matches = make_toy_matches()
        elo = FakeEloVarying()
        X, y = build_features(matches, elo, self._FIFA_Z)
        assert X.shape == (20, 3)
        assert y.shape == (20,)

    def test_build_features_label_range(self):
        matches = make_toy_matches()
        elo = FakeEloVarying()
        _, y = build_features(matches, elo, self._FIFA_Z)
        assert set(y).issubset({0, 1, 2})

    def test_build_features_skips_nan_goals(self):
        matches = make_toy_matches().copy()
        matches.loc[0, "home_goals"] = float("nan")
        elo = FakeEloVarying()
        X, y = build_features(matches, elo, self._FIFA_Z)
        assert X.shape[0] == 19

    def test_build_features_pre_match_ratings_used(self):
        """When pre_match_ratings is populated, those values should be used."""
        matches = make_toy_matches().iloc[:1].copy()

        class EloWithPre:
            pre_match_ratings = {"id_0": (1600.0, 1400.0)}
            ratings = {"TeamA": 1550.0, "TeamB": 1450.0, "TeamC": 1500.0, "TeamD": 1500.0}

        elo_pre = EloWithPre()
        X_pre, _ = build_features(matches, elo_pre, self._FIFA_Z)
        X_post, _ = build_features(matches, FakeEloVarying(), self._FIFA_Z)
        # elo_diff from (1600, 1400) differs from (1550, 1450)
        assert X_pre[0, 0] != X_post[0, 0]

    def test_fit_ordered_logit_returns_dict(self):
        matches = make_toy_matches()
        elo = FakeEloVarying()
        X, y = build_features(matches, elo, self._FIFA_Z)
        model = fit_ordered_logit(X, y)
        assert "scaler" in model
        assert "clf" in model

    def test_predict_accepts_1d_input(self):
        """predict_wdl_ol should accept a bare (3,) row, not just (1,3)."""
        model, _elo, _fifa_z = self._fit()
        x_1d = np.array([0.2, 0.1, 0.2])
        p = predict_wdl_ol(model, x_1d)
        assert p.shape == (3,)
        assert abs(p.sum() - 1.0) < 1e-8
