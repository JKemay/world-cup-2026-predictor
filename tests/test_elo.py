"""Tests for footy/ratings/elo.py.

All tests are deterministic and require no network or data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.ratings.elo import (
    HOME_ADV,
    DrawParams,
    _goal_multiplier,
    elo_match_probs,
    expected_score,
    fit_elo,
    update_ratings,
)

# ---------------------------------------------------------------------------
# expected_score
# ---------------------------------------------------------------------------

class TestExpectedScore:
    def test_equal_ratings_zero_home_adv_returns_half(self):
        """When both teams have the same rating and home_adv=0, We should be 0.5."""
        we = expected_score(1500.0, 1500.0, home_adv=0.0)
        assert we == pytest.approx(0.5, abs=1e-9)

    def test_higher_home_rating_gives_we_above_half(self):
        """Home team with higher rating (no adv) should have We > 0.5."""
        we = expected_score(1600.0, 1500.0, home_adv=0.0)
        assert we > 0.5

    def test_lower_home_rating_gives_we_below_half(self):
        """Home team with lower rating (no adv) should have We < 0.5."""
        we = expected_score(1400.0, 1500.0, home_adv=0.0)
        assert we < 0.5

    def test_home_advantage_shifts_we_up(self):
        """Default HOME_ADV should push We above the zero-adv case for equal teams."""
        we_no_adv = expected_score(1500.0, 1500.0, home_adv=0.0)
        we_adv = expected_score(1500.0, 1500.0, home_adv=HOME_ADV)
        assert we_adv > we_no_adv

    def test_symmetry_zero_adv(self):
        """expected_score(R, S, 0) + expected_score(S, R, 0) == 1."""
        we_h = expected_score(1550.0, 1450.0, home_adv=0.0)
        we_a = expected_score(1450.0, 1550.0, home_adv=0.0)
        assert we_h + we_a == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# elo_match_probs
# ---------------------------------------------------------------------------

class TestEloMatchProbs:
    @pytest.fixture(autouse=True)
    def draw_params(self):
        return DrawParams(D0=0.28, DW=200.0)

    def test_probs_sum_to_one(self, draw_params):
        probs = elo_match_probs(1500.0, 1500.0, draw_params)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_probs_in_unit_interval(self, draw_params):
        probs = elo_match_probs(1500.0, 1500.0, draw_params)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_three_probabilities_returned(self, draw_params):
        probs = elo_match_probs(1600.0, 1400.0, draw_params)
        assert probs.shape == (3,)

    def test_big_favorite_has_high_home_prob(self, draw_params):
        """A team rated 300 points higher should win more often than it loses."""
        probs = elo_match_probs(1700.0, 1400.0, draw_params, home_adv=0.0)
        assert probs[0] > probs[2], "favourite must have p_home > p_away"

    def test_even_match_has_more_draw_than_lopsided(self, draw_params):
        """Draw probability peaks when teams are evenly matched."""
        probs_even = elo_match_probs(1500.0, 1500.0, draw_params, home_adv=0.0)
        probs_lopsided = elo_match_probs(1700.0, 1300.0, draw_params, home_adv=0.0)
        assert probs_even[1] > probs_lopsided[1]


# ---------------------------------------------------------------------------
# update_ratings
# ---------------------------------------------------------------------------

class TestUpdateRatings:
    def test_home_win_increases_home_rating(self):
        r_h, r_a = 1500.0, 1500.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=2, away_goals=0)
        assert new_h > r_h

    def test_home_win_decreases_away_rating(self):
        r_h, r_a = 1500.0, 1500.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=2, away_goals=0)
        assert new_a < r_a

    def test_zero_sum_home_win(self):
        """Total rating is conserved: delta_home + delta_away == 0."""
        r_h, r_a = 1520.0, 1480.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=1, away_goals=0)
        assert (new_h + new_a) == pytest.approx(r_h + r_a, abs=1e-9)

    def test_zero_sum_draw(self):
        r_h, r_a = 1600.0, 1400.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=1, away_goals=1)
        assert (new_h + new_a) == pytest.approx(r_h + r_a, abs=1e-9)

    def test_zero_sum_away_win(self):
        r_h, r_a = 1500.0, 1500.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=0, away_goals=3)
        assert (new_h + new_a) == pytest.approx(r_h + r_a, abs=1e-9)

    def test_upset_gives_larger_delta(self):
        """An upset (low-rated team wins) should give a larger rating change."""
        # Strong home team loses (upset) vs expected home win
        r_h, r_a = 1700.0, 1300.0
        new_h_upset, _ = update_ratings(r_h, r_a, home_goals=0, away_goals=1)
        new_h_exp, _ = update_ratings(r_h, r_a, home_goals=2, away_goals=0)
        # |drop| on upset > |gain| on expected win
        assert abs(new_h_upset - r_h) > abs(new_h_exp - r_h)

    def test_away_win_decreases_home_rating(self):
        r_h, r_a = 1500.0, 1500.0
        new_h, new_a = update_ratings(r_h, r_a, home_goals=0, away_goals=2)
        assert new_h < r_h
        assert new_a > r_a


# ---------------------------------------------------------------------------
# goal multiplier
# ---------------------------------------------------------------------------

class TestGoalMultiplier:
    def test_one_goal_diff_returns_one(self):
        assert _goal_multiplier(1) == pytest.approx(1.0)
        assert _goal_multiplier(-1) == pytest.approx(1.0)

    def test_zero_goal_diff_returns_one(self):
        assert _goal_multiplier(0) == pytest.approx(1.0)

    def test_two_goal_diff_returns_1_5(self):
        assert _goal_multiplier(2) == pytest.approx(1.5)
        assert _goal_multiplier(-2) == pytest.approx(1.5)

    def test_three_goal_diff(self):
        assert _goal_multiplier(3) == pytest.approx((11 + 3) / 8.0)

    def test_large_diff_increases(self):
        assert _goal_multiplier(5) > _goal_multiplier(3)


# ---------------------------------------------------------------------------
# fit_elo — integration test on synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_matches():
    """Small deterministic dataset: 10 matches, 4 teams."""
    rng = np.random.default_rng(42)
    home, away, hg, ag = [], [], [], []
    pairs = [
        ("Alpha", "Beta"), ("Beta", "Gamma"), ("Gamma", "Delta"),
        ("Alpha", "Gamma"), ("Beta", "Delta"), ("Alpha", "Delta"),
        ("Beta", "Alpha"), ("Gamma", "Beta"), ("Delta", "Alpha"),
        ("Alpha", "Beta"),
    ]
    for h, a in pairs:
        home.append(h)
        away.append(a)
        hg.append(int(rng.integers(0, 4)))
        ag.append(int(rng.integers(0, 4)))

    dates = [f"2024-01-{i+1:02d}T12:00:00+00:00" for i in range(len(pairs))]
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(len(pairs))],
        "date": dates,
        "season_id": ["s1"] * len(pairs),
        "home": home,
        "away": away,
        "home_goals": hg,
        "away_goals": ag,
    })


class TestFitElo:
    def test_all_teams_have_ratings(self, synthetic_matches):
        elo = fit_elo(synthetic_matches)
        for t in ["Alpha", "Beta", "Gamma", "Delta"]:
            assert t in elo.ratings

    def test_pre_match_ratings_stored_for_all_matches(self, synthetic_matches):
        elo = fit_elo(synthetic_matches)
        for mid in synthetic_matches["match_id"]:
            assert mid in elo.pre_match_ratings

    def test_pre_match_ratings_are_tuples_of_two(self, synthetic_matches):
        elo = fit_elo(synthetic_matches)
        for mid, (r_h, r_a) in elo.pre_match_ratings.items():
            assert isinstance(r_h, float)
            assert isinstance(r_a, float)

    def test_draw_params_fitted(self, synthetic_matches):
        elo = fit_elo(synthetic_matches)
        assert isinstance(elo.draw_params, DrawParams)
        assert 0 < elo.draw_params.D0 < 1
        assert elo.draw_params.DW > 0

    def test_predict_wdl_shape(self, synthetic_matches):
        elo = fit_elo(synthetic_matches)
        r_h, r_a = elo.pre_match_ratings["m0"]
        from footy.ratings.elo import predict_wdl
        probs = predict_wdl(elo, r_h, r_a)
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
