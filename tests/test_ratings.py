"""Tests for footy/ratings/dixon_coles.py — tau, grid_summary, DixonColesRatings."""

import numpy as np
import pandas as pd
import pytest

from footy.ratings.dixon_coles import DixonColesRatings, grid_summary, tau

# ---------------------------------------------------------------------------
# tau — Dixon-Coles low-score correction
# ---------------------------------------------------------------------------

class TestTau:
    def test_high_score_returns_one(self):
        """tau returns 1.0 when h>1 or a>1 (no correction for high-scoring cells)."""
        assert tau(2, 3, lam=1.5, mu=1.2, rho=0.1) == pytest.approx(1.0)
        assert tau(2, 0, lam=1.5, mu=1.2, rho=0.1) == pytest.approx(1.0)
        assert tau(0, 2, lam=1.5, mu=1.2, rho=0.1) == pytest.approx(1.0)
        assert tau(5, 5, lam=1.5, mu=1.2, rho=0.1) == pytest.approx(1.0)

    def test_zero_zero_with_positive_rho_less_than_one(self):
        """tau(0,0,...) < 1.0 when rho > 0 (draws slightly suppressed)."""
        lam, mu, rho = 1.5, 1.2, 0.1
        result = tau(0, 0, lam, mu, rho)
        assert result < 1.0
        assert result == pytest.approx(1.0 - lam * mu * rho)

    def test_zero_one_formula(self):
        """tau(0,1,...) = 1 + lam*rho."""
        lam, mu, rho = 1.3, 0.9, 0.05
        assert tau(0, 1, lam, mu, rho) == pytest.approx(1.0 + lam * rho)

    def test_one_zero_formula(self):
        """tau(1,0,...) = 1 + mu*rho."""
        lam, mu, rho = 1.3, 0.9, 0.05
        assert tau(1, 0, lam, mu, rho) == pytest.approx(1.0 + mu * rho)

    def test_one_one_formula(self):
        """tau(1,1,...) = 1 - rho."""
        rho = 0.07
        assert tau(1, 1, 1.5, 1.2, rho) == pytest.approx(1.0 - rho)

    def test_zero_rho_always_one(self):
        """When rho=0 all corrections are exactly 1.0."""
        for h in range(3):
            for a in range(3):
                assert tau(h, a, 1.5, 1.2, 0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Synthetic match dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_matches():
    """
    6 matches, 4 teams. "Arsenal" (strong) vs "Brentford" (weak), with
    "Chelsea" and "Fulham" as mid-table sides. Arsenal scores 3+ xG per game.
    """
    return pd.DataFrame({
        "home":       ["Arsenal", "Arsenal", "Chelsea",  "Brentford", "Fulham",    "Arsenal"],
        "away":       ["Brentford","Chelsea","Brentford","Fulham",    "Arsenal",   "Fulham"],
        "home_goals": [3, 2, 1, 0, 1, 4],
        "away_goals": [0, 1, 0, 1, 2, 0],
        "home_xg":    [3.2, 2.4, 1.1, 0.6, 1.0, 3.8],
        "away_xg":    [0.4, 0.9, 0.5, 0.7, 2.1, 0.5],
    })


@pytest.fixture(scope="module")
def fitted_model(synthetic_matches):
    model = DixonColesRatings(alpha=0.5, response="xg", fifa=None, team_effects=True)
    model.fit(synthetic_matches)
    return model


# ---------------------------------------------------------------------------
# DixonColesRatings — scoreline_grid
# ---------------------------------------------------------------------------

class TestScorelineGrid:
    def test_grid_sums_to_one(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Arsenal", "Brentford")
        assert np.isclose(grid.sum(), 1.0, atol=1e-6), f"grid.sum()={grid.sum()}"

    def test_grid_all_non_negative(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Arsenal", "Brentford")
        assert (grid >= 0).all()

    def test_grid_shape(self, fitted_model):
        """Default max_goals=6 → 7x7 grid."""
        grid, _, _ = fitted_model.scoreline_grid("Chelsea", "Fulham")
        assert grid.shape == (7, 7)

    def test_lam_mu_positive(self, fitted_model):
        _, lam, mu = fitted_model.scoreline_grid("Arsenal", "Chelsea")
        assert lam > 0
        assert mu > 0


# ---------------------------------------------------------------------------
# DixonColesRatings — grid_summary
# ---------------------------------------------------------------------------

class TestGridSummary:
    def test_probabilities_sum_to_one(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Arsenal", "Brentford")
        s = grid_summary(grid)
        total = s["home_win"] + s["draw"] + s["away_win"]
        assert np.isclose(total, 1.0, atol=1e-6), f"probs sum={total}"

    def test_keys_present(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Arsenal", "Brentford")
        s = grid_summary(grid)
        for key in ("home_win", "draw", "away_win", "top_score", "top_prob"):
            assert key in s

    def test_top_prob_consistent(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Arsenal", "Brentford")
        s = grid_summary(grid)
        assert s["top_prob"] == pytest.approx(grid.max())

    def test_top_score_is_tuple_of_ints(self, fitted_model):
        grid, _, _ = fitted_model.scoreline_grid("Chelsea", "Fulham")
        s = grid_summary(grid)
        h, a = s["top_score"]
        assert isinstance(h, int) and isinstance(a, int)


# ---------------------------------------------------------------------------
# DixonColesRatings — ratings_frame
# ---------------------------------------------------------------------------

class TestRatingsFrame:
    def test_all_teams_present(self, fitted_model):
        rf = fitted_model.ratings_frame()
        teams_in_frame = set(rf["team"].tolist())
        for t in ["Arsenal", "Brentford", "Chelsea", "Fulham"]:
            assert t in teams_in_frame

    def test_arsenal_higher_att_xg_than_brentford(self, fitted_model):
        """The dominant team (Arsenal) should have higher att_xg than the weak one."""
        rf = fitted_model.ratings_frame().set_index("team")
        assert rf.loc["Arsenal", "att_xg"] > rf.loc["Brentford", "att_xg"]

    def test_att_xg_positive(self, fitted_model):
        """att_xg represents expected goals and must be positive."""
        rf = fitted_model.ratings_frame()
        assert (rf["att_xg"] > 0).all()

    def test_def_xg_allowed_positive(self, fitted_model):
        rf = fitted_model.ratings_frame()
        assert (rf["def_xg_allowed"] > 0).all()

    def test_net_equals_att_minus_def(self, fitted_model):
        """net should equal att_xg minus def_xg_allowed for every row."""
        rf = fitted_model.ratings_frame()
        diff = (rf["att_xg"] - rf["def_xg_allowed"] - rf["net"]).abs()
        assert (diff < 1e-9).all()


# ---------------------------------------------------------------------------
# DixonColesRatings — expected_goals
# ---------------------------------------------------------------------------

class TestExpectedGoals:
    def test_returns_two_positive_floats(self, fitted_model):
        lam, mu = fitted_model.expected_goals("Arsenal", "Brentford")
        assert isinstance(lam, float) and lam > 0
        assert isinstance(mu, float) and mu > 0

    def test_home_advantage_pushes_lam_above_mu_for_equal_teams(self, fitted_model):
        """Even for similar teams, home team should get higher expected goals."""
        lam, mu = fitted_model.expected_goals("Chelsea", "Fulham")
        # Home advantage is baked in; lam should exceed mu
        assert lam > mu


# ---------------------------------------------------------------------------
# DixonColesRatings — bivariate mode
# ---------------------------------------------------------------------------

class TestBivariateMode:
    def test_bivariate_sets_lambda3(self, synthetic_matches):
        dc = DixonColesRatings(bivariate=True).fit(synthetic_matches)
        assert dc.lambda3_ >= 0.0
        assert dc.rho_ == 0.0  # rho not fitted when bivariate

    def test_bivariate_false_leaves_rho(self, synthetic_matches):
        dc = DixonColesRatings(bivariate=False).fit(synthetic_matches)
        assert dc.rho_ != 0.0  # rho is fitted
        assert dc.lambda3_ == 0.0
