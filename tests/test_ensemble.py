"""Tests for footy/ratings/ensemble.py.

All tests are deterministic and require no network or data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footy.ratings.dixon_coles import grid_summary
from footy.ratings.ensemble import EnsemblePredictor

# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_matches() -> pd.DataFrame:
    """Small deterministic match table: 12 matches across 4 teams.

    Includes the columns required by both DixonColesRatings and fit_elo.
    xG values are non-zero so Dixon-Coles has real signal to fit on.
    """
    rng = np.random.default_rng(0)
    pairs = [
        ("Alpha", "Beta"), ("Beta", "Gamma"), ("Gamma", "Delta"),
        ("Alpha", "Gamma"), ("Beta", "Delta"), ("Alpha", "Delta"),
        ("Beta", "Alpha"), ("Gamma", "Beta"), ("Delta", "Alpha"),
        ("Alpha", "Beta"), ("Gamma", "Alpha"), ("Delta", "Beta"),
    ]
    hg = rng.integers(0, 4, size=len(pairs)).tolist()
    ag = rng.integers(0, 4, size=len(pairs)).tolist()
    hx = (rng.uniform(0.3, 2.5, size=len(pairs))).tolist()
    ax = (rng.uniform(0.3, 2.5, size=len(pairs))).tolist()
    dates = [f"2024-01-{i+1:02d}T12:00:00+00:00" for i in range(len(pairs))]
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(len(pairs))],
        "date": dates,
        "season_id": ["s1"] * len(pairs),
        "home": [p[0] for p in pairs],
        "away": [p[1] for p in pairs],
        "home_goals": hg,
        "away_goals": ag,
        "home_xg": hx,
        "away_xg": ax,
    })


@pytest.fixture(scope="module")
def fitted_ensemble(synthetic_matches) -> EnsemblePredictor:
    """Default (50/50) ensemble fitted once for all tests in this module."""
    return EnsemblePredictor().fit(synthetic_matches)


# ---------------------------------------------------------------------------
# Basic fit / wdl sanity
# ---------------------------------------------------------------------------

class TestEnsembleFitAndWdl:
    def test_fit_returns_self(self, synthetic_matches):
        model = EnsemblePredictor()
        result = model.fit(synthetic_matches)
        assert result is model

    def test_wdl_sums_to_one(self, fitted_ensemble):
        wdl = fitted_ensemble.wdl("Alpha", "Beta")
        assert wdl.sum() == pytest.approx(1.0, abs=1e-6)

    def test_wdl_all_entries_in_unit_interval(self, fitted_ensemble):
        wdl = fitted_ensemble.wdl("Alpha", "Beta")
        assert wdl.shape == (3,)
        assert (wdl >= 0).all()
        assert (wdl <= 1).all()

    def test_wdl_different_matchups_differ(self, fitted_ensemble):
        w1 = fitted_ensemble.wdl("Alpha", "Beta")
        w2 = fitted_ensemble.wdl("Beta", "Alpha")
        # home and away flipped — at minimum p_home should swap vs p_away
        assert not np.allclose(w1, w2)


# ---------------------------------------------------------------------------
# Scoreline grid delegation
# ---------------------------------------------------------------------------

class TestScorelineGrid:
    def test_grid_sums_to_one(self, fitted_ensemble):
        grid, _, _ = fitted_ensemble.scoreline_grid("Alpha", "Gamma")
        assert grid.sum() == pytest.approx(1.0, abs=1e-6)

    def test_grid_all_non_negative(self, fitted_ensemble):
        grid, _, _ = fitted_ensemble.scoreline_grid("Alpha", "Gamma")
        assert (grid >= 0).all()

    def test_grid_shape(self, fitted_ensemble):
        grid, _, _ = fitted_ensemble.scoreline_grid("Alpha", "Gamma", max_goals=5)
        assert grid.shape == (6, 6)

    def test_grid_lam_mu_positive(self, fitted_ensemble):
        _, lam, mu = fitted_ensemble.scoreline_grid("Alpha", "Beta")
        assert lam > 0
        assert mu > 0


# ---------------------------------------------------------------------------
# Weight extremes: weight=1.0 → pure DC; weight=0.0 → pure Elo
# ---------------------------------------------------------------------------

class TestWeightExtremes:
    def test_default_weight_is_fifty_fifty(self):
        """Pins the shipped default. Re-tuned via the nested-LOO protocol in
        build_eval.py in 2026-07: the point estimate favored more Elo weight,
        but the gain was not statistically significant (P=0.948, 95% CI
        [-0.0133, +0.0014]) — see docs/METHODOLOGY.md sec.6. Retained 50/50."""
        assert EnsemblePredictor().weight == 0.5

    def test_weight_1_equals_pure_dixon_coles(self, synthetic_matches):
        """weight=1.0: wdl must match the Dixon-Coles grid_summary output."""
        model = EnsemblePredictor(weight=1.0).fit(synthetic_matches)
        wdl = model.wdl("Alpha", "Beta")
        grid, _, _ = model.dc_.scoreline_grid("Alpha", "Beta")
        s = grid_summary(grid)
        dc_raw = np.array([s["home_win"], s["draw"], s["away_win"]])
        dc_wdl = dc_raw / dc_raw.sum()
        np.testing.assert_allclose(wdl, dc_wdl, atol=1e-9)

    def test_weight_0_equals_pure_elo(self, synthetic_matches):
        """weight=0.0: wdl must match the Elo model's final-rating prediction."""
        from footy.ratings.elo import predict_wdl
        model = EnsemblePredictor(weight=0.0).fit(synthetic_matches)
        wdl = model.wdl("Alpha", "Beta")
        r_h = model.elo_.ratings.get("Alpha", 1500.0)
        r_a = model.elo_.ratings.get("Beta", 1500.0)
        elo_wdl = predict_wdl(model.elo_, r_h, r_a)
        np.testing.assert_allclose(wdl, elo_wdl, atol=1e-9)


# ---------------------------------------------------------------------------
# Delegating properties / methods
# ---------------------------------------------------------------------------

class TestDelegation:
    def test_attack_exposes_dc_teams(self, fitted_ensemble):
        att = fitted_ensemble.attack_
        assert isinstance(att, dict)
        for team in ("Alpha", "Beta", "Gamma", "Delta"):
            assert team in att

    def test_ratings_frame_returns_dataframe(self, fitted_ensemble):
        rf = fitted_ensemble.ratings_frame()
        assert isinstance(rf, pd.DataFrame)
        assert "team" in rf.columns
        assert "att_xg" in rf.columns

    def test_expected_goals_positive(self, fitted_ensemble):
        lam, mu = fitted_ensemble.expected_goals("Alpha", "Gamma")
        assert lam > 0
        assert mu > 0


# ---------------------------------------------------------------------------
# draw_k and bivariate params
# ---------------------------------------------------------------------------

class TestDrawKAndBivariate:
    def test_draw_k_identity(self, synthetic_matches):
        """draw_k=1.0 should not change wdl output."""
        model_default = EnsemblePredictor(draw_k=1.0).fit(synthetic_matches)
        model_no_k = EnsemblePredictor().fit(synthetic_matches)
        np.testing.assert_allclose(model_default.wdl("Alpha", "Beta"), model_no_k.wdl("Alpha", "Beta"))

    def test_draw_k_increases_draw(self, synthetic_matches):
        model = EnsemblePredictor().fit(synthetic_matches)
        model_k = EnsemblePredictor(draw_k=1.5).fit(synthetic_matches)
        assert model_k.wdl("Alpha", "Beta")[1] > model.wdl("Alpha", "Beta")[1]

    def test_draw_k_sums_to_one(self, synthetic_matches):
        model = EnsemblePredictor(draw_k=1.5).fit(synthetic_matches)
        assert abs(model.wdl("Alpha", "Beta").sum() - 1.0) < 1e-8

    def test_bivariate_smoke(self, synthetic_matches):
        model = EnsemblePredictor(bivariate=True).fit(synthetic_matches)
        grid, lam, mu = model.scoreline_grid("Alpha", "Beta")
        assert grid.shape == (7, 7)
        assert abs(grid.sum() - 1.0) < 1e-6


