"""Tests for footy/features/xg.py — geometry helpers and model training."""

import os

import numpy as np
import pandas as pd
import pytest

from footy.features.xg import (
    add_geometry,
    predict_xg,
    train_xg,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHOTS_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "shots_xg.csv",
)


def _make_shot(x_norm, y_norm, is_goal=0):
    return pd.DataFrame({"x_norm": [x_norm], "y_norm": [y_norm], "is_goal": [is_goal]})


# ---------------------------------------------------------------------------
# add_geometry — distance
# ---------------------------------------------------------------------------

class TestAddGeometryDistance:
    def test_centre_goal_line_distance_is_near_zero(self):
        """A shot at x_norm=100, y_norm=50 is on the goal line at centre: distance ~ 0."""
        df = add_geometry(_make_shot(100, 50))
        assert df["distance"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_midfield_shot_farther_than_penalty_shot(self):
        """A shot from x_norm=50 (midfield) is further away than from x_norm=85."""
        far = add_geometry(_make_shot(50, 50))["distance"].iloc[0]
        near = add_geometry(_make_shot(85, 50))["distance"].iloc[0]
        assert far > near

    def test_distance_positive_for_all_shots(self):
        """Distance is always non-negative."""
        df = pd.DataFrame({
            "x_norm": [10, 50, 80, 99, 100],
            "y_norm": [10, 50, 50, 50,  50],
            "is_goal": [0, 0, 0, 0, 0],
        })
        result = add_geometry(df)
        assert (result["distance"] >= 0).all()

    def test_distance_units_plausible(self):
        """A central shot at x_norm=80 should be roughly 25-30 m from goal."""
        df = add_geometry(_make_shot(80, 50))
        d = df["distance"].iloc[0]
        # x_m = 80/100*105 = 84 m, so distance from goal (105 m line) = 21 m
        assert 15.0 < d < 30.0


# ---------------------------------------------------------------------------
# add_geometry — angle
# ---------------------------------------------------------------------------

class TestAddGeometryAngle:
    def test_central_close_angle_larger_than_tight_corner(self):
        """A shot close to goal at centre has a wider angle than from a tight corner."""
        central = add_geometry(_make_shot(95, 50))["angle"].iloc[0]
        tight_corner = add_geometry(_make_shot(99, 5))["angle"].iloc[0]
        assert central > tight_corner

    def test_angle_positive(self):
        """Angle (in radians) should be > 0 for any shot not at infinity."""
        df = pd.DataFrame({
            "x_norm": [50, 80, 90, 95],
            "y_norm": [50, 50, 20, 80],
            "is_goal": [0, 0, 0, 0],
        })
        result = add_geometry(df)
        assert (result["angle"] > 0).all()

    def test_angle_in_radians_range(self):
        """Angle should be in [0, pi]."""
        df = pd.DataFrame({
            "x_norm": [10, 50, 80, 90, 100],
            "y_norm": [10, 50, 50, 50,  50],
            "is_goal": [0, 0, 0, 0, 0],
        })
        result = add_geometry(df)
        assert (result["angle"] >= 0).all()
        assert (result["angle"] <= np.pi).all()


# ---------------------------------------------------------------------------
# predict_xg — monotonicity
# ---------------------------------------------------------------------------

class TestPredictXgMonotonicity:
    def _train_simple_model(self):
        """Train on tiny labelled data enough for logistic regression to converge."""
        rng = np.random.default_rng(42)
        n = 200
        x_norm = rng.uniform(50, 100, n)
        y_norm = rng.uniform(20, 80, n)
        df = pd.DataFrame({"x_norm": x_norm, "y_norm": y_norm, "is_goal": 0})
        df = add_geometry(df)
        # assign goals biased toward close central shots so the model is learnable
        prob = 0.05 + 0.35 * np.exp(-df["distance"] / 15) * np.exp(-df["angle"].abs() / 0.5)
        df["is_goal"] = (rng.random(n) < prob).astype(int)
        # ensure at least one goal for each fold
        df.loc[df.nsmallest(10, "distance").index, "is_goal"] = 1
        model, _, _ = train_xg(df)
        return model

    def test_close_central_higher_xg_than_distant(self):
        """A shot at x_norm=95,y_norm=50 should have higher xG than x_norm=55,y_norm=50."""
        model = self._train_simple_model()
        close = add_geometry(_make_shot(95, 50))
        far = add_geometry(_make_shot(55, 50))
        xg_close = predict_xg(model, close)[0]
        xg_far = predict_xg(model, far)[0]
        assert xg_close > xg_far

    def test_xg_probabilities_in_unit_interval(self):
        """All xG values must be in [0, 1]."""
        model = self._train_simple_model()
        df = pd.DataFrame({
            "x_norm": [10, 50, 80, 90, 100],
            "y_norm": [10, 50, 50, 50,  50],
            "is_goal": [0, 0, 0, 0, 0],
        })
        df = add_geometry(df)
        xgs = predict_xg(model, df)
        assert (xgs >= 0).all() and (xgs <= 1).all()


# ---------------------------------------------------------------------------
# train_xg — calibration on real data (skip if CSV absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(SHOTS_CSV), reason="shots_xg.csv not found")
class TestTrainXgRealData:
    @pytest.fixture(scope="class")
    def trained(self):
        df = pd.read_csv(SHOTS_CSV)
        df = add_geometry(df)
        model, cv_prob, metrics = train_xg(df)
        return model, cv_prob, metrics, df

    def test_metrics_keys_present(self, trained):
        _, _, metrics, _ = trained
        for key in ("n_shots", "n_goals", "conversion", "cv_logloss",
                    "baseline_logloss", "cv_auc", "cv_brier"):
            assert key in metrics

    def test_cv_auc_above_chance(self, trained):
        """Cross-validated AUC should be meaningfully above 0.6 on real shot data."""
        _, _, metrics, _ = trained
        assert metrics["cv_auc"] > 0.6, f"cv_auc={metrics['cv_auc']:.4f} is too low"

    def test_calibration_total_xg_within_1pct_of_goals(self, trained):
        """Sum of predicted xG should be within 1% of actual goal count (calibration)."""
        model, _, _, df = trained
        total_xg = predict_xg(model, df).sum()
        total_goals = float(df["is_goal"].sum())
        rel_error = abs(total_xg - total_goals) / total_goals
        assert rel_error < 0.01, (
            f"Calibration error too large: predicted={total_xg:.1f}, "
            f"actual={total_goals:.1f}, rel_error={rel_error:.4f}"
        )

    def test_n_shots_matches_dataframe(self, trained):
        """Metrics n_shots should match the DataFrame length."""
        _, _, metrics, df = trained
        assert metrics["n_shots"] == len(df)
