"""Ensemble predictor: Dixon-Coles (xG) blended with World-Football Elo.

The ensemble averages W/D/L *probabilities* 50/50 between the two models.
The scoreline grid (for display and top-score extraction) is provided
exclusively by Dixon-Coles, which has an explicit Poisson model for scorelines;
the Elo model has no scoreline representation.

Typical usage
-------------
>>> from footy.ratings.ensemble import EnsemblePredictor
>>> model = EnsemblePredictor(alpha=0.05, fifa=fifa_strength(teams)).fit(matches)
>>> model.wdl('France', 'Iraq')          # blended W/D/L
>>> model.scoreline_grid('France', 'Iraq')  # Dixon-Coles grid for display
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footy.config import WC_SEASON_ID
from footy.ratings.dixon_coles import DixonColesRatings, grid_summary
from footy.ratings.elo import EloRatings, fit_elo, predict_wdl


class EnsemblePredictor:
    """Blend Dixon-Coles xG ratings with World-Football Elo ratings.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength passed to :class:`DixonColesRatings`.
    fifa : dict[str, float] or None
        Standardised FIFA-rank prior for Dixon-Coles (and Elo initialisation).
    fifa_scale : float
        Scaling factor for the FIFA prior in Dixon-Coles.
    team_effects : bool
        Whether to include per-team parameters in Dixon-Coles (True = full model).
    weight : float
        Blend weight *on the Dixon-Coles W/D/L*. ``weight=0.5`` (default) gives
        equal contribution from both models. ``weight=1.0`` is pure Dixon-Coles;
        ``weight=0.0`` is pure Elo.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        fifa: dict[str, float] | None = None,
        fifa_scale: float = 1.0,
        team_effects: bool = True,
        weight: float = 0.5,
        draw_k: float = 1.0,
        bivariate: bool = False,
    ) -> None:
        self.alpha = alpha
        self.fifa = fifa
        self.fifa_scale = fifa_scale
        self.team_effects = team_effects
        self.weight = weight
        self.draw_k = draw_k
        self.bivariate = bivariate
        # Fitted models — populated by fit()
        self.dc_: DixonColesRatings | None = None
        self.elo_: EloRatings | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, matches: pd.DataFrame) -> "EnsemblePredictor":
        """Fit both constituent models on *matches*.

        Parameters
        ----------
        matches : pd.DataFrame
            Match table as returned by :func:`footy.features.matches.build_match_table`.
            Must contain: ``match_id``, ``date``, ``season_id``, ``home``, ``away``,
            ``home_goals``, ``away_goals``, ``home_xg``, ``away_xg``.

        Returns
        -------
        self
        """
        self.dc_ = DixonColesRatings(
            alpha=self.alpha,
            fifa=self.fifa,
            fifa_scale=self.fifa_scale,
            team_effects=self.team_effects,
            bivariate=self.bivariate,
        ).fit(matches)

        # Fit Elo with draw-params calibrated on non-WC data to avoid leakage
        wc_season_id = WC_SEASON_ID if "season_id" in matches.columns else None
        self.elo_ = fit_elo(matches, wc_season_id=wc_season_id)
        return self

    # ------------------------------------------------------------------
    # Delegation to Dixon-Coles (scoreline, ratings, expected goals)
    # ------------------------------------------------------------------

    def scoreline_grid(self, home: str, away: str, max_goals: int = 6):
        """Return the Dixon-Coles scoreline grid (grid, lam, mu).

        The grid is from the xG model only; Elo has no scoreline representation.
        """
        assert self.dc_ is not None, "Call fit() first."
        return self.dc_.scoreline_grid(home, away, max_goals=max_goals)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Return expected goals (lam, mu) from the Dixon-Coles component."""
        assert self.dc_ is not None, "Call fit() first."
        return self.dc_.expected_goals(home, away)

    def ratings_frame(self) -> pd.DataFrame:
        """Per-team attack/defense ratings from the Dixon-Coles component."""
        assert self.dc_ is not None, "Call fit() first."
        return self.dc_.ratings_frame()

    @property
    def attack_(self) -> dict[str, float]:
        """Attack ratings dict (Dixon-Coles), used to populate the team dropdown."""
        assert self.dc_ is not None, "Call fit() first."
        return self.dc_.attack_

    # ------------------------------------------------------------------
    # Ensemble W/D/L
    # ------------------------------------------------------------------

    def wdl(self, home: str, away: str) -> np.ndarray:
        """Blended W/D/L outcome probabilities.

        Combines Dixon-Coles (from the scoreline grid) and Elo (from pre-match
        ratings) using ``self.weight`` as the DC blend fraction.

        Parameters
        ----------
        home, away : str
            Team names as they appear in the match table.

        Returns
        -------
        np.ndarray of shape (3,)
            [p_home_win, p_draw, p_away_win], normalised to sum to 1.
        """
        assert self.dc_ is not None and self.elo_ is not None, "Call fit() first."

        # Dixon-Coles W/D/L from the scoreline grid
        grid, _, _ = self.dc_.scoreline_grid(home, away)
        s = grid_summary(grid)
        dc_raw = np.array([s["home_win"], s["draw"], s["away_win"]], dtype=float)
        dc_wdl = dc_raw / dc_raw.sum()

        # Elo W/D/L from final ratings (for live prediction; LOO uses pre-match)
        r_h = self.elo_.ratings.get(home, 1500.0)
        r_a = self.elo_.ratings.get(away, 1500.0)
        elo_wdl = predict_wdl(self.elo_, r_h, r_a)

        ens = self.weight * dc_wdl + (1.0 - self.weight) * elo_wdl
        ens = ens / ens.sum()
        if self.draw_k != 1.0:
            from footy.evaluate.backtest import apply_draw_scalar
            ens = apply_draw_scalar(ens, self.draw_k)
        return ens
