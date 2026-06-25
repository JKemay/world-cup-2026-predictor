"""World-Football-style Elo rating system for international football.

This module implements the Elo rating method used by the World Football Elo
Ratings (eloratings.net), adapted for use as an external benchmark alongside
the xG/Dixon-Coles model.

Key design choices:
- Ratings are initialised from FIFA rank via ``fifa_strength()`` so we do not
  start every team at equal strength on a short sample of 376 matches.
- Matches are processed in strict chronological order (sorted by ``date``).
- The win-probability model uses the standard Elo expectancy plus a draw
  component whose parameters (D0, DW) are fit by maximum likelihood on all
  non-WC training matches (season_id != WC_SEASON_ID).  This avoids leaking
  WC fixture-level labels into the draw model.
- Pre-match ratings are stored before each update so that WC match predictions
  are truly leakage-free.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from footy.ratings.fifa import fifa_strength

# ---------------------------------------------------------------------------
# Constants (World Football Elo standard values)
# ---------------------------------------------------------------------------

HOME_ADV: float = 65.0   # rating-point home advantage (WFE default)
K_FACTOR: float = 40.0   # base K-factor (WFE value for World Cup matches)
ELO_BASE: float = 1500.0 # default rating for teams with no FIFA rank
FIFA_SCALE: float = 150.0 # scale for z-scored FIFA strength → Elo points


# ---------------------------------------------------------------------------
# Core Elo maths
# ---------------------------------------------------------------------------

def _goal_multiplier(goal_diff: int) -> float:
    """World Football Elo goal-difference multiplier G.

    G = 1           if |gd| <= 1
    G = 1.5         if |gd| == 2
    G = (11+|gd|)/8 if |gd| >= 3
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11 + gd) / 8.0


def expected_score(r_home: float, r_away: float, home_adv: float = HOME_ADV) -> float:
    """Expected score (win probability component) for the home team.

    We = 1 / (1 + 10^(-(dr)/400))
    where dr = R_home - R_away + HOME_ADV.

    Returns a value in (0, 1) that satisfies:
        We = P_home + 0.5 * P_draw
    """
    dr = r_home - r_away + home_adv
    return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))


def update_ratings(
    r_home: float,
    r_away: float,
    home_goals: int,
    away_goals: int,
    k: float = K_FACTOR,
    home_adv: float = HOME_ADV,
) -> tuple[float, float]:
    """Apply one Elo update and return new (r_home, r_away).

    The update is zero-sum: rating gained by home = rating lost by away.

    Parameters
    ----------
    r_home, r_away : float
        Pre-match Elo ratings.
    home_goals, away_goals : int
        Final score.
    k : float
        Base K-factor.
    home_adv : float
        Home-advantage rating bonus.

    Returns
    -------
    (new_r_home, new_r_away)
    """
    we = expected_score(r_home, r_away, home_adv)
    gd = home_goals - away_goals
    g = _goal_multiplier(gd)
    if gd > 0:
        s = 1.0   # home win
    elif gd == 0:
        s = 0.5   # draw
    else:
        s = 0.0   # home loss
    delta = k * g * (s - we)
    return r_home + delta, r_away - delta


# ---------------------------------------------------------------------------
# Draw-probability model
# ---------------------------------------------------------------------------

class DrawParams(NamedTuple):
    """Parameters of the draw-probability model.

    P_draw = D0 * exp(-( dr / DW )**2 )

    where dr = R_home - R_away + HOME_ADV.
    """
    D0: float  # peak draw probability (when teams are evenly matched)
    DW: float  # width parameter (larger DW → draws extend further from parity)


def _draw_nll(params: tuple[float, float], drs: np.ndarray, outcomes: np.ndarray) -> float:
    """Negative log-likelihood of the draw model on a set of matches.

    Parameters
    ----------
    params : (D0, DW)
    drs : array of pre-match rating differentials (dr = R_h - R_a + HOME_ADV)
    outcomes : int array, 0=home win, 1=draw, 2=away win
    """
    d0, dw = params
    if d0 <= 0 or d0 >= 1 or dw <= 1:
        return 1e12
    p_draw = d0 * np.exp(-((drs / dw) ** 2))
    we = 1.0 / (1.0 + 10.0 ** (-drs / 400.0))
    p_home = np.clip(we - 0.5 * p_draw, 1e-9, 1.0)
    p_away = np.clip(1.0 - p_home - p_draw, 1e-9, 1.0)
    p_draw = np.clip(p_draw, 1e-9, 1.0)
    # Renormalise
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    log_liks = np.where(
        outcomes == 0, np.log(p_home),
        np.where(outcomes == 1, np.log(p_draw), np.log(p_away)),
    )
    return -float(log_liks.sum())


def fit_draw_params(drs: np.ndarray, outcomes: np.ndarray) -> DrawParams:
    """Fit D0 and DW by maximum likelihood.

    Parameters
    ----------
    drs : 1-D array of pre-match Elo differentials (home-adjusted).
    outcomes : 1-D int array, 0=home win, 1=draw, 2=away win.

    Returns
    -------
    DrawParams
    """
    result = minimize(
        _draw_nll,
        x0=[0.30, 200.0],
        args=(drs, outcomes),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000},
    )
    d0, dw = result.x
    # Clamp to sensible range
    d0 = float(np.clip(d0, 0.05, 0.60))
    dw = float(np.clip(abs(dw), 50.0, 1000.0))
    return DrawParams(D0=d0, DW=dw)


def elo_match_probs(
    r_home: float,
    r_away: float,
    draw_params: DrawParams,
    home_adv: float = HOME_ADV,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert a pre-match rating pair to win/draw/loss probabilities.

    Returns
    -------
    np.ndarray of shape (3,): [p_home, p_draw, p_away], summing to 1.
    """
    dr = r_home - r_away + home_adv
    we = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    p_draw = draw_params.D0 * np.exp(-((dr / draw_params.DW) ** 2))
    p_home = we - 0.5 * p_draw
    p_away = 1.0 - p_home - p_draw
    probs = np.clip(np.array([p_home, p_draw, p_away]), eps, 1.0)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# EloRatings — main class
# ---------------------------------------------------------------------------

@dataclass
class EloRatings:
    """Fitted Elo ratings with full history for leakage-free prediction.

    Attributes
    ----------
    ratings : dict[str, float]
        Final Elo rating for each team.
    pre_match_ratings : dict[str, tuple[float, float]]
        Mapping of match_id → (r_home, r_away) *before* that match was played.
    draw_params : DrawParams
        Fitted draw-probability parameters.
    """
    ratings: dict[str, float] = field(default_factory=dict)
    pre_match_ratings: dict[str, tuple[float, float]] = field(default_factory=dict)
    draw_params: DrawParams = field(default_factory=lambda: DrawParams(D0=0.28, DW=200.0))


def _init_ratings(teams: list[str]) -> dict[str, float]:
    """Initialise Elo ratings from FIFA rank.

    Computes the z-scored ``fifa_strength`` for the supplied teams and maps it
    to Elo points: R0 = ELO_BASE + FIFA_SCALE * z_strength.
    Teams absent from FIFA_RANK start at ELO_BASE (z-score treated as 0).
    If the team list contains no teams in FIFA_RANK (so std of log-rank is 0),
    all teams are initialised to ELO_BASE.
    """
    fs = fifa_strength(teams)
    # fifa_strength returns NaN when std==0 (all teams use the fallback rank).
    # Guard: if any value is NaN, fall back to ELO_BASE for all.
    if any(math.isnan(v) for v in fs.values()):
        return {t: ELO_BASE for t in teams}
    return {t: ELO_BASE + FIFA_SCALE * fs.get(t, 0.0) for t in teams}


def fit_elo(
    matches: pd.DataFrame,
    wc_season_id: str | None = None,
    k: float = K_FACTOR,
    home_adv: float = HOME_ADV,
) -> EloRatings:
    """Fit Elo ratings on ``matches`` in chronological order.

    Parameters
    ----------
    matches : DataFrame
        Must contain: match_id, date, season_id (optional), home, away,
        home_goals, away_goals.  ``date`` is ISO-8601 string; sorting is
        lexicographic (works for the +00:00 suffix timestamps in this dataset).
    wc_season_id : str or None
        If provided, draw parameters are fit on non-WC matches only (avoids
        leaking WC labels into the draw model).  If None, all matches are used.
    k : float
        Base K-factor.
    home_adv : float
        Home-advantage rating bonus.

    Returns
    -------
    EloRatings
        Populated with final ratings, per-match pre-match ratings, and fitted
        draw parameters.
    """
    # Sort chronologically
    df = matches.sort_values("date").reset_index(drop=True)

    all_teams = sorted(set(df["home"]).union(df["away"]))
    ratings = _init_ratings(all_teams)

    pre_match: dict[str, tuple[float, float]] = {}

    for _, row in df.iterrows():
        mid = row["match_id"]
        home, away = row["home"], row["away"]
        r_h = ratings.get(home, ELO_BASE)
        r_a = ratings.get(away, ELO_BASE)
        pre_match[mid] = (r_h, r_a)

        new_r_h, new_r_a = update_ratings(
            r_h, r_a, int(row["home_goals"]), int(row["away_goals"]),
            k=k, home_adv=home_adv,
        )
        ratings[home] = new_r_h
        ratings[away] = new_r_a

    # ------------------------------------------------------------------
    # Fit draw parameters
    # ------------------------------------------------------------------
    # Use non-WC matches for fitting to avoid leakage (if wc_season_id given)
    if wc_season_id is not None and "season_id" in df.columns:
        train_df = df[df["season_id"] != wc_season_id]
    else:
        train_df = df

    if len(train_df) < 10:
        warnings.warn("Too few training matches for draw-param fitting; using defaults.", stacklevel=2)
        draw_params = DrawParams(D0=0.28, DW=200.0)
    else:
        drs = np.array([
            pre_match[row["match_id"]][0] - pre_match[row["match_id"]][1] + home_adv
            for _, row in train_df.iterrows()
            if row["match_id"] in pre_match
        ])
        from footy.evaluate.backtest import actual_outcome
        outcomes = np.array([
            actual_outcome(int(row["home_goals"]), int(row["away_goals"]))
            for _, row in train_df.iterrows()
            if row["match_id"] in pre_match
        ])
        draw_params = fit_draw_params(drs, outcomes)

    return EloRatings(
        ratings=ratings,
        pre_match_ratings=pre_match,
        draw_params=draw_params,
    )


def elo_strength(matches: pd.DataFrame) -> dict[str, float]:
    """Fit Elo on *matches* and return each team's final rating, z-scored.

    The z-scoring (subtract mean, divide by std across all teams) makes the
    output drop-in compatible with the ``fifa=`` prior argument of
    ``DixonColesRatings``, which expects standardized strengths on the same
    scale as ``fifa_strength()``.

    Parameters
    ----------
    matches : DataFrame
        Must contain the columns required by :func:`fit_elo` (``match_id``,
        ``date``, ``home``, ``away``, ``home_goals``, ``away_goals``).

    Returns
    -------
    dict[str, float]
        Mapping of team name → z-scored Elo rating (mean ≈ 0, std ≈ 1 across
        the teams present in *matches*).
    """
    elo = fit_elo(matches)
    ratings = elo.ratings
    vals = np.array(list(ratings.values()), dtype=float)
    mean, std = float(vals.mean()), float(vals.std())
    if std == 0.0:
        return {t: 0.0 for t in ratings}
    return {t: (v - mean) / std for t, v in ratings.items()}


def predict_wdl(
    elo: EloRatings,
    r_home: float,
    r_away: float,
    home_adv: float = HOME_ADV,
) -> np.ndarray:
    """Predict W/D/L probabilities from pre-match ratings.

    Parameters
    ----------
    elo : EloRatings
        Fitted model (used for draw_params).
    r_home, r_away : float
        Pre-match Elo ratings.
    home_adv : float
        Home-advantage bonus.

    Returns
    -------
    np.ndarray of shape (3,): [p_home, p_draw, p_away]
    """
    return elo_match_probs(r_home, r_away, elo.draw_params, home_adv=home_adv)
