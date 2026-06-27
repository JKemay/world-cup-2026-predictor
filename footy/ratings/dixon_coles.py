"""Dixon-Coles style team ratings + scoreline grid.

Attack and defense strengths are fit by **regularized Poisson regression** on
expected goals (xG) — a more stable signal than raw goals on a small sample.
A home-advantage term is included. The **Dixon-Coles tau** correction adjusts the
low-score cells (0-0, 1-0, 0-1, 1-1) where independent Poisson misprices draws;
its rho parameter is fit by maximum likelihood on the actual scorelines.

For matchup (H home, A away):
    log lambda_home = intercept + home_adv + attack[H] + defense[A]
    log lambda_away = intercept            + attack[A] + defense[H]
where defense[t] is the coefficient for *conceding* to team t (negative = strong
defense, because facing a good defense lowers the opponent's expected goals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from footy.ratings.bivariate_poisson import bivpois_grid, bivpois_pmf

# ---------------------------------------------------------------------------
# Strength-of-schedule (SoS) weighting for goals-fallback rows
# ---------------------------------------------------------------------------
# These are fixed a priori — do NOT tune on the eval set.
# A fallback row (goals used instead of xG) is down-weighted when the
# opponent (defender) had low FIFA strength, so minnow goals vs weak teams
# count less in the Poisson regression.
#
#   w = clip(SOS_W0 + SOS_K * fifa_defense_z, SOS_WLO, SOS_WHI)
#
# where fifa_defense_z is the z-scored FIFA rating of the *opponent*.
SOS_W0 = 0.5        # base weight at average opponent strength (z=0)
SOS_K = 0.30        # shift per unit of opponent FIFA z-score
SOS_WLO, SOS_WHI = 0.1, 1.0   # clip range


def tau(h: int, a: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependency correction."""
    if h == 0 and a == 0:
        return 1.0 - lam * mu * rho
    if h == 0 and a == 1:
        return 1.0 + lam * rho
    if h == 1 and a == 0:
        return 1.0 + mu * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0


def grid_summary(grid: np.ndarray) -> dict:
    h_idx, a_idx = np.indices(grid.shape)
    top_h, top_a = np.unravel_index(grid.argmax(), grid.shape)
    return {
        "home_win": float(grid[h_idx > a_idx].sum()),
        "draw": float(grid[h_idx == a_idx].sum()),
        "away_win": float(grid[h_idx < a_idx].sum()),
        "top_score": (int(top_h), int(top_a)),
        "top_prob": float(grid.max()),
    }


class DixonColesRatings:
    def __init__(self, alpha: float = 0.5, response: str = "xg",
                 fifa: dict[str, float] | None = None, fifa_scale: float = 1.0,
                 team_effects: bool = True, goals_fallback: bool = False,
                 sos_weighting: bool = False, bivariate: bool = False):
        self.alpha = alpha          # L2 strength on per-team adjustments
        self.response = response    # 'xg' (default) or 'goals'
        self.fifa = fifa            # {team: standardized strength} prior, or None
        self.fifa_scale = fifa_scale
        self.team_effects = team_effects  # False = FIFA-only baseline (no per-team data)
        # if True, use actual goals for matches with no shot data instead of skipping them.
        # Default off: it improves aggregate RPS only within noise, lowers top-1 accuracy, and
        # overrates minnows that ran up goals vs weak opposition (no strength-of-schedule weighting).
        self.goals_fallback = goals_fallback
        # if True (and goals_fallback=True and fifa is not None), down-weight goals-fallback rows
        # by the FIFA strength of the opponent (defender): goals scored against weak teams count less,
        # reducing strength-of-schedule bias for thin CAF/Curaçao-style teams.
        self.sos_weighting = sos_weighting
        # if True, use bivariate Poisson (KN2003) instead of DC tau correction.
        self.bivariate = bivariate
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_ = 0.0
        self.intercept_ = 0.0
        self.rho_ = 0.0
        self.lambda3_: float = 0.0
        self.fifa_attack_coef_ = 0.0
        self.fifa_defense_coef_ = 0.0

    def _long_format(self, matches: pd.DataFrame) -> pd.DataFrame:
        col_h, col_a = ("home_xg", "away_xg") if self.response == "xg" else ("home_goals", "away_goals")
        rows = []
        for m in matches.itertuples(index=False):
            yh, ya = getattr(m, col_h), getattr(m, col_a)
            # skip when both xG columns are exactly 0 — provider returned no shot data,
            # not a genuine 0-xG game (real matches always produce some xG from corners/set-pieces)
            is_fallback = 0
            if self.response == "xg" and yh == 0.0 and ya == 0.0:
                if not self.goals_fallback:
                    continue
                # Fall back to actual goals for matches where the provider returned no shot data.
                # Scale is consistent: total xG ≈ total goals by calibration, so goals are an
                # unbiased (if noisier) stand-in when xG is unavailable.
                yh, ya = float(m.home_goals), float(m.away_goals)
                is_fallback = 1
            rows.append((m.home, m.away, 1, yh, is_fallback))
            rows.append((m.away, m.home, 0, ya, is_fallback))
        return pd.DataFrame(rows, columns=["attack", "defense", "is_home", "y", "is_fallback"])

    def fit(self, matches: pd.DataFrame) -> "DixonColesRatings":
        long = self._long_format(matches)
        teams = sorted(set(long["attack"]) | set(long["defense"]))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        use_fifa = self.fifa is not None
        use_dummies = self.team_effects
        off_home = (2 * n) if use_dummies else 0
        off_fifa = off_home + 1
        ncol = off_fifa + (2 if use_fifa else 0)

        X = np.zeros((len(long), ncol))
        for r, row in enumerate(long.itertuples(index=False)):
            if use_dummies:
                X[r, idx[row.attack]] = 1.0
                X[r, n + idx[row.defense]] = 1.0
            X[r, off_home] = row.is_home
            if use_fifa:
                X[r, off_fifa] = self.fifa.get(row.attack, 0.0) * self.fifa_scale
                X[r, off_fifa + 1] = self.fifa.get(row.defense, 0.0) * self.fifa_scale
        y = long["y"].to_numpy(dtype=float)

        # Build sample weights: 1.0 for xG rows; optionally down-weight fallback rows.
        w = np.ones(len(long), dtype=float)
        if self.sos_weighting and self.fifa is not None:
            for r, row in enumerate(long.itertuples(index=False)):
                if row.is_fallback:
                    # row.defense is the opponent team being scored against;
                    # weak opponent (negative z) → lower weight so their inflated
                    # goals-against count less in the regression.
                    z = self.fifa.get(row.defense, 0.0)
                    w[r] = float(np.clip(SOS_W0 + SOS_K * z, SOS_WLO, SOS_WHI))

        model = PoissonRegressor(alpha=self.alpha, max_iter=10000, fit_intercept=True)
        model.fit(X, y, sample_weight=w)
        coef = model.coef_
        self.intercept_ = float(model.intercept_)
        self.home_adv_ = float(coef[off_home])
        fa = float(coef[off_fifa]) if use_fifa else 0.0
        fd = float(coef[off_fifa + 1]) if use_fifa else 0.0
        self.fifa_attack_coef_, self.fifa_defense_coef_ = fa, fd

        # fold FIFA prior + (optional) per-team adjustment into each team's rating;
        # cover every FIFA team so leave-one-out still rates teams absent from train
        rating_teams = sorted(set(teams) | (set(self.fifa) if use_fifa else set()))
        self.attack_, self.defense_ = {}, {}
        for t in rating_teams:
            di = idx.get(t)
            dum_att = float(coef[di]) if (use_dummies and di is not None) else 0.0
            dum_def = float(coef[n + di]) if (use_dummies and di is not None) else 0.0
            f = (self.fifa.get(t, 0.0) * self.fifa_scale) if use_fifa else 0.0
            self.attack_[t] = dum_att + fa * f
            self.defense_[t] = dum_def + fd * f
        if self.bivariate:
            self._fit_lambda3(matches)
        else:
            self._fit_rho(matches)
        return self

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        lam = np.exp(self.intercept_ + self.home_adv_
                     + self.attack_.get(home, 0.0) + self.defense_.get(away, 0.0))
        mu = np.exp(self.intercept_ + self.attack_.get(away, 0.0) + self.defense_.get(home, 0.0))
        return float(lam), float(mu)

    def _fit_rho(self, matches: pd.DataFrame) -> None:
        cases = [
            (self.expected_goals(m.home, m.away), int(m.home_goals), int(m.away_goals))
            for m in matches.itertuples(index=False)
        ]

        def nll(rho: float) -> float:
            total = 0.0
            for (lam, mu), hg, ag in cases:
                p = poisson.pmf(hg, lam) * poisson.pmf(ag, mu) * tau(hg, ag, lam, mu, rho)
                total -= np.log(max(p, 1e-12))
            return total

        self.rho_ = float(minimize_scalar(nll, bounds=(-0.2, 0.2), method="bounded").x)

    def _fit_lambda3(self, matches: pd.DataFrame) -> None:
        cases = [
            (self.expected_goals(m.home, m.away), int(m.home_goals), int(m.away_goals))
            for m in matches.itertuples(index=False)
        ]

        def nll(l3: float) -> float:
            total = 0.0
            for (lam, mu), hg, ag in cases:
                p = bivpois_pmf(hg, ag, lam, mu, l3)
                total -= np.log(max(p, 1e-12))
            return total

        self.lambda3_ = float(minimize_scalar(nll, bounds=(0.0, 1.0), method="bounded").x)

    def scoreline_grid(self, home: str, away: str, max_goals: int = 6):
        lam, mu = self.expected_goals(home, away)
        if self.bivariate:
            grid = bivpois_grid(lam, mu, self.lambda3_, max_goals)
        else:
            k = np.arange(max_goals + 1)
            grid = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
            for h in (0, 1):
                for a in (0, 1):
                    grid[h, a] *= tau(h, a, lam, mu, self.rho_)
            grid /= grid.sum()
        return grid, lam, mu

    def ratings_frame(self) -> pd.DataFrame:
        """Interpretable per-team ratings: expected xG vs an average opponent."""
        mean_att = float(np.mean(list(self.attack_.values())))
        mean_def = float(np.mean(list(self.defense_.values())))
        rows = []
        for t in self.teams_:
            att_xg = np.exp(self.intercept_ + self.attack_[t] + mean_def)
            def_xg = np.exp(self.intercept_ + mean_att + self.defense_[t])
            rows.append({"team": t, "att_xg": att_xg, "def_xg_allowed": def_xg,
                         "net": att_xg - def_xg})
        return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)
