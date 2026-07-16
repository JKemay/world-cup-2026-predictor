"""Leave-one-out backtest of the rating model with proper scoring rules.

For each match we refit the ratings on the *other* matches and predict the
held-out one, so nothing leaks. Outcomes are scored with multiclass log-loss and
the Ranked Probability Score (RPS) — the standard metric for ordered
home/draw/away football outcomes — plus top-1 accuracy and a reliability curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from footy.ratings.dixon_coles import DixonColesRatings, grid_summary


def actual_outcome(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0  # home
    if home_goals == away_goals:
        return 1  # draw
    return 2      # away


def model_probs(model: DixonColesRatings, home: str, away: str) -> np.ndarray:
    grid, _, _ = model.scoreline_grid(home, away)
    s = grid_summary(grid)
    p = np.array([s["home_win"], s["draw"], s["away_win"]])
    return p / p.sum()


def rps(probs: np.ndarray, outcome: int) -> float:
    """Ranked Probability Score for ordinal home<draw<away (lower is better)."""
    cum_p = np.cumsum(probs)
    cum_o = np.cumsum(np.eye(3)[outcome])
    return float(np.sum((cum_p - cum_o) ** 2) / (len(probs) - 1))


def apply_draw_scalar(probs: np.ndarray, k: float) -> np.ndarray:
    """Multiply P(draw) by k then renormalize. Works on shape (3,) or (N,3)."""
    out = probs.copy().astype(float)
    if out.ndim == 1:
        out[1] *= k
        s = out.sum()
        if s > 0:
            out /= s
    else:
        out[:, 1] *= k
        row_sums = out.sum(axis=1, keepdims=True)
        nonzero = (row_sums > 0).ravel()
        out[nonzero] /= row_sums[nonzero]
    return out


def fit_draw_scalar(preds: np.ndarray, actuals: np.ndarray, bounds: tuple = (0.5, 3.0)) -> float:
    """Minimize mean RPS over k in bounds. Returns fitted float k."""
    def objective(k: float) -> float:
        return score(apply_draw_scalar(preds, k), actuals)["rps"]

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(result.x)


def blend_probs(p_a: np.ndarray, p_b: np.ndarray, w: float) -> np.ndarray:
    """Blend two W/D/L probability arrays: ``w * p_a + (1 - w) * p_b``, renormalized.

    Works on shape ``(3,)`` or ``(N, 3)``.
    """
    out = w * p_a + (1.0 - w) * p_b
    if out.ndim == 1:
        return out / out.sum()
    return out / out.sum(axis=1, keepdims=True)


def fit_blend_weight(p_a: np.ndarray, p_b: np.ndarray, actuals: np.ndarray,
                     grid: np.ndarray | None = None) -> float:
    """Select the blend weight ``w`` (on ``p_a``) minimizing mean RPS over a grid.

    In-sample selection — the same predictions used to pick ``w`` are used to
    score it here, so this is optimistic. Use :func:`nested_blend_predictions`
    for an honest, leakage-free estimate of blend performance.
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)
    best_rps, best_w = np.inf, float(grid[0])
    for w in grid:
        r = score(blend_probs(p_a, p_b, float(w)), actuals)["rps"]
        if r < best_rps:
            best_rps, best_w = r, float(w)
    return best_w


def nested_blend_predictions(p_a: np.ndarray, p_b: np.ndarray, actuals: np.ndarray,
                             grid: np.ndarray | None = None) -> np.ndarray:
    """Honest (nested-LOO) blend predictions: never scores a match with a weight

    that saw that match. For each row ``i``, the blend weight is refit on every
    *other* row, then applied to predict row ``i``. This reports the performance
    of the *procedure* "tune the weight on available data," not the performance
    of a weight that has already seen the answer.
    """
    n = len(actuals)
    preds = np.empty_like(p_a, dtype=float)
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        w_i = fit_blend_weight(p_a[mask], p_b[mask], actuals[mask], grid=grid)
        preds[i] = blend_probs(p_a[i], p_b[i], w_i)
    return preds


def leave_one_out(eval_matches: pd.DataFrame, all_matches: pd.DataFrame | None = None,
                  *, alpha: float, fifa: dict, fifa_scale: float, team_effects: bool = True,
                  goals_fallback: bool = False, sos_weighting: bool = False,
                  bivariate: bool = False):
    """LOO backtest.

    Trains on ``all_matches`` minus the held-out row and evaluates on each row
    of ``eval_matches``.  If ``all_matches`` is None, falls back to
    ``eval_matches`` (original WC-only behaviour).
    """
    if all_matches is None:
        all_matches = eval_matches
    preds, actuals = [], []
    for i in list(eval_matches.index):
        train = all_matches.drop(i, errors="ignore")
        test = eval_matches.loc[i]
        model = DixonColesRatings(alpha=alpha, fifa=fifa, fifa_scale=fifa_scale,
                                  team_effects=team_effects,
                                  goals_fallback=goals_fallback,
                                  sos_weighting=sos_weighting,
                                  bivariate=bivariate).fit(train)
        preds.append(model_probs(model, test["home"], test["away"]))
        actuals.append(actual_outcome(int(test["home_goals"]), int(test["away_goals"])))
    return np.array(preds), np.array(actuals)


def leave_one_out_ol(
    eval_matches: pd.DataFrame,
    all_matches: pd.DataFrame | None = None,
    *,
    C: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """LOO W/D/L predictions from a multinomial logistic regression on Elo/FIFA features.

    Fast: fits Elo once (pre-match ratings are already leakage-free by construction),
    then per fold fits a cheap 3-feature LR on the drop-one training set.

    Leakage note: the single global Elo fit means the held-out match influenced
    post-match Elo updates of later matches — identical approximation to the existing
    _elo_predictions_wc benchmark. Acceptable and consistent.
    """
    from footy.ratings.elo import fit_elo
    from footy.ratings.fifa import fifa_strength
    from footy.ratings.ordered_logit import build_features, fit_ordered_logit, predict_wdl_ol

    # Use config WC_SEASON_ID; fall back to None gracefully
    try:
        from footy.config import WC_SEASON_ID as _WC_SEASON_ID
    except ImportError:
        _WC_SEASON_ID = None

    if all_matches is None:
        all_matches = eval_matches

    # Fit Elo once on all_matches (leakage-free by construction of pre_match_ratings)
    elo = fit_elo(all_matches, wc_season_id=_WC_SEASON_ID)

    # FIFA z-scores for all teams in all_matches
    all_teams = sorted({*all_matches["home"], *all_matches["away"]})
    fifa_z = fifa_strength(all_teams)

    # Build full feature matrix from all_matches
    X_all, y_all = build_features(all_matches, elo, fifa_z)
    assert len(X_all) == len(all_matches), (
        f"build_features dropped rows ({len(X_all)} vs {len(all_matches)}); "
        "positional index mapping is invalid — check for NaN goals in all_matches"
    )
    all_idx = list(all_matches.index)  # positional mapping

    preds: list[np.ndarray] = []
    actuals: list[int] = []

    for i, row in eval_matches.iterrows():
        # Find position of this index in all_matches to drop it
        if i in all_idx:
            pos = all_idx.index(i)
            X_train = np.delete(X_all, pos, axis=0)
            y_train = np.delete(y_all, pos, axis=0)
        else:
            X_train = X_all
            y_train = y_all

        model = fit_ordered_logit(X_train, y_train, C=C)

        # Build single-row feature for this eval match
        x_row, _ = build_features(
            pd.DataFrame([row]),
            elo,
            fifa_z,
        )
        p = predict_wdl_ol(model, x_row)
        preds.append(p)
        actuals.append(actual_outcome(int(row["home_goals"]), int(row["away_goals"])))

    return np.array(preds), np.array(actuals)


def temporal_backtest(eval_matches: pd.DataFrame, all_matches: pd.DataFrame, *,
                      weight: float, alpha: float, fifa: dict, fifa_scale: float,
                      team_effects: bool = True, neutral: bool = True
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Strict chronological out-of-sample backtest.

    Unlike :func:`leave_one_out` (which trains on everything except the held-out
    row, including matches *after* it), this trains each prediction on only the
    matches with an earlier ``date`` — the honest test of genuine forecasting.

    If ``neutral``, home-advantage is cancelled by averaging the home/away
    orientations of the prediction (used for knockout-stage neutral-venue play).
    """
    from footy.ratings.ensemble import EnsemblePredictor

    preds, actuals = [], []
    dates = pd.to_datetime(all_matches["date"], utc=True)
    for _, row in eval_matches.iterrows():
        cutoff = pd.to_datetime(row["date"], utc=True)
        train = all_matches[dates < cutoff]
        model = EnsemblePredictor(alpha=alpha, fifa=fifa, fifa_scale=fifa_scale,
                                  team_effects=team_effects, weight=weight).fit(train)
        if neutral:
            fwd = model.wdl(row["home"], row["away"])
            rev = model.wdl(row["away"], row["home"])
            p = np.array([(fwd[0] + rev[2]) / 2, (fwd[1] + rev[1]) / 2, (fwd[2] + rev[0]) / 2])
            p = p / p.sum()
        else:
            p = model.wdl(row["home"], row["away"])
        preds.append(p)
        actuals.append(actual_outcome(int(row["home_goals"]), int(row["away_goals"])))
    return np.array(preds), np.array(actuals)


def naive_baseline(matches: pd.DataFrame):
    """Predict the empirical home/draw/away base rates for every match."""
    o = np.array([actual_outcome(int(h), int(a))
                  for h, a in zip(matches["home_goals"], matches["away_goals"])])
    rates = np.array([(o == k).mean() for k in range(3)])
    return np.tile(rates, (len(o), 1)), o


def per_match_rps(preds: np.ndarray, actuals: np.ndarray) -> np.ndarray:
    """Return the per-match RPS vector (length n).  Lower is better."""
    return np.array([rps(preds[i], actuals[i]) for i in range(len(actuals))])


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap the MEAN of a 1-D array and return a percentile CI.

    Returns {"mean": float, "lo": float, "hi": float}.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(boot_means, 100 * alpha)),
        "hi": float(np.percentile(boot_means, 100 * (1 - alpha))),
    }


def paired_bootstrap(rps_a: np.ndarray, rps_b: np.ndarray,
                     n_boot: int = 10_000, ci: float = 0.95, seed: int = 0) -> dict:
    """Paired bootstrap on the per-match difference d = rps_a - rps_b.

    The same match indices are resampled for both series (paired).

    Returns
    -------
    {"mean_diff": float, "lo": float, "hi": float, "p_a_better": float}
    where ``p_a_better`` is the fraction of bootstrap resamples in which
    mean(rps_a_boot) < mean(rps_b_boot).  Lower RPS is better.
    """
    rng = np.random.default_rng(seed)
    n = len(rps_a)
    diff = rps_a - rps_b
    boot_diffs = np.empty(n_boot)
    a_better = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = diff[idx].mean()
        boot_diffs[b] = d
        if d < 0:
            a_better += 1
    alpha = (1.0 - ci) / 2.0
    return {
        "mean_diff": float(diff.mean()),
        "lo": float(np.percentile(boot_diffs, 100 * alpha)),
        "hi": float(np.percentile(boot_diffs, 100 * (1 - alpha))),
        "p_a_better": float(a_better / n_boot),
    }


def score(preds: np.ndarray, actuals: np.ndarray) -> dict:
    n = len(actuals)
    picked = preds[np.arange(n), actuals]
    return {
        "n": n,
        "log_loss": float(-np.mean(np.log(np.clip(picked, 1e-9, 1.0)))),
        "rps": float(np.mean([rps(preds[i], actuals[i]) for i in range(n)])),
        "accuracy": float(np.mean(preds.argmax(1) == actuals)),
    }


def reliability(preds: np.ndarray, actuals: np.ndarray, n_bins: int = 5):
    """Pool all W/D/L probabilities and bin predicted vs observed frequency."""
    p = preds.ravel()
    hit = np.zeros_like(preds)
    hit[np.arange(len(actuals)), actuals] = 1.0
    hit = hit.ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & (p <= hi if b == n_bins - 1 else p < hi)
        if mask.sum():
            rows.append((float(p[mask].mean()), float(hit[mask].mean()), int(mask.sum())))
    return rows
