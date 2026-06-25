"""Leave-one-out backtest of the rating model with proper scoring rules.

For each match we refit the ratings on the *other* matches and predict the
held-out one, so nothing leaks. Outcomes are scored with multiclass log-loss and
the Ranked Probability Score (RPS) — the standard metric for ordered
home/draw/away football outcomes — plus top-1 accuracy and a reliability curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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


def leave_one_out(eval_matches: pd.DataFrame, all_matches: pd.DataFrame | None = None,
                  *, alpha: float, fifa: dict, fifa_scale: float, team_effects: bool = True):
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
                                  team_effects=team_effects).fit(train)
        preds.append(model_probs(model, test["home"], test["away"]))
        actuals.append(actual_outcome(int(test["home_goals"]), int(test["away_goals"])))
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
