"""Multinomial logistic regression W/D/L predictor (draw-focused feature engineering)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from footy.ratings.elo import HOME_ADV


def build_features(
    matches: pd.DataFrame,
    elo_ratings,
    fifa_z: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N,3) feature matrix and (N,) label vector from a matches DataFrame.

    Features: elo_diff (home-adjusted, /400), fifa_z_diff, abs(elo_diff).
    Uses pre_match_ratings[match_id] when available (LOO), else final ratings.
    Returns (X, y) where y encodes 0=home win, 1=draw, 2=away win.

    Rows where home_goals or away_goals is NaN are skipped.
    """
    rows_x: list[list[float]] = []
    rows_y: list[int] = []

    for _, row in matches.iterrows():
        hg = row.get("home_goals")
        ag = row.get("away_goals")
        if hg is None or ag is None:
            continue
        try:
            if np.isnan(float(hg)) or np.isnan(float(ag)):
                continue
        except (TypeError, ValueError):
            continue

        home = row["home"]
        away = row["away"]
        mid = row.get("match_id")

        # Pre-match ratings — use stored pair when available (LOO path)
        if mid is not None and mid in elo_ratings.pre_match_ratings:
            r_home, r_away = elo_ratings.pre_match_ratings[mid]
        else:
            r_home = elo_ratings.ratings.get(home, 1500.0)
            r_away = elo_ratings.ratings.get(away, 1500.0)

        elo_diff = (r_home - r_away + HOME_ADV) / 400.0
        fifa_diff = np.nan_to_num(
            fifa_z.get(home, 0.0) - fifa_z.get(away, 0.0), nan=0.0
        )
        abs_diff = abs(elo_diff)

        rows_x.append([float(elo_diff), float(fifa_diff), float(abs_diff)])

        hg_int, ag_int = int(hg), int(ag)
        if hg_int > ag_int:
            label = 0  # home win
        elif hg_int == ag_int:
            label = 1  # draw
        else:
            label = 2  # away win
        rows_y.append(label)

    X = np.array(rows_x, dtype=np.float64)
    y = np.array(rows_y, dtype=np.int64)
    return X, y


def fit_ordered_logit(X: np.ndarray, y: np.ndarray, *, C: float = 1.0) -> dict:
    """Fit scaler + LogisticRegression.

    Parameters
    ----------
    X : (N, 3) float array of features from build_features.
    y : (N,) int array of labels (0=home win, 1=draw, 2=away win).
    C : inverse regularisation strength for LogisticRegression.

    Returns
    -------
    {"scaler": StandardScaler, "clf": LogisticRegression}
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=1000,
    )
    clf.fit(X_scaled, y)
    return {"scaler": scaler, "clf": clf}


def predict_wdl_ol(model: dict, x_row: np.ndarray) -> np.ndarray:
    """Apply scaler then predict_proba for one row.

    Parameters
    ----------
    model : dict with keys "scaler" and "clf" (from fit_ordered_logit).
    x_row : (1, 3) or (3,) feature row.

    Returns
    -------
    (3,) array [P_home_win, P_draw, P_away_win], clipped and renormalised.
    """
    scaler: StandardScaler = model["scaler"]
    clf: LogisticRegression = model["clf"]

    x = np.asarray(x_row, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    x_scaled = scaler.transform(x)
    raw_proba = clf.predict_proba(x_scaled)[0]  # shape (n_classes,)

    # Reindex to [0, 1, 2] = [home, draw, away] regardless of model.classes_ order
    classes = list(clf.classes_)
    p = np.zeros(3, dtype=np.float64)
    for idx, cls in enumerate(classes):
        p[int(cls)] = raw_proba[idx]

    # Clip and renormalise
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()
    return p
