"""Expected-goals (xG) geometry and a logistic-regression model.

Shot geometry mirrors the standard approach (and the reference model): distance
to goal centre and the angle the goal mouth subtends at the shot location. The
model is plain logistic regression on those two features — interpretable, and
evaluated out-of-sample so the reported quality is honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
GOAL_Y = PITCH_WIDTH / 2          # 34.0 m — centre of the goal
GOAL_HALF_WIDTH = 7.32 / 2        # 3.66 m — half the goal mouth
FEATURES = ["distance", "angle"]


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add `distance` (m) and `angle` (rad) columns from normalised coords."""
    df = df.copy()
    x_m = df["x_norm"] / 100.0 * PITCH_LENGTH
    y_m = df["y_norm"] / 100.0 * PITCH_WIDTH
    df["distance"] = np.hypot(PITCH_LENGTH - x_m, GOAL_Y - y_m)
    a = np.hypot(PITCH_LENGTH - x_m, (GOAL_Y - GOAL_HALF_WIDTH) - y_m)
    b = np.hypot(PITCH_LENGTH - x_m, (GOAL_Y + GOAL_HALF_WIDTH) - y_m)
    c = 2 * GOAL_HALF_WIDTH
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    df["angle"] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return df


def train_xg(df: pd.DataFrame):
    """Fit the xG model and return (model, cv_probabilities, metrics)."""
    X = df[FEATURES].to_numpy()
    y = df["is_goal"].to_numpy()
    model = LogisticRegression(max_iter=2000)
    # 5-fold out-of-sample probabilities → honest metrics (no train-set leakage)
    cv_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    model.fit(X, y)
    base_rate = float(y.mean())
    metrics = {
        "n_shots": int(len(y)),
        "n_goals": int(y.sum()),
        "conversion": base_rate,
        "cv_logloss": float(log_loss(y, cv_prob)),
        "baseline_logloss": float(log_loss(y, np.full_like(cv_prob, base_rate, dtype=float))),
        "cv_auc": float(roc_auc_score(y, cv_prob)),
        "cv_brier": float(brier_score_loss(y, cv_prob)),
    }
    return model, cv_prob, metrics


def predict_xg(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].to_numpy())[:, 1]
