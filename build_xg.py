#!/usr/bin/env python3
"""Build the shot dataset, train the xG model, evaluate it, and save artifacts.

    python3 build_xg.py

Outputs:
  footy/models/xg_logreg.joblib   trained model
  data/processed/shots_xg.csv     every open-play shot with its xG
  xg_pitch.png                    xG-by-location heatmap
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from footy.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from footy.features.shots import load_shots  # noqa: E402
from footy.features.xg import (  # noqa: E402
    PITCH_LENGTH,
    PITCH_WIDTH,
    add_geometry,
    predict_xg,
    train_xg,
)

MODELS_DIR = PROJECT_ROOT / "footy" / "models"
PROC_DIR = DATA_DIR / "processed"


def pitch_heatmap(model, out_path: Path) -> None:
    xs = np.linspace(50, 100, 60)
    ys = np.linspace(0, 100, 80)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"x_norm": gx.ravel(), "y_norm": gy.ravel()})
    grid = add_geometry(grid)
    grid["xg"] = predict_xg(model, grid)
    z = grid["xg"].to_numpy().reshape(gx.shape)

    L, W = PITCH_LENGTH, PITCH_WIDTH
    fig, ax = plt.subplots(figsize=(7, 5.2))
    im = ax.imshow(
        z, origin="lower", extent=[50 / 100 * L, L, 0, W],
        aspect="equal", cmap="viridis", vmin=0,
    )
    line = dict(color="white", lw=1.2)
    # penalty box
    ax.plot([L - 16.5, L - 16.5], [34 - 20.15, 34 + 20.15], **line)
    ax.plot([L - 16.5, L], [34 - 20.15, 34 - 20.15], **line)
    ax.plot([L - 16.5, L], [34 + 20.15, 34 + 20.15], **line)
    # six-yard box
    ax.plot([L - 5.5, L - 5.5], [34 - 9.16, 34 + 9.16], **line)
    ax.plot([L - 5.5, L], [34 - 9.16, 34 - 9.16], **line)
    ax.plot([L - 5.5, L], [34 + 9.16, 34 + 9.16], **line)
    # goal
    ax.plot([L, L], [34 - 3.66, 34 + 3.66], color="white", lw=4)
    fig.colorbar(im, ax=ax, label="xG (goal probability)")
    ax.set_title("Expected goals by shot location (logistic xG)")
    ax.set_xlabel("distance toward goal (m)")
    ax.set_ylabel("pitch width (m)")
    ax.set_xlim(50 / 100 * L, L)
    ax.set_ylim(0, W)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    shots = load_shots()
    print(f"Loaded {len(shots)} shot events "
          f"(skipped {shots.attrs.get('skipped_no_xy', 0)} without coordinates)")
    print("  by type:", shots["type"].value_counts().to_dict())
    gc = shots[shots.type == "score_change"]["method"].value_counts().to_dict()
    print("  goal methods:", gc)

    # Only train on matches where shot tracking looks complete: require ≥5 non-goal
    # shots per match so CAF matches (goals tracked but regular shots not) don't inflate
    # conversion rate with "goals-without-misses".
    misses_per_match = shots[shots.is_goal == 0].groupby("match_id").size()
    complete_match_ids = set(misses_per_match[misses_per_match >= 5].index)
    n_complete = len(complete_match_ids)
    n_all = shots["match_id"].nunique()
    print(f"  matches with complete shot tracking: {n_complete}/{n_all}"
          f"  ({n_all - n_complete} excluded — provider returned no shot coordinates)")

    openplay = shots[(shots.is_penalty == 0) & (shots.is_own_goal == 0)
                     & shots["match_id"].isin(complete_match_ids)].copy()
    openplay = add_geometry(openplay)
    model, cv_prob, m = train_xg(openplay)

    improve = (1 - m["cv_logloss"] / m["baseline_logloss"]) * 100
    print("\nxG model — logistic regression on (distance, angle):")
    print(f"  shots={m['n_shots']}  goals={m['n_goals']}  "
          f"conversion={m['conversion'] * 100:.1f}%")
    print(f"  CV log-loss {m['cv_logloss']:.4f}  vs baseline {m['baseline_logloss']:.4f}  "
          f"({improve:+.1f}% better than guessing the base rate)")
    print(f"  CV ROC-AUC  {m['cv_auc']:.3f}    CV Brier {m['cv_brier']:.4f}")

    openplay["xg"] = predict_xg(model, openplay)
    print(f"  calibration check: total xG {openplay.xg.sum():.1f}  "
          f"vs actual goals {int(openplay.is_goal.sum())}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "xg_logreg.joblib")
    openplay.to_csv(PROC_DIR / "shots_xg.csv", index=False)
    pitch_heatmap(model, PROJECT_ROOT / "xg_pitch.png")

    # a few sanity-check examples across the pitch
    examples = pd.DataFrame({"x_norm": [99, 94, 88, 83, 75], "y_norm": [50, 50, 38, 50, 50]})
    examples = add_geometry(examples)
    examples["xg"] = predict_xg(model, examples)
    print("\n  sample xG (centre unless noted):")
    for _, r in examples.iterrows():
        print(f"    {r.distance:5.1f} m, angle {np.degrees(r.angle):4.0f}° -> xG {r.xg:.3f}")

    print("\nSaved: footy/models/xg_logreg.joblib, data/processed/shots_xg.csv, xg_pitch.png")


if __name__ == "__main__":
    main()
