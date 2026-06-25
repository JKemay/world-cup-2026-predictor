#!/usr/bin/env python3
"""Leave-one-out backtest: full model vs FIFA-only vs naive base-rate.

    python3 build_eval.py

Outputs:
  calibration.png                reliability curve for the full model
  data/processed/backtest.csv    per-match predicted probabilities + outcome
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import pandas as pd  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from footy.config import DATA_DIR, PROJECT_ROOT, WC_SEASON_ID  # noqa: E402
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402
from footy.evaluate.backtest import (  # noqa: E402
    leave_one_out, naive_baseline, reliability, score,
)

ALPHA, FIFA_SCALE = 0.05, 1.0


def plot_reliability(rel, out_path):
    xs, ys, ns = zip(*rel)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f77b4", label="model")
    for x, y, nn in zip(xs, ys, ns):
        ax.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability — W/D/L probabilities (leave-one-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    all_matches = build_match_table()
    wc_matches = all_matches[all_matches["season_id"] == WC_SEASON_ID].copy()
    # fall back to all if season_id not populated (e.g. before qualifier pull)
    eval_matches = wc_matches if len(wc_matches) else all_matches

    all_teams = sorted({*all_matches.home, *all_matches.away})
    fifa = fifa_strength(all_teams)

    n_qual = len(all_matches) - len(eval_matches)
    print(f"Training: {len(all_matches)} matches ({len(eval_matches)} WC + {n_qual} qualifiers)")
    print(f"Evaluating on: {len(eval_matches)} WC matches (leave-one-out)\n")

    pf, af = leave_one_out(eval_matches, all_matches, alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE, team_effects=True)
    pq, aq = leave_one_out(eval_matches, all_matches, alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE, team_effects=False)
    pn, an = naive_baseline(eval_matches)

    table = {
        "Full (xG + FIFA + form)": score(pf, af),
        "FIFA-only": score(pq, aq),
        "Naive base-rate": score(pn, an),
    }
    print(f"{'model':<26}{'log-loss':>10}{'RPS':>8}{'top-1':>8}")
    print("-" * 52)
    for name, s in table.items():
        print(f"{name:<26}{s['log_loss']:>10.4f}{s['rps']:>8.4f}{s['accuracy']*100:>7.0f}%")

    full, fo = table["Full (xG + FIFA + form)"], table["FIFA-only"]
    print(f"\nFull vs FIFA-only : RPS {(1 - full['rps']/fo['rps'])*100:+.1f}%   "
          f"log-loss {(1 - full['log_loss']/fo['log_loss'])*100:+.1f}%")
    nb = table["Naive base-rate"]
    print(f"Full vs naive     : RPS {(1 - full['rps']/nb['rps'])*100:+.1f}%   "
          f"log-loss {(1 - full['log_loss']/nb['log_loss'])*100:+.1f}%")

    rel = reliability(pf, af, n_bins=5)
    print("\nCalibration (predicted -> observed):")
    for pp, oo, nn in rel:
        print(f"  {pp*100:4.0f}%  ->  {oo*100:4.0f}%   (n={nn})")
    plot_reliability(rel, PROJECT_ROOT / "calibration.png")

    out = eval_matches[["home", "away", "home_goals", "away_goals"]].copy()
    out[["p_home", "p_draw", "p_away"]] = pf
    out["outcome"] = af
    out.to_csv(DATA_DIR / "processed" / "backtest.csv", index=False)
    print("\nSaved: calibration.png, data/processed/backtest.csv")


if __name__ == "__main__":
    main()
