#!/usr/bin/env python3
"""Elo benchmark: fit World-Football-style Elo and evaluate on WC 2026 matches.

    python3 build_elo.py

Produces a comparison table (Elo vs Full model vs FIFA-only vs Naive),
a paired bootstrap significance test (Elo vs Full model), and the current
Elo top-10 teams.

Optionally saves data/processed/elo_ratings.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from footy.config import DATA_DIR, WC_SEASON_ID  # noqa: E402
from footy.evaluate.backtest import (  # noqa: E402
    actual_outcome,
    leave_one_out,
    paired_bootstrap,
    per_match_rps,
    score,
)
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.elo import EloRatings, fit_elo, predict_wdl  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

ALPHA, FIFA_SCALE_DC = 0.05, 1.0


def _elo_predictions_wc(elo: EloRatings, wc_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, actuals) for WC matches using leakage-free pre-match ratings."""
    preds, actuals = [], []
    for _, row in wc_matches.iterrows():
        mid = row["match_id"]
        if mid not in elo.pre_match_ratings:
            # Fallback: use final ratings (should not happen)
            r_h = elo.ratings.get(row["home"], 1500.0)
            r_a = elo.ratings.get(row["away"], 1500.0)
        else:
            r_h, r_a = elo.pre_match_ratings[mid]
        p = predict_wdl(elo, r_h, r_a)
        preds.append(p)
        actuals.append(actual_outcome(int(row["home_goals"]), int(row["away_goals"])))
    return np.array(preds), np.array(actuals)


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    all_matches = build_match_table()
    wc_matches = all_matches[all_matches["season_id"] == WC_SEASON_ID].copy()
    eval_matches = wc_matches if len(wc_matches) else all_matches

    n_qual = len(all_matches) - len(eval_matches)
    print(f"Training: {len(all_matches)} matches ({len(eval_matches)} WC + {n_qual} qualifiers)")
    print(f"Evaluating on: {len(eval_matches)} WC matches\n")

    # ------------------------------------------------------------------
    # 2. Fit Elo (chronologically; draw params on non-WC data only)
    # ------------------------------------------------------------------
    elo = fit_elo(all_matches, wc_season_id=WC_SEASON_ID)
    print(f"Draw model: D0={elo.draw_params.D0:.4f}  DW={elo.draw_params.DW:.1f}")

    # ------------------------------------------------------------------
    # 3. Predict WC matches (leakage-free pre-match ratings)
    # ------------------------------------------------------------------
    pe, ae = _elo_predictions_wc(elo, eval_matches)
    elo_scores = score(pe, ae)

    # ------------------------------------------------------------------
    # 4. Full model predictions (LOO backtest, same as build_eval.py)
    # ------------------------------------------------------------------
    all_teams = sorted({*all_matches.home, *all_matches.away})
    fifa = fifa_strength(all_teams)

    print("Running LOO backtest for Full model (this may take ~30 s)...")
    pf, af = leave_one_out(
        eval_matches, all_matches,
        alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE_DC, team_effects=True,
    )
    full_scores = score(pf, af)

    # ------------------------------------------------------------------
    # 5. Known reference numbers (from build_eval.py runs)
    # ------------------------------------------------------------------
    known = {
        "Full (xG+FIFA+form)": {"rps": full_scores["rps"],
                                "log_loss": full_scores["log_loss"],
                                "accuracy": full_scores["accuracy"]},
        "Elo benchmark": {"rps": elo_scores["rps"],
                         "log_loss": elo_scores["log_loss"],
                         "accuracy": elo_scores["accuracy"]},
        # Hard-coded reference values from the project's known results
        "FIFA-only (ref)": {"rps": 0.1618, "log_loss": None, "accuracy": None},
        "Naive base-rate (ref)": {"rps": 0.2019, "log_loss": None, "accuracy": None},
    }

    # ------------------------------------------------------------------
    # 6. Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'Model':<28}{'log-loss':>10}{'RPS':>10}{'top-1':>8}")
    print("-" * 56)
    for name, s in known.items():
        ll = f"{s['log_loss']:.4f}" if s["log_loss"] is not None else "  —"
        rps_str = f"{s['rps']:.4f}"
        acc = f"{s['accuracy']*100:.0f}%" if s["accuracy"] is not None else "  —"
        print(f"{name:<28}{ll:>10}{rps_str:>10}{acc:>8}")

    full_rps = full_scores["rps"]
    elo_rps = elo_scores["rps"]
    print(f"\nElo vs Full model : ΔRPS {elo_rps - full_rps:+.4f}  "
          f"({'Elo worse' if elo_rps > full_rps else 'Elo better'})")

    # ------------------------------------------------------------------
    # 7. Paired bootstrap: Elo vs Full model
    # ------------------------------------------------------------------
    N_BOOT, SEED = 10_000, 0
    rps_elo = per_match_rps(pe, ae)
    rps_full = per_match_rps(pf, af)

    bs = paired_bootstrap(rps_elo, rps_full, n_boot=N_BOOT, seed=SEED)
    ci_str = f"[{bs['lo']:+.4f}, {bs['hi']:+.4f}]"

    print(f"\nPaired bootstrap (N={N_BOOT:,}, 95% CI) — Elo minus Full model:")
    print(f"  mean ΔRPS = {bs['mean_diff']:+.4f}   95% CI {ci_str}")
    print(f"  P(Elo better than Full) = {bs['p_a_better']:.3f}")
    straddles = bs["lo"] < 0 < bs["hi"]
    if straddles:
        print("  -> not statistically distinguishable on this sample")
    elif bs["mean_diff"] < 0:
        print("  -> Elo significantly better than Full model")
    else:
        print("  -> Full model significantly better than Elo")

    # ------------------------------------------------------------------
    # 8. Elo top-10 (sanity check)
    # ------------------------------------------------------------------
    top10 = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nElo top-10 teams (final ratings after all matches):")
    for rank, (team, rating) in enumerate(top10, 1):
        print(f"  {rank:2d}. {team:<25} {rating:.1f}")

    # ------------------------------------------------------------------
    # 9. Save optional CSV
    # ------------------------------------------------------------------
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    ratings_df = pd.DataFrame(
        [(t, r) for t, r in sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)],
        columns=["team", "elo_rating"],
    )
    ratings_df.to_csv(out_dir / "elo_ratings.csv", index=False)
    print(f"\nSaved: data/processed/elo_ratings.csv ({len(ratings_df)} teams)")


if __name__ == "__main__":
    main()
