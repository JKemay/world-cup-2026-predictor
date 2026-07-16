#!/usr/bin/env python3
"""Strict temporal (chronological) out-of-sample backtest on the WC 2026 knockouts.

Unlike leave-one-out (which trains on everything except the held-out match,
including matches *after* it), this predicts each knockout match using a model
trained *only* on matches with an earlier date — the honest test of genuine
forecasting rather than in-sample fit. See docs/METHODOLOGY.md §7a.

    python3 backtest_temporal.py               # weight=0.5 (shipped default)
    python3 backtest_temporal.py --weight 0.3   # test an alternative blend weight
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from footy.config import KNOCKOUT_START, WC_SEASON_ID  # noqa: E402
from footy.evaluate.backtest import actual_outcome, score, temporal_backtest  # noqa: E402
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

ALPHA, FIFA_SCALE = 0.05, 1.0

# Actual shootout winners for the 90'-drawn knockout matches, sourced from the
# user-reported tournament results this session. Hand-maintained: deriving
# "who advanced" from later-round bracket appearance breaks for the Final.
SHOOTOUT_WINNERS = {
    frozenset({"Germany", "Paraguay"}): "Paraguay",
    frozenset({"Netherlands", "Morocco"}): "Morocco",
    frozenset({"Australia", "Egypt"}): "Egypt",
    frozenset({"Switzerland", "Colombia"}): "Switzerland",
}


def naive_temporal(eval_matches: pd.DataFrame, all_matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Naive baseline refit at each cutoff: training-set base rates, not the global rate."""
    preds, actuals = [], []
    dates = pd.to_datetime(all_matches["date"], utc=True)
    for _, row in eval_matches.iterrows():
        cutoff = pd.to_datetime(row["date"], utc=True)
        train = all_matches[dates < cutoff]
        o = np.array([actual_outcome(int(h), int(a))
                      for h, a in zip(train["home_goals"], train["away_goals"])])
        rates = np.array([(o == k).mean() for k in range(3)]) if len(o) else np.array([1 / 3, 1 / 3, 1 / 3])
        preds.append(rates)
        actuals.append(actual_outcome(int(row["home_goals"]), int(row["away_goals"])))
    return np.array(preds), np.array(actuals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, default=0.5,
                        help="Ensemble blend weight on Dixon-Coles (default 0.5, the shipped value)")
    parser.add_argument("--shootout", action="store_true",
                        help="Also print the shootout-advancement plausibility check (n=4, anecdotal)")
    args = parser.parse_args()

    all_matches = build_match_table()
    wc = all_matches[all_matches["season_id"] == WC_SEASON_ID].copy()
    wc["_date"] = pd.to_datetime(wc["date"], utc=True)
    cutoff = pd.to_datetime(KNOCKOUT_START, utc=True)
    knockouts = wc[wc["_date"] >= cutoff].sort_values("_date").drop(columns="_date")

    print(f"Temporal out-of-sample backtest — weight={args.weight}")
    print(f"Knockout matches identified: {len(knockouts)} (cutoff {KNOCKOUT_START})\n")

    fifa = fifa_strength(sorted({*all_matches.home, *all_matches.away}))

    preds, actuals = temporal_backtest(
        knockouts, all_matches, weight=args.weight, alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE,
    )
    p_naive, a_naive = naive_temporal(knockouts, all_matches)

    s = score(preds, actuals)
    s_naive = score(p_naive, a_naive)
    print(f"Ensemble (weight={args.weight})   RPS {s['rps']:.4f}   log-loss {s['log_loss']:.4f}   "
          f"top-1 {s['accuracy']*100:.0f}% ({int(s['accuracy']*len(actuals))}/{len(actuals)})")
    print(f"Naive (refit per cutoff)   RPS {s_naive['rps']:.4f}")
    print(f"RPS improvement vs naive: {(1 - s['rps']/s_naive['rps'])*100:+.1f}%")

    n = len(knockouts)
    # Round of 32 is always 16 matches; everything chronologically after it
    # (Round of 16 onward) is scored separately. Matches are pre-sorted by date.
    r32_n = min(16, n)
    r32, r16plus = preds[:r32_n], preds[r32_n:]
    a32, a16plus = actuals[:r32_n], actuals[r32_n:]
    s32 = score(r32, a32)
    print(f"\nRound of 32 ({r32_n}): RPS {s32['rps']:.4f}  top-1 "
          f"{int(s32['accuracy']*r32_n)}/{r32_n} ({s32['accuracy']*100:.0f}%)")
    if n > r32_n:
        rest_n = n - r32_n
        s16 = score(r16plus, a16plus)
        print(f"Round of 16 onward ({rest_n}): RPS {s16['rps']:.4f}  top-1 "
              f"{int(s16['accuracy']*rest_n)}/{rest_n} ({s16['accuracy']*100:.0f}%)")

    fav_conf = preds.max(axis=1)
    hits = (preds.argmax(axis=1) == actuals)
    print(f"\nFavorite pick hit rate: {hits.sum()}/{n} ({hits.mean()*100:.0f}%) | "
          f"mean favorite confidence: {fav_conf.mean()*100:.0f}%")

    misses = knockouts.reset_index(drop=True)[~hits]
    if len(misses):
        print("\nMissed picks (model favorite != actual result):")
        for i, row in misses.iterrows():
            orig_idx = knockouts.reset_index(drop=True).index.get_loc(i)
            lbl = {0: "home win", 1: "draw", 2: "away win"}[actuals[orig_idx]]
            print(f"  {row['home']} vs {row['away']}: actual={lbl}, "
                  f"fav conf {fav_conf[orig_idx]*100:.0f}%")

    if args.shootout:
        from footy.ratings.ensemble import EnsemblePredictor
        from footy.ratings.shootout import advancement_prob

        print("\nShootout plausibility check — n=4, anecdotal only, not statistically powered.")
        dates = pd.to_datetime(all_matches["date"], utc=True)
        for _, row in knockouts.iterrows():
            key = frozenset({row["home"], row["away"]})
            if key not in SHOOTOUT_WINNERS:
                continue
            cutoff_i = pd.to_datetime(row["date"], utc=True)
            train = all_matches[dates < cutoff_i]
            model = EnsemblePredictor(alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE,
                                      weight=args.weight).fit(train)
            gap = model.elo_.ratings.get(row["home"], 1500.0) - model.elo_.ratings.get(row["away"], 1500.0)
            # Neutral-venue W/D/L (average both orientations) — knockout draws
            # going to extra time/pens are neutral-venue, same convention as
            # temporal_backtest(neutral=True).
            fwd = model.wdl(row["home"], row["away"])
            rev = model.wdl(row["away"], row["home"])
            wdl_neutral = np.array([(fwd[0] + rev[2]) / 2, (fwd[1] + rev[1]) / 2, (fwd[2] + rev[0]) / 2])
            wdl_neutral = wdl_neutral / wdl_neutral.sum()
            p_home_adv, p_away_adv = advancement_prob(wdl_neutral, gap)
            winner = SHOOTOUT_WINNERS[key]
            print(f"  {row['home']} vs {row['away']}: P(advance) "
                  f"{row['home']}={p_home_adv*100:.0f}% / {row['away']}={p_away_adv*100:.0f}%  "
                  f"(actual shootout winner: {winner})")


if __name__ == "__main__":
    main()
