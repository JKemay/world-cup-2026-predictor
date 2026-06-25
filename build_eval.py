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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from footy.config import DATA_DIR, PROJECT_ROOT, WC_SEASON_ID  # noqa: E402
from footy.evaluate.backtest import (  # noqa: E402
    bootstrap_ci,
    leave_one_out,
    naive_baseline,
    paired_bootstrap,
    per_match_rps,
    reliability,
    score,
)
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

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

    # ------------------------------------------------------------------
    # Statistical significance (paired bootstrap, N=10 000 resamples)
    # ------------------------------------------------------------------
    N_BOOT, SEED = 10_000, 0
    rps_full = per_match_rps(pf, af)
    rps_fifa = per_match_rps(pq, aq)
    rps_naive = per_match_rps(pn, an)

    ci_full = bootstrap_ci(rps_full, n_boot=N_BOOT, seed=SEED)
    ci_fifa = bootstrap_ci(rps_fifa, n_boot=N_BOOT, seed=SEED)
    ci_naive = bootstrap_ci(rps_naive, n_boot=N_BOOT, seed=SEED)

    diff_fo = paired_bootstrap(rps_full, rps_fifa, n_boot=N_BOOT, seed=SEED)
    diff_nb = paired_bootstrap(rps_full, rps_naive, n_boot=N_BOOT, seed=SEED)

    print(f"\nStatistical significance (paired bootstrap, N={N_BOOT:,} resamples, 95 % CI)")
    print("-" * 70)
    print(f"  Full model   : mean RPS {ci_full['mean']:.4f}  95% CI [{ci_full['lo']:.4f}, {ci_full['hi']:.4f}]")
    print(f"  FIFA-only    : mean RPS {ci_fifa['mean']:.4f}  95% CI [{ci_fifa['lo']:.4f}, {ci_fifa['hi']:.4f}]")
    print(f"  Naive        : mean RPS {ci_naive['mean']:.4f}  95% CI [{ci_naive['lo']:.4f}, {ci_naive['hi']:.4f}]")
    print()
    fo_straddles = diff_fo["lo"] < 0 < diff_fo["hi"]
    fo_interp = "not statistically distinguishable on this sample" if fo_straddles else (
        "Full significantly better" if diff_fo["mean_diff"] < 0 else "FIFA-only significantly better"
    )
    print(f"  Full vs FIFA-only : mean ΔRPS {diff_fo['mean_diff']:+.4f}  "
          f"95% CI [{diff_fo['lo']:+.4f}, {diff_fo['hi']:+.4f}]  "
          f"P(Full better)={diff_fo['p_a_better']:.3f}")
    print(f"    -> {fo_interp}")
    nb_straddles = diff_nb["lo"] < 0 < diff_nb["hi"]
    nb_interp = "not statistically distinguishable on this sample" if nb_straddles else (
        "Full significantly better" if diff_nb["mean_diff"] < 0 else "Naive significantly better"
    )
    print(f"  Full vs Naive     : mean ΔRPS {diff_nb['mean_diff']:+.4f}  "
          f"95% CI [{diff_nb['lo']:+.4f}, {diff_nb['hi']:+.4f}]  "
          f"P(Full better)={diff_nb['p_a_better']:.3f}")
    print(f"    -> {nb_interp}")

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
