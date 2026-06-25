#!/usr/bin/env python3
"""Leave-one-out backtest: ensemble vs full model vs FIFA-only vs naive base-rate.

    python3 build_eval.py

Outputs:
  calibration.png                reliability curve for the ensemble model
  data/processed/backtest.csv    per-match ensemble predicted probabilities + outcome
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from footy.config import DATA_DIR, PROJECT_ROOT, WC_SEASON_ID  # noqa: E402
from footy.evaluate.backtest import (  # noqa: E402
    actual_outcome,
    bootstrap_ci,
    leave_one_out,
    naive_baseline,
    paired_bootstrap,
    per_match_rps,
    reliability,
    score,
)
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.elo import fit_elo, predict_wdl  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

ALPHA, FIFA_SCALE = 0.05, 1.0


def _elo_predictions_wc(all_matches, eval_matches):
    """Leakage-free Elo predictions for WC matches (pre-match ratings).

    Fits Elo on *all_matches* with draw-params calibrated on non-WC data,
    then uses the pre-match Elo ratings stored for each WC match.  The
    predictions are returned in the same row-order as *eval_matches*.
    """
    elo = fit_elo(all_matches, wc_season_id=WC_SEASON_ID)
    preds, actuals = [], []
    for _, row in eval_matches.iterrows():
        mid = row["match_id"]
        if mid in elo.pre_match_ratings:
            r_h, r_a = elo.pre_match_ratings[mid]
        else:
            r_h = elo.ratings.get(row["home"], 1500.0)
            r_a = elo.ratings.get(row["away"], 1500.0)
        preds.append(predict_wdl(elo, r_h, r_a))
        actuals.append(actual_outcome(int(row["home_goals"]), int(row["away_goals"])))
    return np.array(preds), np.array(actuals)


def plot_reliability(rel, out_path):
    xs, ys, ns = zip(*rel)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#1f77b4", label="model")
    for x, y, nn in zip(xs, ys, ns):
        ax.annotate(f"n={nn}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability — W/D/L probabilities (leave-one-out, ensemble)")
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

    # ------------------------------------------------------------------
    # Model predictions
    # ------------------------------------------------------------------
    pf, af = leave_one_out(eval_matches, all_matches, alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE, team_effects=True)
    pq, aq = leave_one_out(eval_matches, all_matches, alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE, team_effects=False)
    pn, an = naive_baseline(eval_matches)

    # Elo: leakage-free pre-match WC predictions
    pe, ae = _elo_predictions_wc(all_matches, eval_matches)

    # Ensemble: 50/50 blend of Full LOO (xG/DC) and Elo per-match predictions
    p_ens_raw = 0.5 * pf + 0.5 * pe
    row_sums = p_ens_raw.sum(axis=1, keepdims=True)
    p_ens = p_ens_raw / row_sums
    a_ens = af  # same actuals as Full model (same eval_matches order)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    table = {
        "Ensemble (xG + Elo)": score(p_ens, a_ens),
        "Full (xG + FIFA + form)": score(pf, af),
        "Elo benchmark": score(pe, ae),
        "FIFA-only": score(pq, aq),
        "Naive base-rate": score(pn, an),
    }
    print(f"{'model':<28}{'log-loss':>10}{'RPS':>8}{'top-1':>8}")
    print("-" * 54)
    for name, s in table.items():
        print(f"{name:<28}{s['log_loss']:>10.4f}{s['rps']:>8.4f}{s['accuracy']*100:>7.0f}%")

    ens = table["Ensemble (xG + Elo)"]
    full = table["Full (xG + FIFA + form)"]
    fo = table["FIFA-only"]
    nb = table["Naive base-rate"]
    print(f"\nEnsemble vs Full  : RPS {(1 - ens['rps']/full['rps'])*100:+.1f}%   "
          f"log-loss {(1 - ens['log_loss']/full['log_loss'])*100:+.1f}%")
    print(f"Full vs FIFA-only : RPS {(1 - full['rps']/fo['rps'])*100:+.1f}%   "
          f"log-loss {(1 - full['log_loss']/fo['log_loss'])*100:+.1f}%")
    print(f"Ensemble vs naive : RPS {(1 - ens['rps']/nb['rps'])*100:+.1f}%   "
          f"log-loss {(1 - ens['log_loss']/nb['log_loss'])*100:+.1f}%")

    # ------------------------------------------------------------------
    # Statistical significance (paired bootstrap, N=10 000 resamples)
    # ------------------------------------------------------------------
    N_BOOT, SEED = 10_000, 0
    rps_ens = per_match_rps(p_ens, a_ens)
    rps_full = per_match_rps(pf, af)
    rps_elo = per_match_rps(pe, ae)
    rps_fifa = per_match_rps(pq, aq)
    rps_naive = per_match_rps(pn, an)

    ci_ens = bootstrap_ci(rps_ens, n_boot=N_BOOT, seed=SEED)
    ci_full = bootstrap_ci(rps_full, n_boot=N_BOOT, seed=SEED)
    ci_elo = bootstrap_ci(rps_elo, n_boot=N_BOOT, seed=SEED)
    ci_fifa = bootstrap_ci(rps_fifa, n_boot=N_BOOT, seed=SEED)
    ci_naive = bootstrap_ci(rps_naive, n_boot=N_BOOT, seed=SEED)

    diff_ef = paired_bootstrap(rps_ens, rps_full, n_boot=N_BOOT, seed=SEED)
    diff_fo = paired_bootstrap(rps_full, rps_fifa, n_boot=N_BOOT, seed=SEED)
    diff_en = paired_bootstrap(rps_ens, rps_naive, n_boot=N_BOOT, seed=SEED)

    print(f"\nStatistical significance (paired bootstrap, N={N_BOOT:,} resamples, 95 % CI)")
    print("-" * 74)
    print(f"  Ensemble     : mean RPS {ci_ens['mean']:.4f}  95% CI [{ci_ens['lo']:.4f}, {ci_ens['hi']:.4f}]")
    print(f"  Full model   : mean RPS {ci_full['mean']:.4f}  95% CI [{ci_full['lo']:.4f}, {ci_full['hi']:.4f}]")
    print(f"  Elo          : mean RPS {ci_elo['mean']:.4f}  95% CI [{ci_elo['lo']:.4f}, {ci_elo['hi']:.4f}]")
    print(f"  FIFA-only    : mean RPS {ci_fifa['mean']:.4f}  95% CI [{ci_fifa['lo']:.4f}, {ci_fifa['hi']:.4f}]")
    print(f"  Naive        : mean RPS {ci_naive['mean']:.4f}  95% CI [{ci_naive['lo']:.4f}, {ci_naive['hi']:.4f}]")
    print()

    ef_straddles = diff_ef["lo"] < 0 < diff_ef["hi"]
    ef_interp = "not statistically distinguishable on this sample" if ef_straddles else (
        "Ensemble significantly better" if diff_ef["mean_diff"] < 0 else "Full model significantly better"
    )
    print(f"  Ensemble vs Full      : mean ΔRPS {diff_ef['mean_diff']:+.4f}  "
          f"95% CI [{diff_ef['lo']:+.4f}, {diff_ef['hi']:+.4f}]  "
          f"P(Ensemble better)={diff_ef['p_a_better']:.3f}")
    print(f"    -> {ef_interp}")

    fo_straddles = diff_fo["lo"] < 0 < diff_fo["hi"]
    fo_interp = "not statistically distinguishable on this sample" if fo_straddles else (
        "Full significantly better" if diff_fo["mean_diff"] < 0 else "FIFA-only significantly better"
    )
    print(f"  Full vs FIFA-only     : mean ΔRPS {diff_fo['mean_diff']:+.4f}  "
          f"95% CI [{diff_fo['lo']:+.4f}, {diff_fo['hi']:+.4f}]  "
          f"P(Full better)={diff_fo['p_a_better']:.3f}")
    print(f"    -> {fo_interp}")

    en_straddles = diff_en["lo"] < 0 < diff_en["hi"]
    en_interp = "not statistically distinguishable on this sample" if en_straddles else (
        "Ensemble significantly better" if diff_en["mean_diff"] < 0 else "Naive significantly better"
    )
    print(f"  Ensemble vs Naive     : mean ΔRPS {diff_en['mean_diff']:+.4f}  "
          f"95% CI [{diff_en['lo']:+.4f}, {diff_en['hi']:+.4f}]  "
          f"P(Ensemble better)={diff_en['p_a_better']:.3f}")
    print(f"    -> {en_interp}")

    # ------------------------------------------------------------------
    # Calibration for the ensemble model
    # ------------------------------------------------------------------
    rel = reliability(p_ens, a_ens, n_bins=5)
    print("\nCalibration — Ensemble (predicted -> observed):")
    for pp, oo, nn in rel:
        print(f"  {pp*100:4.0f}%  ->  {oo*100:4.0f}%   (n={nn})")
    plot_reliability(rel, PROJECT_ROOT / "calibration.png")

    # Write ensemble preds to backtest.csv
    out = eval_matches[["home", "away", "home_goals", "away_goals"]].copy()
    out[["p_home", "p_draw", "p_away"]] = p_ens
    out["outcome"] = a_ens
    out.to_csv(DATA_DIR / "processed" / "backtest.csv", index=False)
    print("\nSaved: calibration.png, data/processed/backtest.csv")


if __name__ == "__main__":
    main()
