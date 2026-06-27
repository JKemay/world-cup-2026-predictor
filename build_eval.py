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
    apply_draw_scalar,
    bootstrap_ci,
    fit_draw_scalar,
    leave_one_out,
    leave_one_out_ol,
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


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    row_sums = arr.sum(axis=1, keepdims=True)
    return arr / row_sums


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
    print(f"Evaluating on: {len(eval_matches)} WC matches (leave-one-out)")

    qual_matches = all_matches[all_matches["season_id"] != WC_SEASON_ID]
    def draw_rate(df):
        return (df["home_goals"] == df["away_goals"]).mean()
    print(f"\nDraw rates — WC eval: {draw_rate(eval_matches):.1%}  |  "
          f"Qualifiers: {draw_rate(qual_matches):.1%}  |  "
          f"All training: {draw_rate(all_matches):.1%}")
    print()

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

    # Draw scalar calibration
    k_star = fit_draw_scalar(p_ens, a_ens)
    p_ens_cal = apply_draw_scalar(p_ens, k_star)
    print(f"\nFitted draw scalar k = {k_star:.3f}")

    def draws_correct(preds, actuals):
        return int(((preds.argmax(axis=1) == 1) & (actuals == 1)).sum())

    n_draws = int((a_ens == 1).sum())
    print(f"Draws predicted correctly: {draws_correct(p_ens, a_ens)}/{n_draws}  ->  "
          f"{draws_correct(p_ens_cal, a_ens)}/{n_draws}  (after draw scalar)")

    # Bivariate Poisson variant
    pf_bp, af_bp = leave_one_out(eval_matches, all_matches, alpha=ALPHA, fifa=fifa,
                                 fifa_scale=FIFA_SCALE, team_effects=True, bivariate=True)
    p_ens_bp = normalize_rows(0.5 * pf_bp + 0.5 * pe)
    k_star_bp = fit_draw_scalar(p_ens_bp, a_ens)
    p_ens_bp_cal = apply_draw_scalar(p_ens_bp, k_star_bp)
    print(f"Bivariate draw scalar k = {k_star_bp:.3f}")
    print(f"Draws correct (bivariate+cal): {draws_correct(p_ens_bp_cal, a_ens)}/{n_draws}")

    # Ordered logit (multinomial LR on Elo/FIFA features)
    print("\nFitting ordered logit LOO (this may take a moment)...")
    p_ol, a_ol = leave_one_out_ol(eval_matches, all_matches)

    # Three-way equal blend
    p_ens3 = (pf + pe + p_ol) / 3.0
    p_ens3 /= p_ens3.sum(axis=1, keepdims=True)

    # Optimized weights — coarse grid search over 2-simplex, step 0.1
    best_rps, best_w = 1.0, (1 / 3, 1 / 3, 1 / 3)
    for w1 in np.arange(0.0, 1.01, 0.1):
        for w2 in np.arange(0.0, 1.01 - w1, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9:
                continue
            p_blend = w1 * pf + w2 * pe + w3 * p_ol
            p_blend /= p_blend.sum(axis=1, keepdims=True)
            r = score(p_blend, af)["rps"]
            if r < best_rps:
                best_rps, best_w = r, (w1, w2, w3)
    p_ens_opt = best_w[0] * pf + best_w[1] * pe + best_w[2] * p_ol
    p_ens_opt /= p_ens_opt.sum(axis=1, keepdims=True)
    print(f"Optimal blend weights: DC={best_w[0]:.2f} Elo={best_w[1]:.2f} OL={best_w[2]:.2f}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    table = {
        "Ensemble (xG + Elo)": score(p_ens, a_ens),
        "Ensemble + draw cal": score(p_ens_cal, a_ens),
        "Ensemble (bivariate + draw cal)": score(p_ens_bp_cal, a_ens),
        "Full (xG + FIFA + form)": score(pf, af),
        "Elo benchmark": score(pe, ae),
        "Ordered logit (OL)": score(p_ol, a_ol),
        "Ensemble 1/3+1/3+1/3": score(p_ens3, af),
        "Ensemble (optimized weights)": score(p_ens_opt, af),
        "FIFA-only": score(pq, aq),
        "Naive base-rate": score(pn, an),
    }
    print(f"{'model':<36}{'log-loss':>10}{'RPS':>8}{'top-1':>8}")
    print("-" * 62)
    for name, s in table.items():
        print(f"{name:<36}{s['log_loss']:>10.4f}{s['rps']:>8.4f}{s['accuracy']*100:>7.0f}%")

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

    print(f"\nDraws correct — OL: {draws_correct(p_ol, a_ol)}/{n_draws}  "
          f"| 3-way equal: {draws_correct(p_ens3, af)}/{n_draws}  "
          f"| Optimized: {draws_correct(p_ens_opt, af)}/{n_draws}")

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

    rps_ens_cal = per_match_rps(p_ens_cal, a_ens)

    diff_ef = paired_bootstrap(rps_ens, rps_full, n_boot=N_BOOT, seed=SEED)
    diff_fo = paired_bootstrap(rps_full, rps_fifa, n_boot=N_BOOT, seed=SEED)
    diff_en = paired_bootstrap(rps_ens, rps_naive, n_boot=N_BOOT, seed=SEED)
    diff_cal = paired_bootstrap(rps_ens_cal, rps_ens, n_boot=N_BOOT, seed=SEED)

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

    cal_straddles = diff_cal["lo"] < 0 < diff_cal["hi"]
    cal_interp = "not statistically distinguishable on this sample" if cal_straddles else (
        "Draw calibration significantly better" if diff_cal["mean_diff"] < 0 else "Raw ensemble significantly better"
    )
    print(f"  Ens+cal vs Ens (raw)  : mean ΔRPS {diff_cal['mean_diff']:+.4f}  "
          f"95% CI [{diff_cal['lo']:+.4f}, {diff_cal['hi']:+.4f}]  "
          f"P(cal better)={diff_cal['p_a_better']:.3f}")
    print(f"    -> {cal_interp}")

    rps_ens_opt = per_match_rps(p_ens_opt, af)
    diff_opt_ens = paired_bootstrap(rps_ens_opt, rps_ens, n_boot=N_BOOT, seed=SEED)
    opt_straddles = diff_opt_ens["lo"] < 0 < diff_opt_ens["hi"]
    if opt_straddles:
        opt_interp = "not statistically distinguishable on this sample"
    elif diff_opt_ens["mean_diff"] < 0:
        opt_interp = "Optimized blend significantly better"
    else:
        opt_interp = "Ensemble (xG+Elo) significantly better"
    print(f"  Opt blend vs Ens      : mean ΔRPS {diff_opt_ens['mean_diff']:+.4f}  "
          f"95% CI [{diff_opt_ens['lo']:+.4f}, {diff_opt_ens['hi']:+.4f}]  "
          f"P(Opt better)={diff_opt_ens['p_a_better']:.3f}")
    print(f"    -> {opt_interp}")

    # ------------------------------------------------------------------
    # Calibration for the ensemble model
    # ------------------------------------------------------------------
    rel = reliability(p_ens, a_ens, n_bins=5)
    print("\nCalibration — Ensemble (predicted -> observed):")
    for pp, oo, nn in rel:
        print(f"  {pp*100:4.0f}%  ->  {oo*100:4.0f}%   (n={nn})")
    plot_reliability(rel, PROJECT_ROOT / "calibration.png")

    # Write calibrated ensemble preds to backtest.csv
    out = eval_matches[["home", "away", "home_goals", "away_goals"]].copy()
    out[["p_home", "p_draw", "p_away"]] = p_ens_cal
    out["outcome"] = a_ens
    out.to_csv(DATA_DIR / "processed" / "backtest.csv", index=False)
    print("\nSaved: calibration.png, data/processed/backtest.csv")


if __name__ == "__main__":
    main()
