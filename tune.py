"""Hyperparameter grid search: Dixon-Coles alpha × fifa_scale.

Maps how the leave-one-out backtest score depends on the two key hyper-
parameters:
  alpha      -- Dixon-Coles time-decay constant (higher = faster decay)
  fifa_scale -- weight of the FIFA-ranking prior relative to match data

Saves a heatmap to tune_alpha_fifa.png.

OVERFITTING WARNING (printed at runtime):
  Tuning on the same 52 LOO matches used for evaluation can select noise.
  The goal here is to see whether the loss surface is FLAT near the shipped
  default (alpha=0.05, fifa_scale=1.0), which would *validate* the default,
  or whether there is a robust basin elsewhere that warrants investigation.

Usage:
    python3 tune.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from footy.config import WC_SEASON_ID  # noqa: E402
from footy.evaluate.backtest import leave_one_out, score  # noqa: E402
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
FIFA_SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]

DEFAULT_ALPHA = 0.05
DEFAULT_FIFA_SCALE = 1.0

OUTPUT_PNG = Path(__file__).resolve().parent / "tune_alpha_fifa.png"

OVERFITTING_NOTE = (
    "\n[NOTE] Overfitting risk: all 52 LOO cells are used both to tune and to report "
    "performance.\nA flat surface near the default validates the choice; a sharp "
    "isolated minimum is likely noise.\n"
)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(eval_matches, all_matches, fifa) -> list[dict]:
    """Run the full alpha × fifa_scale grid and return a list of result dicts."""
    n_cells = len(ALPHAS) * len(FIFA_SCALES)
    results = []
    best_rps = float("inf")
    cell_idx = 0

    for alpha in ALPHAS:
        for fs in FIFA_SCALES:
            cell_idx += 1
            t0 = time.time()
            preds, actuals = leave_one_out(
                eval_matches,
                all_matches,
                alpha=alpha,
                fifa=fifa,
                fifa_scale=fs,
                team_effects=True,
            )
            s = score(preds, actuals)
            elapsed = time.time() - t0
            is_best = s["rps"] < best_rps
            if is_best:
                best_rps = s["rps"]
            marker = " <-- BEST SO FAR" if is_best else ""
            print(
                f"[{cell_idx:2d}/{n_cells}] alpha={alpha:.2f}  fifa_scale={fs:.2f}"
                f"  RPS={s['rps']:.5f}  log_loss={s['log_loss']:.5f}"
                f"  ({elapsed:.1f}s){marker}"
            )
            results.append(
                {
                    "alpha": alpha,
                    "fifa_scale": fs,
                    "rps": s["rps"],
                    "log_loss": s["log_loss"],
                    "n": s["n"],
                }
            )
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: list[dict]) -> None:
    sorted_rows = sorted(results, key=lambda r: r["rps"])

    default_row = next(
        (r for r in results if r["alpha"] == DEFAULT_ALPHA and r["fifa_scale"] == DEFAULT_FIFA_SCALE),
        None,
    )
    best_row = sorted_rows[0]

    header = f"{'rank':>4}  {'alpha':>6}  {'fifa_scale':>10}  {'RPS':>9}  {'log_loss':>9}  {'note'}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for rank, row in enumerate(sorted_rows, 1):
        tags = []
        if row["alpha"] == DEFAULT_ALPHA and row["fifa_scale"] == DEFAULT_FIFA_SCALE:
            tags.append("DEFAULT")
        if row is best_row:
            tags.append("BEST")
        note = "  [" + ", ".join(tags) + "]" if tags else ""
        print(
            f"{rank:>4}  {row['alpha']:>6.2f}  {row['fifa_scale']:>10.2f}"
            f"  {row['rps']:>9.5f}  {row['log_loss']:>9.5f}{note}"
        )
    print("=" * len(header))

    if default_row is not None:
        delta_rps = best_row["rps"] - default_row["rps"]
        delta_rps_rel = delta_rps / default_row["rps"] * 100
        delta_ll = best_row["log_loss"] - default_row["log_loss"]
        delta_ll_rel = delta_ll / default_row["log_loss"] * 100
        print(
            f"\nDefault  (alpha={DEFAULT_ALPHA}, fifa_scale={DEFAULT_FIFA_SCALE}):"
            f"  RPS={default_row['rps']:.5f}  log_loss={default_row['log_loss']:.5f}"
        )
        print(
            f"Best     (alpha={best_row['alpha']}, fifa_scale={best_row['fifa_scale']}):"
            f"  RPS={best_row['rps']:.5f}  log_loss={best_row['log_loss']:.5f}"
        )
        print(
            f"Delta    RPS: {delta_rps:+.5f} ({delta_rps_rel:+.2f}%)"
            f"   log_loss: {delta_ll:+.5f} ({delta_ll_rel:+.2f}%)"
        )


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def save_heatmap(results: list[dict]) -> None:
    """Save a heatmap of mean RPS over (alpha, fifa_scale); lower = better = darker."""
    # Build 2-D arrays: rows = alpha index, cols = fifa_scale index
    rps_grid = np.zeros((len(ALPHAS), len(FIFA_SCALES)))
    for r in results:
        ai = ALPHAS.index(r["alpha"])
        fi = FIFA_SCALES.index(r["fifa_scale"])
        rps_grid[ai, fi] = r["rps"]

    best_rps = rps_grid.min()
    best_pos = np.unravel_index(rps_grid.argmin(), rps_grid.shape)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Use YlOrRd_r: dark purple = low RPS = better; light = worse
    im = ax.imshow(rps_grid, aspect="auto", cmap="YlOrRd_r",
                   vmin=rps_grid.min() * 0.995, vmax=rps_grid.max() * 1.005)

    # Annotate cells with RPS value
    for ai in range(len(ALPHAS)):
        for fi in range(len(FIFA_SCALES)):
            val = rps_grid[ai, fi]
            colour = "white" if val <= best_rps + 0.002 else "black"
            ax.text(fi, ai, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color=colour)

    # Mark best cell with a border
    from matplotlib.patches import Rectangle  # local import to keep top-level clean
    rect = Rectangle(
        (best_pos[1] - 0.5, best_pos[0] - 0.5), 1, 1,
        linewidth=2.5, edgecolor="#0055ff", facecolor="none",
    )
    ax.add_patch(rect)

    # Axes labels (categorical ticks for alpha)
    ax.set_xticks(range(len(FIFA_SCALES)))
    ax.set_xticklabels([str(fs) for fs in FIFA_SCALES])
    ax.set_yticks(range(len(ALPHAS)))
    ax.set_yticklabels([str(a) for a in ALPHAS])
    ax.set_xlabel("fifa_scale")
    ax.set_ylabel("alpha (time-decay)")
    ax.set_title(
        "LOO RPS by (alpha, fifa_scale) — lower is better\n"
        f"Blue border = best; default = alpha={DEFAULT_ALPHA}, fifa_scale={DEFAULT_FIFA_SCALE}"
    )

    fig.colorbar(im, ax=ax, label="mean RPS (lower = better)")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\nHeatmap saved to {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(OVERFITTING_NOTE)

    # Load data
    try:
        all_matches = build_match_table()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] build_match_table() failed: {exc}")
        print("Ensure data/raw and data/processed are populated before running tune.py.")
        sys.exit(0)

    if all_matches.empty:
        print("[ERROR] build_match_table() returned an empty DataFrame. Check your data directory.")
        sys.exit(0)

    wc_matches = all_matches[all_matches["season_id"] == WC_SEASON_ID].copy()
    if wc_matches.empty:
        print(f"[WARNING] No matches found for WC_SEASON_ID={WC_SEASON_ID!r}. Falling back to all matches.")
        wc_matches = all_matches.copy()

    all_teams = sorted({*all_matches["home"], *all_matches["away"]})
    fifa = fifa_strength(all_teams)

    n_qual = len(all_matches) - len(wc_matches)
    print(f"Training pool : {len(all_matches)} matches ({len(wc_matches)} WC + {n_qual} qualifiers)")
    print(f"Evaluation set: {len(wc_matches)} WC matches (leave-one-out)\n")
    n_cells = len(ALPHAS) * len(FIFA_SCALES)
    print(f"Grid: {len(ALPHAS)} alpha values × {len(FIFA_SCALES)} fifa_scale values = {n_cells} cells\n")

    results = run_grid(wc_matches, all_matches, fifa)

    print_table(results)
    print(OVERFITTING_NOTE)
    save_heatmap(results)


if __name__ == "__main__":
    main()
