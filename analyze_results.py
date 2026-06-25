"""analyze_results.py — Post-backtest analysis for the WC football predictor.

Loads:
  data/processed/backtest.csv     — leave-one-out backtest match rows
  data/processed/team_ratings.csv — xG-based team ratings

Prints key model metrics to stdout and writes a 3-panel figure to
model_analysis.png (uses the Agg backend; no display required).

Usage:
    python3 analyze_results.py
"""

from __future__ import annotations

import os
import sys

# Allow `from footy...` imports when running from the repo root.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BACKTEST_CSV = os.path.join(_REPO_ROOT, "data", "processed", "backtest.csv")
RATINGS_CSV = os.path.join(_REPO_ROOT, "data", "processed", "team_ratings.csv")
OUTPUT_PNG = os.path.join(_REPO_ROOT, "model_analysis.png")

OUTCOME_LABEL = {0: "H", 1: "D", 2: "A"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str, label: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"[analyze_results] Missing input file: {path}  ({label}) — skipping.")
        return None
    df = pd.read_csv(path)
    print(f"[analyze_results] Loaded {label}: {len(df)} rows from {path}")
    return df


def _prob_of_actual(row: pd.Series) -> float:
    """Return the predicted probability the model assigned to the actual outcome."""
    col = ["p_home", "p_draw", "p_away"][int(row["outcome"])]
    return float(row[col])


def _rps(row: pd.Series) -> float:
    """Ranked Probability Score for a single 3-outcome row (lower = better)."""
    probs = np.array([row["p_home"], row["p_draw"], row["p_away"]], dtype=float)
    outcome = int(row["outcome"])
    actuals = np.zeros(3)
    actuals[outcome] = 1.0
    cum_pred = np.cumsum(probs)
    cum_act = np.cumsum(actuals)
    return float(np.sum((cum_pred - cum_act) ** 2) / 2.0)


# ---------------------------------------------------------------------------
# Panel builders
# ---------------------------------------------------------------------------

def panel_calibration(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Reliability diagram pooling all three outcome probabilities."""
    # Flatten: one row per (match x outcome)
    records = []
    for _, row in df.iterrows():
        for outcome_idx, prob_col in enumerate(["p_home", "p_draw", "p_away"]):
            records.append({
                "predicted": float(row[prob_col]),
                "actual": int(int(row["outcome"]) == outcome_idx),
            })
    cal_df = pd.DataFrame(records)

    n_bins = 5
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, obs_freq, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (cal_df["predicted"] >= lo) & (cal_df["predicted"] < hi)
        subset = cal_df[mask]
        if len(subset) == 0:
            continue
        bin_centers.append(float(subset["predicted"].mean()))
        obs_freq.append(float(subset["actual"].mean()))
        counts.append(len(subset))

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration", zorder=1)
    ax.scatter(bin_centers, obs_freq, s=80, zorder=3, color="steelblue",
               edgecolors="white", linewidths=0.8, label="Model bins")
    ax.plot(bin_centers, obs_freq, color="steelblue", lw=1.5, zorder=2)

    for bx, by, n in zip(bin_centers, obs_freq, counts):
        ax.annotate(f"n={n}", xy=(bx, by), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7,
                    color="dimgray")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_ylabel("Observed frequency", fontsize=9)
    ax.set_title("Calibration / Reliability", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def panel_surprises(ax: plt.Axes, df: pd.DataFrame, top_n: int = 8) -> None:
    """Horizontal bar chart of the matches where the model was most wrong."""
    df = df.copy()
    df["p_actual"] = df.apply(_prob_of_actual, axis=1)
    worst = df.nsmallest(top_n, "p_actual").iloc[::-1]  # ascending for hbar

    labels = [
        f"{r['home']} vs {r['away']}  (actual: {OUTCOME_LABEL[int(r['outcome'])]})"
        for _, r in worst.iterrows()
    ]
    values = worst["p_actual"].tolist()

    colors = plt.cm.RdYlGn(np.array(values))  # type: ignore[attr-defined]
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Model's predicted prob of actual result", fontsize=9)
    ax.set_title("Biggest Surprises (model most wrong)", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=8, color="black",
        )
    ax.grid(axis="x", alpha=0.3)


def panel_team_landscape(ax: plt.Axes, ratings: pd.DataFrame) -> None:
    """Scatter of att vs def for rated WC teams; invert y so good = upper-right."""
    try:
        from footy.ratings.fifa import FIFA_RANK
    except ImportError:
        FIFA_RANK = {}
        print("[analyze_results] Could not import FIFA_RANK — showing all rated teams.")

    if FIFA_RANK:
        wc_teams = set(FIFA_RANK.keys())
        sub = ratings[ratings["team"].isin(wc_teams)].copy()
    else:
        sub = ratings.copy()

    if sub.empty:
        ax.text(0.5, 0.5, "No team data to display", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Team Ratings Landscape", fontsize=10, fontweight="bold")
        return

    med_att = sub["att_xg"].median()
    med_def = sub["def_xg_allowed"].median()

    ax.axvline(med_att, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_def, color="gray", lw=0.8, ls="--", alpha=0.6)

    # Color by net rating
    sc = ax.scatter(
        sub["att_xg"], sub["def_xg_allowed"],
        c=sub["net"], cmap="RdYlGn", s=60,
        edgecolors="white", linewidths=0.6, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="net rating", pad=0.02)

    ax.invert_yaxis()  # low goals-allowed = "up" = good

    # Label top 10 by net
    top10 = sub.nlargest(10, "net")
    for _, row in top10.iterrows():
        ax.annotate(
            row["team"],
            xy=(row["att_xg"], row["def_xg_allowed"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="black",
        )

    ax.set_xlabel("Attack xG (higher = more dangerous)", fontsize=9)
    ax.set_ylabel("Def xG allowed (lower = stingier)", fontsize=9)
    ax.set_title("Team Ratings Landscape (WC teams)", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------------------------------------
# Printed insights
# ---------------------------------------------------------------------------

def print_insights(df: pd.DataFrame, ratings: pd.DataFrame | None) -> None:
    print("\n" + "=" * 60)
    print("  MODEL INSIGHTS SUMMARY")
    print("=" * 60)

    # Overall accuracy
    preds = df[["p_home", "p_draw", "p_away"]].values
    predicted_outcome = np.argmax(preds, axis=1)
    accuracy = float((predicted_outcome == df["outcome"].values).mean())
    print(f"\nOverall accuracy (argmax vs actual): {accuracy:.1%}")

    # Mean RPS
    rps_scores = df.apply(_rps, axis=1)
    print(f"Mean RPS (lower = better):           {rps_scores.mean():.4f}")
    print("  (random baseline RPS ≈ 0.333)")

    # Outcome distribution
    actual_counts = df["outcome"].value_counts().sort_index()
    outcome_names = {0: "Home wins", 1: "Draws", 2: "Away wins"}
    print(f"\nOutcome distribution in backtest ({len(df)} matches):")
    for k, name in outcome_names.items():
        n = actual_counts.get(k, 0)
        print(f"  {name:12s}: {n:3d}  ({n/len(df):.1%})")

    # Biggest surprises
    df_s = df.copy()
    df_s["p_actual"] = df_s.apply(_prob_of_actual, axis=1)
    surprises = df_s.nsmallest(3, "p_actual")
    print("\nTop-3 biggest surprises (model most wrong):")
    for _, row in surprises.iterrows():
        act = OUTCOME_LABEL[int(row["outcome"])]
        print(f"  {row['home']} vs {row['away']}  "
              f"→ actual: {act},  model gave it {row['p_actual']:.1%}")

    # Team ratings
    if ratings is not None and not ratings.empty:
        print("\nTop-5 teams by net rating:")
        for _, row in ratings.nlargest(5, "net").iterrows():
            print(f"  {row['team']:20s}  net={row['net']:.3f}  "
                  f"att={row['att_xg']:.3f}  def_allowed={row['def_xg_allowed']:.3f}")
        print("\nBottom-5 teams by net rating:")
        for _, row in ratings.nsmallest(5, "net").iterrows():
            print(f"  {row['team']:20s}  net={row['net']:.3f}  "
                  f"att={row['att_xg']:.3f}  def_allowed={row['def_xg_allowed']:.3f}")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    backtest = _load_csv(BACKTEST_CSV, "backtest")
    ratings = _load_csv(RATINGS_CSV, "team_ratings")

    if backtest is None:
        print("[analyze_results] Cannot proceed without backtest.csv — exiting.")
        sys.exit(0)

    # Ensure correct dtypes
    for col in ["p_home", "p_draw", "p_away"]:
        backtest[col] = backtest[col].astype(float)
    backtest["outcome"] = backtest["outcome"].astype(int)

    # --- Printed insights -----------------------------------------------
    print_insights(backtest, ratings)

    # --- Figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Football Predictor — Model Analysis", fontsize=13, fontweight="bold",
                 y=1.01)

    panel_calibration(axes[0], backtest)
    panel_surprises(axes[1], backtest)

    if ratings is not None and not ratings.empty:
        panel_team_landscape(axes[2], ratings)
    else:
        axes[2].text(0.5, 0.5, "team_ratings.csv not available",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Team Ratings Landscape", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze_results] Figure saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
