#!/usr/bin/env python3
"""Fit Dixon-Coles team ratings and produce the France vs Iraq scoreline grid.

    python3 build_ratings.py

Outputs:
  data/processed/team_ratings.csv   per-team attack/defense ratings
  france_iraq_grid.png              scoreline probability heatmap
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from footy.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from footy.features.matches import build_match_table  # noqa: E402
from footy.ratings.dixon_coles import DixonColesRatings, grid_summary  # noqa: E402
from footy.ratings.fifa import fifa_strength  # noqa: E402

ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05      # L2 on team adjustments
FIFA_SCALE = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # weight on the FIFA prior


def plot_grid(grid, home, away, lam, mu, out_path):
    pct = grid * 100.0
    fig, ax = plt.subplots(figsize=(7, 5.6))
    im = ax.imshow(pct, origin="lower", cmap="YlGnBu")
    hi = pct.max()
    for h in range(pct.shape[0]):
        for a in range(pct.shape[1]):
            if pct[h, a] >= 0.05:
                ax.text(a, h, f"{pct[h, a]:.1f}", ha="center", va="center",
                        fontsize=8, color="white" if pct[h, a] > hi * 0.55 else "black")
    ax.set_xticks(range(pct.shape[1]))
    ax.set_yticks(range(pct.shape[0]))
    ax.set_xlabel(f"{away} goals   (expected {mu:.2f})")
    ax.set_ylabel(f"{home} goals   (expected {lam:.2f})")
    ax.set_title(f"Poisson scoreline grid: {home} vs {away}")
    fig.colorbar(im, ax=ax, label="joint probability (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    matches = build_match_table()
    n_teams = len({*matches.home, *matches.away})
    tot_g = (matches.home_goals + matches.away_goals).mean()
    tot_x = (matches.home_xg + matches.away_xg).mean()
    print(f"{len(matches)} matches, {n_teams} teams")
    print(f"avg goals/match {tot_g:.2f}   avg xG/match {tot_x:.2f}")

    fifa = fifa_strength(sorted({*matches.home, *matches.away}))
    r = DixonColesRatings(alpha=ALPHA, response="xg", fifa=fifa, fifa_scale=FIFA_SCALE).fit(matches)
    print(f"home advantage x{np.exp(r.home_adv_):.2f}   rho {r.rho_:+.3f}   "
          f"fifa coef: att {r.fifa_attack_coef_:+.3f}  def {r.fifa_defense_coef_:+.3f}")

    table = r.ratings_frame()
    table.to_csv(DATA_DIR / "processed" / "team_ratings.csv", index=False)
    fmt = lambda v: f"{v:.2f}"
    print("\nTop 8 (att_xg / def_xg_allowed / net):")
    print(table.head(8).to_string(index=False, formatters={c: fmt for c in ["att_xg", "def_xg_allowed", "net"]}))
    print("\nBottom 5:")
    print(table.tail(5).to_string(index=False, formatters={c: fmt for c in ["att_xg", "def_xg_allowed", "net"]}))

    ranks = {t: i for i, t in enumerate(table.team, 1)}
    print("\nWhere the usual suspects land:")
    for name in ("France", "Spain", "Argentina", "Brazil", "England", "Iraq", "Qatar"):
        if name in ranks:
            row = table[table.team == name].iloc[0]
            print(f"  #{ranks[name]:>2}/{len(table)}  {name:<10} "
                  f"att {row.att_xg:.2f}  def_allowed {row.def_xg_allowed:.2f}")

    for name in ("France", "Iraq"):
        if name not in r.teams_:
            print(f"\n!! '{name}' not found. Teams: {sorted(r.teams_)}")
            return

    grid, lam, mu = r.scoreline_grid("France", "Iraq", max_goals=6)
    s = grid_summary(grid)
    print("\n=== France vs Iraq ===")
    print(f"  expected goals : France {lam:.2f}   Iraq {mu:.2f}")
    print(f"  likely score   : France {s['top_score'][0]}-{s['top_score'][1]} ({s['top_prob']*100:.1f}%)")
    print(f"  outcome        : France {s['home_win']*100:.1f}%  draw {s['draw']*100:.1f}%  Iraq {s['away_win']*100:.1f}%")
    print("  reference model: France 3-0, 90.6% / 7.2% / 2.2%")

    plot_grid(grid, "France", "Iraq", lam, mu, PROJECT_ROOT / "france_iraq_grid.png")
    print("\nSaved: data/processed/team_ratings.csv, france_iraq_grid.png")


if __name__ == "__main__":
    main()
