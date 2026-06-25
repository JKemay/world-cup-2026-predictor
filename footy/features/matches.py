"""Build a match-level table (final scores + xG per side) for rating models."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from footy.config import DATA_DIR, RAW_DIR
from footy.ratings.fifa import normalize_team

SHOTS_XG_CSV = DATA_DIR / "processed" / "shots_xg.csv"


def _final_score(timeline: list[dict]) -> tuple[int, int]:
    """Final score = running home/away score carried on score_change events."""
    home = away = 0
    for e in timeline:
        if e.get("type") == "score_change":
            home = max(home, e.get("home_score", home))
            away = max(away, e.get("away_score", away))
    return home, away


def build_match_table(cache_dir: Path = RAW_DIR, shots_csv: Path = SHOTS_XG_CSV) -> pd.DataFrame:
    xg = pd.read_csv(shots_csv)
    xg_by = xg.groupby(["match_id", "team"])["xg"].sum()

    rows: list[dict] = []
    for path in sorted(glob.glob(str(Path(cache_dir) / "*timeline*.json"))):
        data = json.loads(Path(path).read_text())
        se = data.get("sport_event", {})
        names = {c.get("qualifier"): normalize_team(c.get("name", ""))
                 for c in se.get("competitors", [])}
        home, away = names.get("home"), names.get("away")
        if not home or not away:
            continue
        mid = se.get("id")
        hg, ag = _final_score(data.get("timeline", []))
        ctx = se.get("sport_event_context", {})
        season_id = ctx.get("season", {}).get("id", "")
        rows.append(
            {
                "match_id": mid,
                "date": se.get("start_time"),
                "season_id": season_id,
                "home": home,
                "away": away,
                "home_goals": hg,
                "away_goals": ag,
                "home_xg": round(float(xg_by.get((mid, home), 0.0)), 3),
                "away_xg": round(float(xg_by.get((mid, away), 0.0)), 3),
            }
        )
    return pd.DataFrame(rows)
