"""Extract a clean, shot-level dataset from cached Sportradar timelines.

Each row is one shot (or shot-goal) with pitch coordinates, normalised so every
team attacks toward x=100. Notes on the event taxonomy (verified from the data):

* ``shot_on_target`` / ``shot_off_target`` — non-goal shots, carry x/y.
* ``score_change`` with ``method == 'shot'`` — a goal from open play, carries x/y.
* ``score_change`` with ``method`` of ``penalty`` / ``own_goal`` — flagged and
  excluded from the open-play xG dataset.
* ``shot_saved`` — intentionally ignored: it is the keeper's-side duplicate of an
  opponent ``shot_on_target`` (same timestamp, opposite competitor), so counting
  it would double-count the shot.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from footy.config import RAW_DIR
from footy.ratings.fifa import normalize_team

NON_GOAL_SHOTS = {"shot_on_target", "shot_off_target"}


def _timeline_files(cache_dir: Path) -> list[str]:
    return sorted(glob.glob(str(Path(cache_dir) / "*timeline*.json")))


def load_shots(cache_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Return a tidy DataFrame of all shots across the cached matches."""
    rows: list[dict] = []
    skipped_no_xy = 0
    for path in _timeline_files(cache_dir):
        data = json.loads(Path(path).read_text())
        se = data.get("sport_event", {})
        names = {c.get("qualifier"): normalize_team(c.get("name", ""))
                 for c in se.get("competitors", [])}
        match_id = se.get("id")
        for e in data.get("timeline", []):
            etype = e.get("type")
            is_goal_event = etype == "score_change"
            if etype not in NON_GOAL_SHOTS and not is_goal_event:
                continue
            x, y = e.get("x"), e.get("y")
            if x is None or y is None:
                skipped_no_xy += 1
                continue
            method = e.get("method", "shot")  # plain shots are open play
            side = e.get("competitor")          # 'home' or 'away'
            opp = "away" if side == "home" else "home"
            rows.append(
                {
                    "match_id": match_id,
                    "team": names.get(side),
                    "opponent": names.get(opp),
                    "side": side,
                    "minute": e.get("match_time"),
                    "type": etype,
                    "method": method,
                    # any scored shot counts (foot, header, direct free-kick);
                    # penalties and own goals are excluded from the open-play model
                    "is_goal": int(is_goal_event and method not in ("own_goal", "penalty")),
                    "is_penalty": int(method == "penalty"),
                    "is_own_goal": int(method == "own_goal"),
                    "x": x,
                    "y": y,
                    # normalise so every team attacks toward x=100
                    "x_norm": x if side == "home" else 100 - x,
                    "y_norm": y if side == "home" else 100 - y,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["skipped_no_xy"] = skipped_no_xy
    return df
