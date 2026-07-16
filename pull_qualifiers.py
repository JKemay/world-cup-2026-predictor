#!/usr/bin/env python3
"""Discover WC 2026 qualifying competitions and cache match timelines.

Strategy:
  1. List all Sportradar competitions and filter to those whose name contains
     "qualif" + "world cup" (case-insensitive).
  2. For each, take the most recent season.
  3. Pull the schedule and keep only finished matches where at least one
     competitor is a WC 2026 team (by canonical name).
  4. Fetch and cache those timelines — idempotent, re-runs cost zero calls.

    python3 pull_qualifiers.py           # full run
    python3 pull_qualifiers.py --dry-run # discover only, no timeline pulls
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from footy.ingest.sportradar import SportradarError, SportradarSource  # noqa: E402
from footy.ratings.fifa import FIFA_RANK, normalize_team  # noqa: E402

WC_TEAMS: set[str] = set(FIFA_RANK.keys())
DRY_RUN = "--dry-run" in sys.argv


def _is_qualifier(name: str) -> bool:
    n = name.lower()
    if "women" in n or "qualif" not in n:
        return False
    if "world cup" in n:
        return True
    # AFC brands its WC qualifying competition without the literal "World Cup"
    # phrase ("AFC Asian Qualifiers 2026", sr:competition:308) — every other
    # confederation (UEFA/CAF/CONCACAF/CONMEBOL/OFC) says "FIFA World Cup
    # Qualification, <CONFED>", so this is a targeted allowlist, not a
    # loosened match — it must not also catch "AFC Asian Cup, Qualification"
    # (a different tournament) or similarly-named non-WC qualifiers.
    return "afc asian qualifiers" in n


def _is_finished(entry: dict) -> bool:
    st = entry.get("sport_event_status", {})
    return st.get("status") in ("closed", "ended") or st.get("match_status") == "ended"


def _involves_wc_team(entry: dict) -> bool:
    competitors = entry.get("sport_event", {}).get("competitors", [])
    return any(normalize_team(c.get("name", "")) in WC_TEAMS for c in competitors)


def main() -> None:
    src = SportradarSource()

    # --- 1. Discover qualifier competitions ---
    print("Listing all competitions...")
    all_comps = src.get("/competitions.json").get("competitions", [])
    qualifiers = [c for c in all_comps if _is_qualifier(c.get("name", ""))]
    print(f"Found {len(qualifiers)} WC qualifying competition(s):\n")
    for q in qualifiers:
        print(f"  {q['id']}  {q.get('name', '?')}")

    if not qualifiers:
        print("\nNo qualifier competitions found — check trial access or API name patterns.")
        return

    # --- 2. Find the most recent season per competition ---
    seasons_to_pull: list[tuple[str, str, str]] = []  # (comp_name, season_id, season_name)
    print()
    for comp in qualifiers:
        try:
            seasons = src.seasons(comp["id"])
        except SportradarError as e:
            print(f"  {comp.get('name', comp['id'])}: skipped — {e}")
            continue
        if not seasons:
            continue
        seasons.sort(key=lambda s: s.get("start_date", ""), reverse=True)
        recent = seasons[0]
        seasons_to_pull.append((comp.get("name", ""), recent["id"], recent.get("name", "")))
        print(f"  {comp.get('name', '?')}  →  {recent.get('name', recent['id'])}")

    if DRY_RUN:
        print("\n--dry-run: stopping before timeline pulls.")
        return

    # --- 3. Pull schedules and filter to WC team matches ---
    all_entries: list[dict] = []
    print()
    for comp_name, season_id, season_name in seasons_to_pull:
        try:
            entries = src.schedule(season_id)
        except SportradarError as e:
            print(f"  {comp_name}: schedule error — {e}")
            continue
        finished = [e for e in entries if _is_finished(e)]
        relevant = [e for e in finished if _involves_wc_team(e)]
        print(f"  {comp_name} / {season_name}")
        print(f"    {len(entries)} total  |  {len(finished)} finished  |  {len(relevant)} involve a WC team")
        all_entries.extend(relevant)

    print(f"\nTotal matches to fetch: {len(all_entries)}")
    if not all_entries:
        print("Nothing to pull.")
        return

    # --- 4. Fetch timelines ---
    failures = total_events = total_shots = 0
    for i, entry in enumerate(all_entries, 1):
        ev = entry["sport_event"]
        names_str = " vs ".join(c.get("name", "?") for c in ev.get("competitors", []))
        try:
            tl = src.timeline(ev["id"])
        except SportradarError as err:
            print(f"[{i:>3}/{len(all_entries)}] {names_str:<44} FAILED — {err}")
            failures += 1
            continue
        events = tl.get("timeline", [])
        shots = [x for x in events if "shot" in (x.get("type") or "")]
        total_events += len(events)
        total_shots += len(shots)
        print(f"[{i:>3}/{len(all_entries)}] {names_str:<44} {len(events):>4} events  {len(shots):>3} shots")

    print(f"\nDone — {len(all_entries) - failures}/{len(all_entries)} timelines cached")
    print(f"  API calls: {src.calls_made}   cache hits: {src.cache_hits}")
    print(f"  New events: {total_events:,}   new shots: {total_shots:,}")
    print("\nNext steps:")
    print("  python3 build_xg.py       # retrain xG on all shots (WC + qualifiers)")
    print("  python3 build_eval.py     # re-run backtest — should now see Full > FIFA-only")


if __name__ == "__main__":
    main()
