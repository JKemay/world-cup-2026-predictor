#!/usr/bin/env python3
"""Pull all finished World Cup 2026 match timelines into the local cache.

Idempotent: every response is cached, so re-running costs zero API calls.

    python3 pull_worldcup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `footy` importable

from footy.config import WC_SEASON_ID  # noqa: E402
from footy.ingest.sportradar import SportradarError, SportradarSource  # noqa: E402


def is_finished(entry: dict) -> bool:
    st = entry.get("sport_event_status", {})
    return st.get("status") in ("closed", "ended") or st.get("match_status") == "ended"


def main() -> None:
    src = SportradarSource()
    print(f"Loading World Cup 2026 schedule ({WC_SEASON_ID})...")
    entries = src.schedule(WC_SEASON_ID)
    finished = [e for e in entries if is_finished(e)]
    print(f"{len(entries)} scheduled, {len(finished)} finished — pulling timelines\n")

    total_events = total_shots = failures = 0
    for i, entry in enumerate(finished, 1):
        ev = entry["sport_event"]
        names = " vs ".join(c.get("name", "?") for c in ev.get("competitors", []))
        try:
            timeline = src.timeline(ev["id"])
        except SportradarError as err:
            print(f"[{i:>2}/{len(finished)}] {names:<36} FAILED — {err}")
            failures += 1
            continue
        events = timeline.get("timeline", [])
        shots = [x for x in events if "shot" in (x.get("type") or "")]
        total_events += len(events)
        total_shots += len(shots)
        print(f"[{i:>2}/{len(finished)}] {names:<36} {len(events):>3} events, {len(shots):>2} shots")

    cached = len(finished) - failures
    print(f"\nDone — {cached}/{len(finished)} matches cached"
          + (f" ({failures} failed)" if failures else ""))
    print(f"  API calls this run: {src.calls_made}   cache hits: {src.cache_hits}")
    print(f"  Dataset so far: {total_events} events, {total_shots} shots")
    print(f"  Cache: {src.cache_dir}")


if __name__ == "__main__":
    main()
