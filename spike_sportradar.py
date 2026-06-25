#!/usr/bin/env python3
"""
Sportradar Soccer Extended — Phase 0 data spike.

Answers the GO/NO-GO gate: does our trial tier return X/Y event coordinates for
World Cup matches? Everything downstream depends on it, so we check before building.

Standard library only — no pip install needed. Reads SPORTRADAR_API_KEY from .env
and NEVER prints the key value.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

BASE = "https://api.sportradar.com/soccer-extended/trial/v4/en"
FIFA_WORLD_CUP = "sr:competition:16"  # historical Sportradar id; used as a fallback
PAUSE = 1.3  # seconds between calls — trial rate limits are tight


def load_key(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        sys.exit(f"X  No {path} found. Create it with: SPORTRADAR_API_KEY=your_key")
    key = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("SPORTRADAR_API_KEY") and "=" in s:
                key = s.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("X  SPORTRADAR_API_KEY is empty in .env — paste your trial key after the '='.")
    print(f"OK  Key loaded ({len(key)} chars). [value never printed]")
    return key


def api_get(path, key, _retry_qs=True):
    """GET BASE+path. Tries x-api-key header; on 401/403 retries with ?api_key=."""
    url = BASE + path
    req = urllib.request.Request(url, headers={"accept": "application/json", "x-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and _retry_qs:
            sep = "&" if "?" in path else "?"
            url2 = f"{BASE}{path}{sep}api_key={key}"
            req2 = urllib.request.Request(url2, headers={"accept": "application/json"})
            try:
                with urllib.request.urlopen(req2, timeout=30) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e2:
                return e2.code, _safe_body(e2)
        return e.code, _safe_body(e)
    except urllib.error.URLError as e:
        return 0, {"_error": str(e)}


def _safe_body(e):
    try:
        return json.load(e)
    except Exception:
        try:
            return {"_raw": e.read()[:300].decode("utf-8", "replace")}
        except Exception:
            return {"_raw": "<unreadable>"}


def has_xy(obj):
    """True if a coordinate-like field appears anywhere in the event."""
    if isinstance(obj, dict):
        keys = {k.lower() for k in obj}
        if "coordinates" in keys or ("x" in keys and "y" in keys):
            return True
        return any(has_xy(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_xy(v) for v in obj)
    return False


def main():
    key = load_key()

    # [1/4] validate key + endpoint
    print("\n[1/4] Validating key against the competitions endpoint...")
    status, body = api_get("/competitions.json", key)
    if status != 200:
        sys.exit(f"X  Auth/endpoint check failed (HTTP {status}): {str(body)[:300]}")
    comps = body.get("competitions", [])
    print(f"OK  Authenticated. {len(comps)} competitions visible on this tier.")

    # [2/4] locate the World Cup competition + a 2026 season
    time.sleep(PAUSE)
    print("\n[2/4] Locating the World Cup 2026 season...")
    wc = next((c for c in comps if c.get("name", "").strip().lower() == "world cup"), None)
    comp_id = wc["id"] if wc else FIFA_WORLD_CUP
    print(f"    competition = {comp_id} ({wc['name'] if wc else 'fallback id'})")
    status, body = api_get(f"/competitions/{comp_id}/seasons.json", key)
    if status != 200:
        sys.exit(f"X  Could not list seasons (HTTP {status}): {str(body)[:300]}")
    seasons = body.get("seasons", [])
    season = next(
        (s for s in seasons if "2026" in f"{s.get('name', '')}{s.get('year', '')}"), None
    )
    if not season:
        recent = [s.get("name") for s in seasons][-8:]
        sys.exit(f"X  No 2026 season found. Recent seasons seen: {recent}")
    print(f"OK  season = {season.get('name')} ({season['id']})")

    # [3/4] find a finished match in the schedule
    time.sleep(PAUSE)
    print("\n[3/4] Finding a finished match in the season schedule...")
    status, body = api_get(f"/seasons/{season['id']}/schedules.json", key)
    if status != 200:
        sys.exit(f"X  Could not load schedule (HTTP {status}): {str(body)[:300]}")
    entries = body.get("schedules") or body.get("sport_events") or []
    finished = []
    for e in entries:
        ev = e.get("sport_event", e)
        st = e.get("sport_event_status", {})
        if st.get("status") in ("closed", "ended") or st.get("match_status") == "ended":
            finished.append(ev)
    print(f"    {len(entries)} scheduled events, {len(finished)} finished.")
    if not finished:
        if entries:
            print(f"    (debug) first entry keys: {list(entries[0].keys())}")
        sys.exit("X  No finished matches found yet — re-run once a WC match has completed.")
    match = finished[0]
    names = " vs ".join(c.get("name", "?") for c in match.get("competitors", []))
    print(f"OK  match = {names or match['id']} ({match['id']})")

    # [4/4] the real test — does the timeline carry X/Y coordinates?
    time.sleep(PAUSE)
    print("\n[4/4] Pulling the timeline and scanning for X/Y coordinates...")
    status, body = api_get(f"/sport_events/{match['id']}/timeline.json", key)
    if status != 200:
        sys.exit(f"X  Timeline fetch failed (HTTP {status}): {str(body)[:300]}")
    timeline = body.get("timeline", [])
    if not timeline:
        sys.exit(f"X  Timeline empty. Response keys: {list(body.keys())}")
    coord_events = [t for t in timeline if has_xy(t)]
    types = Counter(t.get("type", "?") for t in timeline)
    print(f"    {len(timeline)} timeline events; {len(coord_events)} carry coordinates.")
    print(f"    top event types: {dict(types.most_common(8))}")

    print("\n" + "=" * 52)
    if coord_events:
        print("GO  --  X/Y coordinates ARE present on this tier.")
        sample = coord_events[0]
        print("   sample event:")
        print("   " + json.dumps(sample, indent=2)[:700].replace("\n", "\n   "))
    else:
        print("NO-GO  --  no X/Y coordinates on this tier for the World Cup.")
        print("   Pivot: use StatsBomb open data (has WC shot coordinates, free).")
    print("=" * 52)


if __name__ == "__main__":
    main()
