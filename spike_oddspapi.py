#!/usr/bin/env python3
"""GO/NO-GO probe for OddsPapi historical 1X2 odds.

Reads ODDSPAPI_API_KEY from .env (never printed). Lists soccer tournaments,
finds the World Cup + qualifiers, then inspects a fixture's odds structure so
we can build the adapter. Stdlib only.

    python3 spike_oddspapi.py
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://api.oddspapi.io"
ENV = Path(__file__).resolve().parent / ".env"


def load_key() -> str:
    if not ENV.exists():
        sys.exit("No .env file")
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("ODDSPAPI_API_KEY") and "=" in line:
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    sys.exit("ODDSPAPI_API_KEY missing or empty in .env")


KEY = load_key()


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def get(path: str, **params):
    params["apiKey"] = KEY
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    safe = url.replace(KEY, "***")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r), safe
    except urllib.error.HTTPError as e:
        body = e.read()[:400].decode("utf-8", "replace")
        print(f"  HTTP {e.code} for {safe}\n    body: {body}")
        return None, safe
    except Exception as e:  # noqa: BLE001
        print(f"  error for {safe}: {e}")
        return None, safe


def structure(obj, indent=2, maxk=12):
    pad = " " * indent
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{pad}dict with {len(keys)} keys: {keys[:maxk]}")
        for k in keys[:4]:
            v = obj[k]
            kind = type(v).__name__
            sample = v if isinstance(v, (str, int, float, bool, type(None))) else f"<{kind}>"
            print(f"{pad}  {k!r}: {sample}")
    elif isinstance(obj, list):
        print(f"{pad}list of {len(obj)} items")
        if obj:
            print(f"{pad}  first item:")
            structure(obj[0], indent + 4)


def main():
    print("OddsPapi GO/NO-GO probe (key loaded, hidden)\n")

    print("1) GET /v4/tournaments?sportId=10  (soccer)")
    data, _ = get("/v4/tournaments", sportId=10)
    if data is None:
        sys.exit("NO-GO: tournaments call failed")
    tours = data if isinstance(data, list) else data.get("tournaments") or data.get("data") or []
    print(f"   got {len(tours)} tournaments")
    if tours:
        structure(tours[0])

    def name_of(t):
        return (t.get("tournamentName") or t.get("name") or "").lower()

    wc = [t for t in tours if "world cup" in name_of(t)]
    print(f"\n   World-Cup-named tournaments: {len(wc)}")
    for t in wc[:15]:
        tid = t.get("tournamentId") or t.get("id")
        print(f"     id={tid}  name={t.get('tournamentName') or t.get('name')}  "
              f"cat={t.get('categoryName')}")

    if not wc:
        sys.exit("\nNO-GO: no World Cup tournament found under soccer")

    # pick the main men's World Cup 2026 (avoid women/qualification for the first fixture probe)
    main_wc = None
    for t in wc:
        n = name_of(t)
        if "qualif" not in n and "women" not in n:
            main_wc = t
            break
    main_wc = main_wc or wc[0]
    tid = main_wc.get("tournamentId") or main_wc.get("id")
    print(f"\n2) GET /v4/odds-by-tournaments  tournamentIds={tid}")
    data, _ = get("/v4/odds-by-tournaments", tournamentIds=tid)
    if data is None:
        print("   (retrying without — inspect error above for required params)")
    else:
        fixtures = data if isinstance(data, list) else (
            data.get("fixtures") or data.get("data") or data.get("events") or [])
        print("   structure of response:")
        structure(data)
        sample = fixtures[0] if isinstance(fixtures, list) and fixtures else None
        if sample:
            print("\n   sample fixture:")
            structure(sample)
            fid = sample.get("fixtureId") or sample.get("id")
            print(f"\n3) GET /v4/historical-odds  fixtureId={fid}")
            hod, _ = get("/v4/historical-odds", fixtureId=fid)
            if hod is None:
                hod, _ = get("/v4/get-historical-odds", fixtureId=fid)
            if hod is not None:
                print("   historical-odds structure:")
                structure(hod, maxk=20)
                print("\nGO: historical odds reachable — ready to build the adapter.")
            else:
                print("\nPARTIAL: fixtures listable but historical-odds shape unknown "
                      "(see errors above for required params).")


if __name__ == "__main__":
    main()
