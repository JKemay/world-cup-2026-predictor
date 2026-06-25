"""Sportradar Soccer Extended API client with transparent on-disk caching.

Every successful response is cached as JSON under ``data/raw/sportradar/`` keyed
by request path, so re-runs cost zero API calls — important because trial keys
have a limited quota. The key is read from ``.env`` and never logged.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from footy.config import BASE_URL, PROJECT_ROOT, RAW_DIR, REQUEST_PAUSE


class SportradarError(RuntimeError):
    """Raised when the API returns a non-success response."""


class _AuthError(Exception):
    """Internal signal for 401/403 so we can retry with query-string auth."""


def load_api_key(env_path: Path | None = None) -> str:
    """Read SPORTRADAR_API_KEY from .env without printing it."""
    env_path = env_path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        raise SportradarError(f"No .env file at {env_path}")
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("SPORTRADAR_API_KEY") and "=" in line:
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                return value
    raise SportradarError("SPORTRADAR_API_KEY is missing or empty in .env")


def _slug(path: str) -> str:
    """Turn an API path into a safe, human-readable cache filename."""
    s = path.strip("/")
    if s.endswith(".json"):
        s = s[:-5]
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _read_snippet(err) -> str:
    try:
        return err.read()[:200].decode("utf-8", "replace")
    except Exception:
        return "<no body>"


class SportradarSource:
    """Cached client for the Soccer Extended API."""

    def __init__(
        self,
        key: str | None = None,
        cache_dir: Path = RAW_DIR,
        pause: float = REQUEST_PAUSE,
    ):
        self.key = key or load_api_key()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self._last_call = 0.0
        self.calls_made = 0
        self.cache_hits = 0

    def get(self, path: str, force: bool = False) -> dict:
        """Cached GET. Reads from disk when available unless ``force=True``."""
        cache_file = self.cache_dir / f"{_slug(path)}.json"
        if cache_file.exists() and not force:
            self.cache_hits += 1
            return json.loads(cache_file.read_text())
        data = self._fetch(path)
        cache_file.write_text(json.dumps(data))
        return data

    def _fetch(self, path: str) -> dict:
        wait = self.pause - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        header_url = BASE_URL + path
        header = {"accept": "application/json", "x-api-key": self.key}
        sep = "&" if "?" in path else "?"
        qs_url = f"{BASE_URL}{path}{sep}api_key={self.key}"
        try:
            try:
                data = self._http_get(header_url, header)
            except _AuthError:
                data = self._http_get(qs_url, {"accept": "application/json"})
        except _AuthError:
            raise SportradarError(
                "Authentication failed (401/403) with both header and query-string "
                "key — check the key value and that the trial dates are active."
            )
        finally:
            self._last_call = time.monotonic()
        self.calls_made += 1
        return data

    @staticmethod
    def _http_get(url: str, headers: dict) -> dict:
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise _AuthError()
            # split('?') so a query-string key never appears in error output
            raise SportradarError(
                f"HTTP {e.code} for {url.split('?')[0]}: {_read_snippet(e)}"
            )
        except urllib.error.URLError as e:
            raise SportradarError(f"Network error for {url.split('?')[0]}: {e.reason}")

    # -- high-level helpers --
    def seasons(self, competition_id: str) -> list[dict]:
        return self.get(f"/competitions/{competition_id}/seasons.json").get("seasons", [])

    def schedule(self, season_id: str) -> list[dict]:
        data = self.get(f"/seasons/{season_id}/schedules.json")
        return data.get("schedules") or data.get("sport_events") or []

    def timeline(self, event_id: str) -> dict:
        return self.get(f"/sport_events/{event_id}/timeline.json")
