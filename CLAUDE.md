# Project context for Claude Code

See **[AGENTS.md](AGENTS.md)** for the full operational handoff — current state,
setup, how to run things without the API key, how to regenerate `data/`, and the
open next steps. It's the single source of truth for working on this repo.

Quick reminders:
- The Sportradar API key lives only in `.env` (git-ignored) — never commit or print it.
- `data/` is git-ignored; a fresh clone regenerates it via the `pull_*` / `build_*`
  scripts (needs the key). The dashboard and tests run without it.
- Keep `ruff check .` clean and `python3 -m pytest -q` green before pushing — CI enforces both.
