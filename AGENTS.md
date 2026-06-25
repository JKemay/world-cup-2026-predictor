# Agent / contributor context

Orientation for an AI agent (Claude Code, Codex, etc.) or a human picking this
project up cold. Read `README.md` for the pitch and `docs/METHODOLOGY.md` for the
modeling rationale; this file is the **operational** handoff: state, setup, and
what's next.

## What this is

A World Cup 2026 match predictor that **fits** its parameters from event data
instead of hand-tuning them: Sportradar events → xG (logistic on shot geometry)
→ Dixon-Coles attack/defense ratings (FIFA-ranking prior) → Poisson scoreline
grid → leave-one-out backtest. There's an interactive Streamlit dashboard.

## Current state (✅ done)

- Cached Sportradar client; 52 World Cup + 324 qualifier matches pulled.
- xG model (logistic, distance + angle), perfectly calibrated.
- Dixon-Coles ratings with FIFA prior — ratings pass the smell test
  (Spain #1, France #4; weakest: Gibraltar, Curaçao).
- Honest LOO backtest (RPS, log-loss). **Headline finding:** the event-data
  model went from −3.0% RPS vs a FIFA-only baseline on WC-only data
  (~2 games/team) to **+0.7%** once qualifier data (~10 games/team) was added,
  and **+20.5% vs naive**. 67% top-1, 0.161 RPS.
- Streamlit dashboard (`app/streamlit_app.py`), deploy-ready from a committed
  snapshot (`app/match_table.csv`).
- **Elo benchmark** (`footy/ratings/elo.py`, `build_elo.py`) — World-Football-style
  Elo, leakage-free on the WC eval set. **Key finding:** Elo (RPS 0.1459, log-loss
  0.8505) *edges* the Full xG model (0.1606 / 0.8727), because Elo learns from goals
  in all 376 matches while the xG model discards the ~133 with no shot data. Not
  significant (ΔRPS 95% CI [−0.0356, +0.0077], P(Elo better)=0.91); Full still wins
  top-1 (67% vs 60%).
- 101-test pytest suite, GitHub Actions CI (ruff + pytest), MIT license,
  ruff-clean (line-length 120).

## Open items / next steps

- **Pull AFC (Asian) qualifiers** — they aren't under the same Sportradar
  "FIFA World Cup Qualification" competition tree, so Japan / South Korea / Iran
  currently lean on the FIFA prior. Extend `pull_qualifiers.py` discovery.
- **Streamlit Community Cloud deploy** — point it at `app/streamlit_app.py`
  (needs a GitHub OAuth login; the public app URL is the resume link).
- **Market-odds benchmark (in progress)** — OddsPapi has a free tier with historical
  pre-match 1X2 odds covering internationals (needs a free self-serve `ODDSPAPI_API_KEY`
  in `.env`). Plan: match our fixtures to OddsPapi by team+date, convert to no-vig
  probabilities, compare model vs market vs Elo vs FIFA vs naive + a betting-ROI sim.
- **Use Elo as the prior / ensemble** — Elo beating the xG model suggests the dynamic,
  goal-informed Elo rating is a *better prior than static FIFA rank*. Swapping/augmenting
  the `fifa` prior in `DixonColesRatings` with Elo (or ensembling the two) is the most
  promising accuracy lever, and complements the `goals_fallback`/`sos_weighting` flags
  (both validated, default off) that move in the same direction.
- The +0.7% edge over FIFA-only is within sampling noise on 52 eval matches —
  more/better data is the path to a decisive result.
- Biggest model misses are **under-predicted draws** (favorites dropping points,
  e.g. Spain–Cape Verde, England–Ghana, Portugal–Congo DR) — a candidate modeling improvement.
- **Bootstrap significance + hyperparameter tuning are done.** `build_eval.py` now
  prints 95% CIs (Full vs Naive: significant; Full vs FIFA-only: not distinguishable
  on 52 eval matches). `tune.py` maps the `alpha` × `fifa_scale` surface
  (`tune_alpha_fifa.png`) — defaults are within 0.18% RPS of the global best, surface
  is flat.

### Explored and rejected

- **Shot-type xG features (`is_header`, `is_freekick`):** the Sportradar trial feed
  only populates the `method` field on goal events, never on non-goal shots — so
  `is_header = 1` was a near-perfect label for `is_goal = 1` (leak). CV AUC inflated
  to 0.719 and produced a 0.957 xG for an 11 m header. Geometry-only xG (distance +
  angle) is the correct and shipped model. Do not reintroduce shot-type flags without
  verifying that a non-trial Sportradar tier populates `method` on non-goal events.
- **Goals fallback for no-xG matches (`goals_fallback`, default off):** using actual
  goals for matches whose feed lacks shot data lifts thin CAF/Curaçao teams from 2 to
  12 matches of signal. *Naive* fallback improved aggregate RPS only within noise
  (0.1606 → 0.1586) but *lowered* top-1 accuracy (67% → 63%) by overrating minnows that
  ran up goals vs weak opposition (Netherlands–Tunisia 65/24/11 → 53/30/17).
- **Strength-of-schedule weighting (`sos_weighting`, default off):** the principled fix
  for the above — down-weights a goals-fallback row by the opponent's FIFA strength
  (`w = clip(0.5 + 0.30·z_opp, 0.1, 1.0)`), so goals vs weak teams count less. Validated:
  with `goals_fallback=True, sos_weighting=True` the backtest is ≥ baseline on **all
  three** metrics (RPS 0.1588, log-loss 0.8653, top-1 **67.3%** — recovers the accuracy
  naive fallback lost) and pulls Netherlands–Tunisia back to 58/28/14. Still within noise
  (ΔRPS 95% CI straddles 0) and still credits thin teams a touch more than the prior-only
  baseline, so it ships **off** pending a deliberate call. Constants are fixed a priori
  (not tuned on the eval set). To enable: pass `goals_fallback=True, sos_weighting=True`
  to `DixonColesRatings` / `leave_one_out`.

## Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt   # runtime + dev (pytest, ruff)
cp .env.example .env                                       # then paste a Sportradar
                                                           # Soccer Extended trial key
```

The **API key lives only in `.env`** (git-ignored) — never commit or print it.

## What runs WITHOUT the key or a data re-pull

`data/` (raw cache + processed CSVs) is git-ignored, so a fresh clone lacks it.
These still work because their inputs are committed:

```bash
streamlit run app/streamlit_app.py   # dashboard — uses app/match_table.csv
python3 -m pytest -q                  # 73/74 pass; 1 xG-calibration test skips w/o data
ruff check .                          # lint
```

## Regenerating data/ from scratch (needs the key)

Run in order — each step is idempotent and caches every API response:

```bash
python3 pull_worldcup.py     # 52 WC match timelines        -> data/raw/
python3 pull_qualifiers.py   # ~324 qualifier timelines (~10 min, rate-limited)
python3 build_xg.py          # train xG -> data/processed/shots_xg.csv, xg_pitch.png
python3 build_ratings.py     # fit ratings -> team_ratings.csv, france_iraq_grid.png
python3 build_eval.py        # LOO backtest -> backtest.csv, calibration.png
python3 analyze_results.py   # 3-panel analysis -> model_analysis.png
```

## Layout

- `footy/` — the package: `ingest/` (cached Sportradar client), `features/`
  (shots → xG, match table), `ratings/` (Dixon-Coles, FIFA prior), `evaluate/`
  (backtest). Trained model in `footy/models/`.
- `build_*.py`, `pull_*.py`, `spike_sportradar.py`, `analyze_results.py` —
  entry-point scripts (they mutate `sys.path` to import `footy`).
- `app/` — Streamlit dashboard + committed `match_table.csv` snapshot.
- `tests/` — pytest suite. `docs/METHODOLOGY.md` — the writeup.

## Conventions

- Don't commit `.env` or anything under `data/`. Don't print the API key.
- Keep `ruff check .` clean and `pytest` green before pushing (CI enforces both).
