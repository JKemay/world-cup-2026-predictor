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

- Cached Sportradar client; 96 World Cup 2026 matches (group stage through the
  Round of 16, pulled as each round completed) + 452 qualifier matches
  (includes AFC — see "DONE — AFC qualifiers pulled" below).
- xG model (logistic, distance + angle) trained on shot data from all 548
  matches, perfectly calibrated.
- Dixon-Coles ratings with FIFA prior — ratings pass the smell test
  (Spain #1, Argentina #2; weakest: Gibraltar, Curaçao).
- Honest LOO backtest (RPS, log-loss), re-run on the 96-match WC eval set
  (trained on all 548). **Headline finding:** the event-data model went from
  −3.0% RPS vs a FIFA-only baseline on WC-only data (~2 games/team) to a real
  edge once qualifier data (~10 games/team) was added.
- Streamlit dashboard (`app/streamlit_app.py`), **deployed**: https://world-cup-2026-ml.streamlit.app
- **Elo benchmark** (`footy/ratings/elo.py`, `build_elo.py`) — World-Football-style
  Elo, leakage-free on the WC eval set. **Key finding:** Elo (RPS 0.1408) *edges* the
  Full xG model (0.1583) and the 50/50 ensemble (0.1460) — see the blend-weight
  finding below (still not significant enough to change the shipped default).
- **Ensemble = SHIPPED predictor** (`footy/ratings/ensemble.py`, `EnsemblePredictor`)
  — 50/50 blend of the xG/Dixon-Coles W/D/L and the Elo W/D/L. **RPS 0.1460, log-loss
  0.8134**; significantly beats the Full model (ΔRPS −0.0123, 95% CI [−0.0194, −0.0053],
  P=1.000) and naive (P=1.000, +34.7% RPS). `build_eval.py`, `build_ratings.py`, and
  the dashboard all use it. The **scoreline grid stays Dixon-Coles** (Elo has no grid);
  only the W/D/L blends. To get pure xG behaviour, use `DixonColesRatings` directly
  instead of `EnsemblePredictor`.
- **Blend-weight re-tune — explored, not shipped** (`footy/evaluate/backtest.py`:
  `fit_blend_weight`, `nested_blend_predictions`). The 50/50 weight was chosen on the
  original 52-match dataset; on the fuller dataset (now 548 matches incl. AFC), Elo
  alone still has a better point estimate than the ensemble, and an in-sample weight
  search lands at `w*=0.0` (pure Elo). A leakage-free nested-LOO re-estimate confirms
  the point-estimate gain (RPS 0.1408 vs 0.1460) but does **not** clear 95%
  significance: P(tuned better)=0.931, 95% CI [−0.0119, +0.0017] (straddles zero,
  narrower gap than pre-AFC's P=0.948). **Decision: kept the 0.5 default.** This
  is a "suggestive, not proven" result, not a rejection — revisit once more tournament
  data accumulates. `build_eval.py`'s previous 3-way (DC/Elo/OL) simplex search was
  **leaky** (tuned and scored on the same LOO set) and has been replaced by this
  honest protocol.
- **Out-of-sample validation — the real 2026 WC knockout stage**
  (`backtest_temporal.py`, `footy.evaluate.backtest.temporal_backtest`). Strict
  chronological backtest: each of the 24 R32+R16 knockout matches predicted using a
  model trained only on data available before that match — no LOO shortcuts. **79%
  top-1 (19/24), RPS 0.1297, +45.7% vs naive.** 4 of the 5 misses were 90-minute draws
  that went to penalty shootouts (the model's favorite won 2 of those 4 shootouts,
  in line with a well-calibrated near-coin-flip shootout model) — see the shootout
  layer below. See `docs/METHODOLOGY.md` §7a for full writeup.
- **Penalty-shootout / advancement layer** (`footy/ratings/shootout.py`,
  `advancement_prob`) — thin, opt-in, decoupled from `EnsemblePredictor`'s W/D/L
  contract. Resolves the 90'-drawn branch via a fixed-a-priori logistic on the Elo
  gap (`SHOOTOUT_ELO_SCALE=2000`, deliberately flat/near-coin-flip; **not** fitted
  against observed shootouts — n=4 in 2026 is noise). Plausibility-checked (not
  statistically validated) against the 4 real 2026 WC shootouts via
  `backtest_temporal.py --shootout`. Surfaced in the Streamlit app as a "Knockout tie"
  toggle.
- 176-test pytest suite, GitHub Actions CI (ruff + pytest), MIT license,
  ruff-clean (line-length 120).

## Scope decision (2026-07-14) — international-only; club football is a separate project

This model **stays international-only** (World Cup + qualifiers). Club football
(Premier League first, then maybe La Liga) will be a **separate project** that
imports a shared core, not an extension of this one. Rationale:

- **Ratings aren't transferable across populations** — clubs and national teams
  share no common opponents, so they cannot be co-calibrated on one scale. Even
  "one model" would need separately-fitted instances, so separation is forced.
- **The FIFA-rank prior has no club equivalent** — this model's identity is
  thin-sample stabilization (2 games/team at a WC). Club football is the
  opposite regime (38+ games/season, rich data); it anchors on Elo / market
  value instead, and leans less on priors.
- **Different confounders:** club football has strong, stable **home advantage**
  (vs neutral-venue WC knockouts), transfer windows, rotation, fixture
  congestion, and per-league scoring baselines — all absent here.

**Reuse plan:** promote the league-agnostic engine (xG-from-geometry,
Dixon-Coles, Elo, ensemble, evaluation harness) to a shared importable core;
the club project swaps ingest, prior, venue handling, and adds club-specific
features. Portfolio-wise, two clean projects beat one sprawling one. Start the
club model on the **Premier League** (richest public xG data), prove it
end-to-end, then generalize — the same incremental path this project took.

## Open items / next steps

- ✅ **DONE — AFC qualifiers pulled.** Root cause: Sportradar names the AFC
  competition `AFC Asian Qualifiers 2026` (`sr:competition:308`) without the
  literal "World Cup" substring every other confederation uses ("FIFA World Cup
  Qualification, UEFA/CAF/CONCACAF/CONMEBOL/OFC"), so `pull_qualifiers.py`'s
  name filter silently skipped it. Fixed with a targeted allowlist (not a
  loosened match — verified it doesn't also sweep in "AFC Asian Cup,
  Qualification", a different tournament). Pulled 452 total qualifier matches
  (was 324); Japan/Korea Republic/IR Iran/Saudi Arabia/Iraq/etc. now have
  19–24 matches each of real shot data instead of leaning entirely on the FIFA
  prior. Re-running the full pipeline afterward shows a small, consistent
  improvement: out-of-sample knockout RPS 0.1316→0.1297, log-loss
  0.7148→0.7040, still 79% top-1 (19/24), +45.7% vs naive (was +45.1%).
- **Market-odds benchmark (in progress)** — OddsPapi has a free tier with historical
  pre-match 1X2 odds covering internationals (needs a free self-serve `ODDSPAPI_API_KEY`
  in `.env`). Plan: match our fixtures to OddsPapi by team+date, convert to no-vig
  probabilities, compare model vs market vs Elo vs FIFA vs naive + a betting-ROI sim.
- ✅ **DONE — Elo ensemble shipped, blend weight re-validated** (see Current state).
  `goals_fallback`/`sos_weighting` remain validated-but-off alternatives.
- ✅ **DONE — Out-of-sample knockout validation + penalty-shootout layer** (see Current
  state). `backtest_temporal.py`, `footy/ratings/shootout.py`.
- **Confidence sharpening — deferred, not implemented.** The knockout-stage favorite
  averages 54% confidence but wins 79% of the time (underconfident). A defensible fix
  is a single temperature/Platt-scaling parameter fit via the same nested-LOO protocol
  as the blend weight. Deferred because 24 knockout games is a thin sample to fit a
  sharpening parameter against without risking refitting noise — only pick this up if
  the reliability curve on the *full* 96-match LOO set (not just the 24 knockout games)
  shows the same underconfidence.
- Pull the remaining knockout rounds (Quarterfinals, Semifinals, Final) into the cache
  as they're played, and re-run `build_eval.py` + `backtest_temporal.py` — the temporal
  script auto-detects however many `WC_SEASON_ID` matches fall after `KNOCKOUT_START`,
  so no code change needed, just a fresh `pull_worldcup.py` run.
- A richer xG model via StatsBomb open data, or a hierarchical Bayesian rebuild for
  principled thin-data shrinkage, remain open longer-term ideas.
- Biggest model misses are **under-predicted draws** (favorites dropping points,
  e.g. Spain–Cape Verde, England–Ghana, Portugal–Congo DR) — a candidate modeling improvement.
- **Bootstrap significance + hyperparameter tuning are done**, re-run on the 96-match
  dataset. `build_eval.py` prints 95% CIs (Ensemble vs Naive: significant; Full vs
  FIFA-only: not distinguishable). `tune.py` maps the `alpha` × `fifa_scale` surface
  (`tune_alpha_fifa.png`) — defaults remain close to the best cell on a flat surface.

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
python3 -m pytest -q                  # 176/177 pass; 1 xG-calibration test skips w/o data
ruff check .                          # lint
```

## Regenerating data/ from scratch (needs the key)

Run in order — each step is idempotent and caches every API response:

```bash
python3 pull_worldcup.py     # WC match timelines (grows each round)  -> data/raw/
python3 pull_qualifiers.py   # ~452 qualifier timelines (incl. AFC) (~10 min, rate-limited)
python3 build_xg.py          # train xG -> data/processed/shots_xg.csv, xg_pitch.png
python3 build_ratings.py     # fit ratings -> team_ratings.csv, france_iraq_grid.png
python3 build_eval.py        # LOO backtest -> backtest.csv, calibration.png
python3 backtest_temporal.py # temporal out-of-sample knockout backtest (see Current state)
python3 analyze_results.py   # 3-panel analysis -> model_analysis.png
```

## Layout

- `footy/` — the package: `ingest/` (cached Sportradar client), `features/`
  (shots → xG, match table), `ratings/` (Dixon-Coles, Elo, FIFA prior,
  `shootout.py` — penalty-shootout advancement layer), `evaluate/` (backtest,
  including `temporal_backtest` for strict chronological out-of-sample scoring).
  Trained model in `footy/models/`.
- `build_*.py`, `pull_*.py`, `spike_sportradar.py`, `analyze_results.py`,
  `backtest_temporal.py` — entry-point scripts (they mutate `sys.path` to
  import `footy`).
- `app/` — Streamlit dashboard + committed `match_table.csv` snapshot.
- `tests/` — pytest suite. `docs/METHODOLOGY.md` — the writeup.

## Conventions

- Don't commit `.env` or anything under `data/`. Don't print the API key.
- Keep `ruff check .` clean and `pytest` green before pushing (CI enforces both).
