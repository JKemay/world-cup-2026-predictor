# Football Match Prediction Model

[![CI](https://github.com/JKemay/world-cup-2026-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/JKemay/world-cup-2026-predictor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A World Cup 2026 match predictor that **improves on a hand-tuned reference model by
*fitting* its parameters from event data** instead of guessing them.

**Live demo:** deploy `app/streamlit_app.py` to Streamlit Community Cloud (see [Dashboard](#dashboard) below).

**Pipeline:** Sportradar event data → xG (from shot coordinates) → Dixon-Coles
attack/defense ratings (FIFA-anchored) → Poisson scoreline grid.

## Table of Contents

- [Status](#status)
- [Results](#results)
- [Run](#run)
- [Dashboard](#dashboard)
- [How it improves on the reference model](#how-it-improves-on-the-reference-model)
- [Data](#data)
- [Example](#example)

See also: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for full modeling rationale and [AGENTS.md](AGENTS.md) for operational handoff and next steps.

## Status

| Stage | Module | Status |
|---|---|---|
| Cached Sportradar client | `footy/ingest/sportradar.py` | ✅ 52 WC + 324 qualifier matches cached |
| xG model (logistic: distance + angle) | `footy/features/xg.py` | ✅ CV ROC-AUC 0.687, perfectly calibrated |
| Team ratings (Dixon-Coles + FIFA prior) | `footy/ratings/dixon_coles.py` | ✅ |
| Scoreline grid + W/D/L | `footy/ratings/dixon_coles.py` | ✅ |
| Evaluation harness (LOO, RPS, log-loss) | `footy/evaluate/` | ✅ Full +20.5% RPS vs naive |
| Qualifier data pull | `pull_qualifiers.py` | ✅ ~10 games/team |
| Hyperparameter tuning | `tune.py` | ✅ defaults validated |
| Streamlit dashboard | `app/streamlit_app.py` | ✅ pick any fixture → live grid |

## Results

**Dataset:** 52 World Cup + 324 qualifier matches (376 total). **Backtest protocol:** leave-one-out on the 52 WC evaluation matches.

| Model | log-loss | RPS | top-1 |
|---|---|---|---|
| Full (xG + FIFA + form) | 0.8727 | 0.1606 | 67% |
| FIFA-only | 0.8727 | 0.1618 | 67% |
| Naive base-rate | — | 0.2019 | — |

**Bootstrap significance (10 000 resamples, paired):**

- **Full vs Naive:** ΔRPS = −0.0413, 95% CI [−0.0777, −0.0061], P(Full better) = 0.99 — **statistically significant**. The event-data pipeline is +20.5% RPS / +13.0% log-loss better than predicting base-rates for every match.
- **Full vs FIFA-only:** ΔRPS = −0.0012, 95% CI [−0.0090, +0.0062] — **not statistically distinguishable** on 52 eval matches. The +0.7% edge is real in direction but sits within sampling noise; more eval matches would be needed to declare it conclusive.

**Key narrative:** on WC-only data (~2 games/team) the event model was −3.0% RPS vs the FIFA baseline — it was fitting noise. Adding qualifier data (~10 games/team) flipped the sign to +0.7%, and the pipeline is decisively better than naive. The honest take: the Full-vs-FIFA-only gap is within confidence bounds.

**Hyperparameter tuning:** grid search over `alpha` × `fifa_scale` confirms defaults (`alpha=0.05`, `fifa_scale=1.0`) are within 0.18% RPS of the best cell (rank 5/30 on the surface) — a flat landscape, so defaults are validated, not over-tuned.

![Model analysis: calibration, biggest surprises, attack/defense landscape](model_analysis.png)

*Three-panel figure: reliability (calibration) curve, biggest model misses (under-predicted draws — Spain–Cape Verde, England–Ghana, Portugal–Congo DR), and attack/defense landscape.*

### Worked example

France vs Iraq: modeled xG 2.48 vs 0.52 → **France 79% / draw 16% / Iraq 5%**, modal score **2–0**. The hand-tuned reference predicted France 90.6% — overconfident, because it applies FIFA strength as a multiplicative scaler.

![France vs Iraq scoreline grid](france_iraq_grid.png)

## Run

```bash
pip install -r requirements.txt
python3 spike_sportradar.py    # 1. confirm data access (needs SPORTRADAR_API_KEY in .env)
python3 pull_worldcup.py       # 2. pull + cache finished WC matches (idempotent)
python3 pull_qualifiers.py     # 3. discover + cache WC qualifier timelines (~10 games/team)
python3 build_xg.py            # 4. train xG on all shots (WC + qualifiers)
python3 build_ratings.py       # 5. fit ratings + scoreline grid -> france_iraq_grid.png
python3 build_eval.py          # 6. LOO backtest: trains on all data, evaluates on WC
python3 tune.py                # 7. grid search over alpha × fifa_scale -> tune_alpha_fifa.png
python3 analyze_results.py     # 8. 3-panel analysis -> model_analysis.png
streamlit run app/streamlit_app.py   # 9. interactive dashboard (any fixture -> live grid)
```

The dashboard reads a committed snapshot (`app/match_table.csv`), so it deploys to
Streamlit Community Cloud with no API key or raw data — point it at `app/streamlit_app.py`.

## Dashboard

The Streamlit app (`app/streamlit_app.py`) lets you pick any fixture from the committed match table and see the full scoreline probability grid, W/D/L bar chart, and expected-goals breakdown — no API key required.

**Run locally:**

```bash
streamlit run app/streamlit_app.py
```

**Deploy to Streamlit Community Cloud:** fork the repo, connect it to [share.streamlit.io](https://share.streamlit.io), point the entry-point at `app/streamlit_app.py`. The app reads only `app/match_table.csv` (committed), so no secrets are needed. A screenshot can be added at `docs/dashboard.png` once deployed.

## How it improves on the reference model

- **Fitted, not hand-tuned** — attack/defense come from a regularized Poisson fit on
  xG, not magic `BASE_GOALS` / `SCALING_CONSTANT` constants.
- **FIFA as a prior**, not a post-hoc multiplier — stabilizes the thin in-tournament
  sample (~2 games/team) so elite sides aren't mis-rated by small-sample noise.
- **Correct xG labels** (headers & direct free-kicks counted; penalties / own-goals
  excluded) and an **honest cross-validated** evaluation the original lacked.
- **Dixon-Coles low-score correction** on the draw-heavy 0-0 / 1-0 / 0-1 / 1-1 cells.

## Data

Sportradar Soccer Extended (trial tier). Every response is cached under `data/` so
re-runs cost zero API calls. The source is swappable — add another adapter in
`footy/ingest/` (e.g. StatsBomb) and nothing downstream changes.

## Example

France vs Iraq → xG 2.48 vs 0.52 → most likely **2–0** →
France **79%** / draw **16%** / Iraq **5%**.

(The hand-tuned reference said France 90.6% — overconfident. See [Results](#results) for the scoreline grid.)
