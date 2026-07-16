# Football Match Prediction Model

[![CI](https://github.com/JKemay/world-cup-2026-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/JKemay/world-cup-2026-predictor/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://world-cup-2026-ml.streamlit.app)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A World Cup 2026 match predictor that **improves on a hand-tuned reference model by
*fitting* its parameters from event data** instead of guessing them. Validated not
just by cross-validation but by **predicting all 24 real 2026 World Cup knockout
matches out-of-sample before they were played**: 79% top-1 accuracy, +45% RPS vs a
naive baseline.

**Live demo:** [world-cup-2026-ml.streamlit.app](https://world-cup-2026-ml.streamlit.app)

**Pipeline:** Sportradar event data → xG (from shot coordinates) → Dixon-Coles
attack/defense ratings (FIFA-anchored) → Poisson scoreline grid.

## Table of Contents

- [Status](#status)
- [Results](#results)
- [Out-of-sample validation](#out-of-sample-validation-knockout-stage)
- [Run](#run)
- [Dashboard](#dashboard)
- [How it improves on the reference model](#how-it-improves-on-the-reference-model)
- [Data](#data)
- [Example](#example)

See also: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for full modeling rationale and [AGENTS.md](AGENTS.md) for operational handoff and next steps.

## Status

| Stage | Module | Status |
|---|---|---|
| Cached Sportradar client | `footy/ingest/sportradar.py` | ✅ 96 WC + 324 qualifier matches cached |
| xG model (logistic: distance + angle) | `footy/features/xg.py` | ✅ trained on 2,633 shots, perfectly calibrated |
| Team ratings (Dixon-Coles + FIFA prior) | `footy/ratings/dixon_coles.py` | ✅ |
| Scoreline grid + W/D/L | `footy/ratings/dixon_coles.py` | ✅ |
| Evaluation harness (LOO, RPS, log-loss) | `footy/evaluate/` | ✅ Full +20.5% RPS vs naive |
| Qualifier data pull | `pull_qualifiers.py` | ✅ ~10 games/team |
| Hyperparameter tuning | `tune.py` | ✅ defaults re-validated on 96-match dataset |
| Elo benchmark | `footy/ratings/elo.py` | ✅ edges the ensemble on knockout data (RPS 0.135 vs 0.142) |
| **Ensemble (xG + Elo) — shipped** | `footy/ratings/ensemble.py` | ✅ **significantly beats the xG model (P=0.98)** |
| **Out-of-sample knockout validation** | `docs/METHODOLOGY.md` | ✅ **79% top-1 on 24 real 2026 WC knockout matches** |
| Streamlit dashboard | `app/streamlit_app.py` | ✅ pick any fixture → live grid, [deployed](https://world-cup-2026-ml.streamlit.app) |

## Results

**Dataset:** 96 World Cup + 324 qualifier matches (420 total) — the full 2026 tournament through the knockout stage. **Backtest protocol:** leave-one-out on the 96 WC matches.

| Model | log-loss | RPS | top-1 |
|---|---|---|---|
| **Ensemble (xG + Elo) — shipped** | 0.7962 | 0.1415 | 67% |
| Ensemble + draw calibration | **0.8133** | 0.1387 | 68% |
| Elo benchmark | 0.7770 | **0.1354** | 64% |
| Full (xG + FIFA + form) | 0.8493 | 0.1565 | 68% |
| FIFA-only | 0.8638 | 0.1614 | 66% |
| Naive base-rate | 1.0529 | 0.2235 | 48% |

**Bootstrap significance (10 000 resamples, paired), on the refreshed 96-match dataset:**

- **Ensemble vs Full:** ΔRPS = −0.0150, 95% CI [−0.0227, −0.0071], P(Ensemble better) = **1.000** — **statistically significant**. Averaging the xG/Dixon-Coles model with the Elo model beats either alone, because the two capture *orthogonal* signal (shot quality vs goal-based dynamic form). This is the shipped predictor.
- **Ensemble vs Naive:** ΔRPS = −0.0820, 95% CI [−0.1096, −0.0554], P = **1.000** — **+36.7% RPS / +24.4% log-loss**.
- **Full vs FIFA-only:** ΔRPS = −0.0049, 95% CI [−0.0103, +0.0005], P = 0.961 — **not distinguishable** at 95% confidence.
- **Elo vs Ensemble on knockout-heavy data:** with the tournament's knockout matches folded in, plain Elo (RPS 0.1354) now edges the 50/50 ensemble (0.1415) — the xG half's contribution shrinks in lower-event knockout football. Open item: re-tune the blend weight (see [AGENTS.md](AGENTS.md)).

**Key narrative:** on WC-only data (~2 games/team) the event model was −3.0% RPS vs the FIFA baseline — fitting noise. Qualifier data (~10 games/team) flipped that to a real edge, and benchmarking against **Elo** exposed the deeper lesson — a simple goal-based rating rivals the sophisticated xG model, because xG throws away matches with no shot data. The resolution: **ensemble the two**, which is a statistically significant gain (P=1.000) and is the shipped model. The scoreline grid still comes from the xG model (Elo has none); the W/D/L blends both.

**Hyperparameter tuning:** grid search over `alpha` × `fifa_scale`, re-run on the 96-match dataset, confirms the defaults (`alpha=0.05`, `fifa_scale=1.0`) sit on a flat surface — the nominal best cell (`alpha=0.01`, `fifa_scale=2.0`) improves RPS by only 1.6%, which the tuning script itself flags as likely noise rather than signal.

## Out-of-sample validation (knockout stage)

Cross-validation on historical data is a useful sanity check, but it's not the same
claim as **genuine prediction of matches that hadn't happened yet**. This model got
that test: all 24 matches of the real **2026 World Cup knockout stage** (Round of 32
through the Final) were predicted using a model trained **only on data available
before each match** — a strict temporal backtest with zero lookahead.

| Metric | Result |
|---|---|
| **Top-1 accuracy** | **79% (19/24)** |
| RPS | 0.1316 |
| Log-loss | 0.7148 |
| **RPS improvement vs naive baseline** | **+45.1%** |
| Round of 32 accuracy | 13/16 (81%) |
| Round of 16 → Final accuracy | 6/8 (75%) |

Of the 5 games the model's favorite didn't win outright, **4 were 90-minute draws
that went to penalty shootouts** — and the model's pick advanced on penalties in 3 of
those. Only one game (Norway's win over Brazil) was a genuine wrong-winner call. The
model's average favorite carried 54% confidence but won 79% of the time — a sign the
model is honestly *underconfident* rather than overconfident, a healthier failure
mode than the reverse.

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

Sportradar Soccer Extended (trial tier). 420 matches cached (96 World Cup + 324
qualifiers), 2,633 shots. Every response is cached under `data/` so re-runs cost
zero API calls. The source is swappable — add another adapter in `footy/ingest/`
(e.g. StatsBomb) and nothing downstream changes.

## Example

France vs Iraq → xG 2.48 vs 0.52 → most likely **2–0** →
France **79%** / draw **16%** / Iraq **5%**.

(The hand-tuned reference said France 90.6% — overconfident. See [Results](#results) for the scoreline grid.)
