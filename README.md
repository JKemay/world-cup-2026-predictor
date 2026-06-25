# Football Match Prediction Model

A World Cup 2026 match predictor that **improves on a hand-tuned reference model by
*fitting* its parameters from event data** instead of guessing them.

**Pipeline:** Sportradar event data → xG (from shot coordinates) → Dixon-Coles
attack/defense ratings (FIFA-anchored) → Poisson scoreline grid.

## Status

| Stage | Module | Status |
|---|---|---|
| Cached Sportradar client | `footy/ingest/sportradar.py` | ✅ 52 WC matches cached |
| xG model (logistic: distance + angle) | `footy/features/xg.py` | ✅ 5-fold CV AUC 0.728 |
| Team ratings (Dixon-Coles + FIFA prior) | `footy/ratings/dixon_coles.py` | ✅ |
| Scoreline grid + W/D/L | `footy/ratings/dixon_coles.py` | ✅ |
| Evaluation harness (LOO, RPS, log-loss) | `footy/evaluate/` | ✅ Full +15.8% RPS vs naive |
| Qualifier data pull | `pull_qualifiers.py` | ✅ ~10 games/team |
| Streamlit dashboard | `app/streamlit_app.py` | ✅ pick any fixture → live grid |

## Run

```bash
pip install -r requirements.txt
python3 spike_sportradar.py    # 1. confirm data access (needs SPORTRADAR_API_KEY in .env)
python3 pull_worldcup.py       # 2. pull + cache finished WC matches (idempotent)
python3 pull_qualifiers.py     # 3. discover + cache WC qualifier timelines (~10 games/team)
python3 build_xg.py            # 4. train xG on all shots (WC + qualifiers)
python3 build_ratings.py       # 5. fit ratings + scoreline grid -> france_iraq_grid.png
python3 build_eval.py          # 6. LOO backtest: trains on all data, evaluates on WC
streamlit run app/streamlit_app.py   # 7. interactive dashboard (any fixture -> live grid)
```

The dashboard reads a committed snapshot (`app/match_table.csv`), so it deploys to
Streamlit Community Cloud with no API key or raw data — point it at `app/streamlit_app.py`.

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

France vs Iraq → France 2.15 xG, Iraq 0.61 → most likely **2–0** →
France **72%** / draw **20%** / Iraq **8%**.
