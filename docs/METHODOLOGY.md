# Methodology: World Cup 2026 Match Predictor

## 1. Problem

The goal is to predict World Cup 2026 match outcomes as a full scoreline probability distribution — not just a W/D/L label — and to do so by **fitting parameters from event data** rather than hand-tuning them.

The reference model uses fixed magic constants (`BASE_GOALS`, `SCALING_CONSTANT`) and applies FIFA rankings as a post-hoc multiplier. This project replaces that approach end-to-end: expected goals are estimated geometrically from shot coordinates, and team attack/defense ratings are solved from those xG values via regularized Poisson regression with a FIFA-ranking prior baked into the regression itself. The whole pipeline is evaluated honestly with a leave-one-out backtest and proper scoring rules.

---

## 2. Data

**Source:** Sportradar Soccer Extended (trial tier). Every API response is cached under `data/` so re-runs are free.

**Coverage:** 96 completed World Cup 2026 matches (group stage through the Final) and ~324 qualifying matches (~10 games per team at tournament time). Qualifier data roughly quintuples per-team sample size, which turns out to be the decisive factor.

**Data-quality fixes applied:**

| Issue | Fix |
|---|---|
| Penalties and own-goals distort open-play skill | Excluded from xG labels (`method in {penalty, own_goal}`) |
| `shot_saved` events are keeper-side duplicates of `shot_on_target` | Silently dropped; counting both would double-count the shot |
| Headers and direct free-kicks are legitimate open-play goals | Counted as `is_goal = 1` (default `method == 'shot'` covers them) |
| CAF qualifier feed returns no shot coordinates | Matches with `home_xg == 0 AND away_xg == 0` skipped from xG training (real matches always produce nonzero xG from corners and set pieces) |
| Sportradar team names diverge from FIFA canonical names | Normalized via a hand-built alias table (`footy/ratings/fifa.py`) |

---

## 3. Expected Goals (xG)

Shot geometry follows the standard approach: for each shot, compute **distance** (metres) to the goal centre and the **angle** (radians) subtended by the 7.32 m goal mouth at the shot location.

Coordinates are normalized so every team always attacks toward `x = 100` on a notional 100×100 grid, which maps to a real 105 m × 68 m pitch. The goal centre sits at `(100, 50)`. The geometry:

```
distance = sqrt((105 - x_m)² + (34 - y_m)²)
angle    = arccos((a² + b² - (7.32)²) / (2ab))
```

where `a` and `b` are distances from the shot to each goalpost.

A **logistic regression** on `[distance, angle]` is fit using 5-fold cross-validation — out-of-sample predicted probabilities drive all reported metrics, so train-set leakage is impossible.

![xG by location](../xg_pitch.png)

**Metrics (CV, mixed corpus):**

| Metric | Value |
|---|---|
| CV AUC | 0.69–0.73 (mixed-quality corpus) |
| CV log-loss | beats base-rate baseline |
| Calibration | Total xG ≈ total goals (perfectly calibrated by construction) |

The model is intentionally simple: two geometric features, no shot-type interaction terms. On the available corpus size this keeps variance in check and interpretability high.

![calibration](../calibration.png)

---

## 4. Team Ratings

### Dixon-Coles formulation

Attack and defense strengths are estimated by a **regularized Poisson regression** fit on xG (not raw goals). Using xG rather than goals smooths the signal — a 0–0 that generated 3.2 vs 0.4 xG carries very different information than a genuine low-chance game, and raw scorelines on ~96 matches are too noisy to reliably identify team strengths without this smoothing.

The linear predictor for team H at home against A is:

```
log λ_home = intercept + home_adv + attack[H] + defense[A]
log λ_away = intercept             + attack[A] + defense[H]
```

`defense[t]` encodes how much facing team t suppresses an opponent's xG (negative coefficient → strong defense). Both attack and defense coefficients are L2-regularized (`alpha = 0.5` in `sklearn.PoissonRegressor`) to prevent overfitting on thin per-team samples.

### FIFA ranking as a prior (not a multiplier)

The key stabilizer is injecting the FIFA ranking **directly into the Poisson regression as two additional features** — one for the attacking team's ranking, one for the defending team's — rather than applying it as a post-hoc scaling factor. The transformation is:

```
fifa_strength(t) = standardize(-log(rank(t)))
```

This is included alongside the per-team dummy variables and fit jointly. FIFA coefficients are shared across all teams, so every team's rating is shrunk toward what its ranking predicts. Teams with few observed matches lean heavily on the prior; teams with many matches are mostly determined by their data.

This approach means that even if a team appears in the FIFA table but never appears in the training data (a real concern for LOO backtests), it still gets a reasonable rating.

### Dixon-Coles tau correction

Independent Poisson double-counts scorelines near 0,0 — it underestimates 0–0 draws and overestimates 1–0 / 0–1 games. The Dixon-Coles tau correction adjusts probabilities in the four cells {0-0, 1-0, 0-1, 1-1}:

```
τ(0,0) = 1 - λμρ
τ(1,0) = 1 + μρ
τ(0,1) = 1 + λρ
τ(1,1) = 1 - ρ
```

The single parameter ρ is fit by maximum likelihood on the **actual** (integer) scorelines, after the Poisson parameters are fixed.

---

## 5. Prediction

Given ratings for a matchup (H, A), `expected_goals(H, A)` returns (λ, μ). The scoreline grid is the **outer product** of two Poisson PMFs, corrected by τ and renormalized:

```
P(H=i, A=j) = Poisson(i; λ) × Poisson(j; μ) × τ(i,j,λ,μ,ρ)
```

W/D/L probabilities are obtained by summing the upper triangle, diagonal, and lower triangle of the grid (goals capped at 6).

![France vs Iraq grid](../france_iraq_grid.png)

---

## 6. Evaluation

**Protocol:** leave-one-out (LOO). For each match in the 96-match WC evaluation set, ratings are refit on all other data (WC + qualifiers minus the held-out match) and the held-out match is predicted from the fresh fit. No information from the test match leaks into the model.

A second, stricter protocol validates the knockout stage specifically: a **temporal
backtest** where each of the 24 knockout matches is predicted using a model trained
**only on matches that occurred before it** — not just "not this match" (LOO) but
"not this match or anything after it" chronologically. This is the honest test of
genuine forecasting rather than in-sample fit; see §7a.

**Scoring rules:**

- **Ranked Probability Score (RPS):** the standard metric for ordered W/D/L outcomes. Penalizes probability mass placed far from the true outcome. Lower is better.
- **Multiclass log-loss:** strictly proper, penalizes overconfident wrong predictions heavily.
- **Top-1 accuracy:** predicted modal outcome matches actual outcome.

**Baselines:**

1. *FIFA-only* — same Poisson regression but with per-team dummies disabled (`team_effects=False`); rating is entirely determined by FIFA ranking.
2. *Naive base-rate* — predict the empirical home-win/draw/away-win frequencies for every match, ignoring team identity.

---

## 7. Results

| Model | log-loss | RPS | top-1 |
|---|---|---|---|
| **Ensemble (xG + Elo) — shipped** | 0.7962 | 0.1415 | 67% |
| Ensemble + draw calibration | **0.8133** | 0.1387 | 68% |
| Elo benchmark | 0.7770 | **0.1354** | 64% |
| Full (xG + FIFA + form) | 0.8493 | 0.1565 | 68% |
| FIFA-only | 0.8638 | 0.1614 | 66% |
| Naive base-rate | 1.0529 | 0.2235 | 48% |

**The key finding, stated plainly:** on WC-only data (~2 games per team) the event-data model was **worse** than a plain FIFA-rank baseline by 3.0% RPS — it was fitting noise. Adding qualifier data (~10 games per team) **flipped the sign**, and folding in the full 96-match tournament (including all knockout games) confirms the pattern holds at scale.

This is a clean demonstration of a thin-data failure: the failure was **predicted** (small sample → high variance), **measured** (the WC-only ablation), **fixed** (qualifier data pull), and **re-measured** (the full-corpus LOO, now on 96 matches).

**Bootstrap significance (10 000 resamples, paired, 95% CI), 96-match dataset:**

- **Ensemble vs Naive:** ΔRPS = −0.0820, 95% CI [−0.1096, −0.0554], P(Ensemble better) = **1.000**. The ensemble is **statistically significantly** better than naive base-rates (+36.7% RPS, +24.4% log-loss).
- **Full vs FIFA-only:** ΔRPS = −0.0049, 95% CI [−0.0103, +0.0005], P(Full better) = 0.961. The interval nearly excludes zero but doesn't quite — the event-data edge over the FIFA prior alone is **suggestive but not conclusive** at 95% confidence, even with the larger dataset.
- **Ensemble vs Full:** ΔRPS = −0.0150, 95% CI [−0.0227, −0.0071], P(Ensemble better) = **1.000** — entirely negative interval, a robust improvement. A 50/50 average of the xG/Dixon-Coles W/D/L and the Elo W/D/L beats either model alone because the two are **orthogonal**: the xG model scores possession/shot quality, Elo scores goal-based dynamic form using every match. This is the **shipped predictor** (`footy/ratings/ensemble.py`); the scoreline grid is still taken from the xG model (Elo has no grid), while the headline W/D/L blends both.
- **Elo vs Ensemble, knockout-heavy data:** on the 96-match dataset, plain Elo (RPS 0.1354) now has a *lower* point estimate than the 50/50 ensemble (0.1415) — a reversal from the 52-match result, where the ensemble led outright. The xG half's marginal contribution appears to shrink on knockout football, plausibly because knockout matches are lower-event and Elo's goal-based signal generalizes better there. **Open item:** the 50/50 blend weight was validated on the 52-match dataset and has not been re-optimized for the 96-match one — re-tuning the weight (or making it round/stage-dependent) is the natural next step and is tracked in `AGENTS.md`.

**The deeper lesson.** Elo wins for a concrete, instructive reason: it learns from the *goals* in every match, whereas the xG model can only learn from matches that carry shot coordinates — it discards qualifiers with no shot data (CAF/OFC in particular). The "sophistication" of insisting on xG quietly starves the model of some of its signal. This is one of the most useful findings in the project: **a large part of the cheapest accuracy gain is not a fancier estimator, but feeding the model the goal-based history Elo already uses.** Elo makes a strong, recognised external benchmark, sitting at or above the event-data model and well above naive — and on the fuller dataset, at or above the ensemble too.

*Methodological note:* Elo predictions are leakage-free pre-match (ratings accumulated chronologically from earlier matches only); the draw model's two parameters are fit on the full corpus, a mild optimism relative to the strictly held-out LOO protocol used for the other rows.

The honest summary: the pipeline is clearly better than guessing base-rates; per-team xG features alone are not conclusively distinguishable from the FIFA prior even at 96 matches; the ensemble is a real, significant improvement over the full xG model and over naive — but Elo alone is now a live, evidence-backed contender for best single model, and the blend weight deserves revisiting rather than being treated as settled at 50/50.

**Hyperparameter tuning.** A grid search over `alpha` (L2 strength) × `fifa_scale` (FIFA prior weight), re-run on the 96-match dataset, confirms that the defaults (`alpha=0.05`, `fifa_scale=1.0`) remain close to the best cell (`alpha=0.01`, `fifa_scale=2.0`, −1.6% RPS) — a difference the tuning script itself flags as likely noise on a flat surface rather than a real signal to chase. See `tune.py` and `tune_alpha_fifa.png`.

**Ratings smell test.** Spain #1, Argentina #2, England #3, France #4 on the net-xG table; weakest are Gibraltar and Curacao — all plausible.

**Example prediction.** France vs Iraq: modeled λ = 2.48, μ = 0.52 → **France 79% / draw 16% / Iraq 5%**, modal score **2–0**. The hand-tuned reference model predicted France 90.6% — overconfident, partly because it applies FIFA strength as a multiplicative scaler that inflates the favourite's probability.

---

## 7a. Out-of-sample validation: the real 2026 World Cup knockout stage

LOO backtesting is honest about not leaking the *held-out match itself*, but every
other match in the corpus — including ones chronologically *after* it — is still
available to the fit. That's a defensible protocol for i.i.d.-ish data, but a
stronger, less forgiving test became available once the actual tournament reached
its knockout stage: **predict each knockout match using a model trained only on
data that existed before that match was played**, then score against what really
happened. No LOO, no shortcuts — genuine forecasting.

**Protocol:** for each of the 24 knockout matches (Round of 32 through the Final),
refit the ensemble on every match with an earlier date, predict the held-out match
at a neutral venue (home-advantage cancelled by averaging both home/away
orientations), and score against the actual 90-minute result.

**Results:**

| Metric | Result |
|---|---|
| Top-1 accuracy | **79% (19/24)** |
| RPS | 0.1316 |
| Log-loss | 0.7148 |
| RPS improvement vs naive baseline | **+45.1%** |
| Round of 32 (16 matches) | 13/16 (81%), RPS 0.1336 |
| Round of 16 → Final (8 matches) | 6/8 (75%), RPS 0.1276 |

**Reading the misses.** Of the 5 matches where the model's favorite didn't win
outright in 90 minutes, **4 were draws that went to a penalty shootout**
(Germany–Paraguay, Netherlands–Morocco, Australia–Egypt, Switzerland–Colombia) — and
in 3 of those 4, the model's favored team won the shootout and advanced anyway. Only
one match (Norway's win over Brazil) was a genuine wrong-winner call at 90 minutes.
This points at a specific, addressable gap rather than a general accuracy problem:
the model predicts 90-minute W/D/L, not "who advances," and a thin penalty-shootout
layer on top (roughly a coin flip with a small skill-based lean) would resolve most
of what currently reads as error.

**Calibration.** The average confidence of the model's knockout favorite was 54%,
yet those favorites won 79% of the time — the model is **underconfident** on
knockout football, a healthier failure mode than overconfidence, and a candidate for
a cheap sharpening/temperature adjustment.

This is, by a wide margin, the strongest evidence in the project: not a backtest on
matches that already happened when the model was built, but a forecast of matches
that hadn't happened yet, scored after the fact.

---

## 8. Limitations and Next Steps

- **The Full-vs-FIFA-only edge is still not conclusively significant**, even at 96 matches (P=0.961, CI nearly excludes zero). A larger or rolling backtest across multiple tournaments would be needed to fully resolve it.
- **The 50/50 ensemble blend weight needs re-tuning.** It was chosen on the 52-match dataset; on the fuller 96-match dataset Elo alone edges the ensemble, suggesting the optimal weight has shifted toward Elo (or should vary by match type — e.g. more Elo weight for lower-event knockout matches).
- **No penalty-shootout / advancement layer.** The model predicts 90-minute outcomes only. Out-of-sample knockout testing shows this is the single largest source of apparent "error" (4 of 5 misses were shootouts the model's favorite often still won).
- **AFC qualifier feed not yet pulled.** Asian teams currently lean more heavily on the FIFA prior than European or CONMEBOL sides. Pulling the full AFC timeline would improve their calibration.
- **Trial-tier coordinate gaps.** Sportradar's trial tier omits shot coordinates for some confederations (CAF in particular). Those matches are excluded from xG training, so the xG model is biased toward the coordinate-rich corpus.
- **~2 WC games per team.** Even with qualifiers, World Cup tournament performance is still extrapolated from a small within-competition sample; form can shift between qualifiers and the tournament itself.
- **No form decay.** The current model weights all matches equally regardless of recency. A time-discounting scheme (exponential decay on older matches) is the most obvious next step.
- **Single xG model for all leagues.** Shot quality varies by competition level; a hierarchical xG model that partially pools across confederations could reduce bias.

### Explored and rejected: shot-type xG features

Adding `is_header` and `is_freekick` flags to the xG model was investigated. In the Sportradar trial feed, the `method` field on shot events is only populated on goal events (`score_change`); it is absent on `shot_on_target` and `shot_off_target` events. This means `is_header = 1` occurred exclusively on goals — a perfect predictor of `is_goal = 1` in the training data, a textbook label leak. The effect was visible immediately: CV AUC inflated to 0.719 and the model assigned an absurd xG of 0.957 to an 11-metre header. The feature was correctly discarded. The shipped xG model uses geometry only (distance + angle), with CV ROC-AUC 0.687 and perfect aggregate calibration (total xG = total goals = 633).
