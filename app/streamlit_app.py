#!/usr/bin/env python3
"""World Cup 2026 match predictor — interactive Streamlit dashboard.

Pick any two teams; get the Poisson scoreline grid, W/D/L probabilities, and a
prediction card. The ratings are fit (cached) from the same Dixon-Coles model
the pipeline uses, on the committed match-table snapshot so the app is
self-contained and deployable.

    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from footy.ratings.dixon_coles import grid_summary  # noqa: E402
from footy.ratings.ensemble import EnsemblePredictor  # noqa: E402
from footy.ratings.fifa import FIFA_RANK, fifa_strength  # noqa: E402

ALPHA, FIFA_SCALE = 0.05, 1.0
WC_SEASON = "sr:season:101177"
SNAPSHOT = APP_DIR / "match_table.csv"

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load_matches() -> pd.DataFrame:
    """Load the committed match-table snapshot, or rebuild it from the cache."""
    if SNAPSHOT.exists():
        return pd.read_csv(SNAPSHOT)
    from footy.features.matches import build_match_table  # local raw cache
    return build_match_table()


@st.cache_resource(show_spinner="Fitting team ratings…")
def fit_model() -> EnsemblePredictor:
    matches = load_matches()
    fifa = fifa_strength(sorted({*matches.home, *matches.away}))
    return EnsemblePredictor(
        alpha=ALPHA, fifa=fifa, fifa_scale=FIFA_SCALE, team_effects=True
    ).fit(matches)


def scoreline_figure(grid: np.ndarray, home: str, away: str) -> go.Figure:
    """Plotly heatmap of P(home_goals=i, away_goals=j)."""
    n = grid.shape[0]
    pct = grid * 100.0
    text = [[f"{pct[i, j]:.1f}" for j in range(n)] for i in range(n)]
    fig = go.Figure(
        go.Heatmap(
            z=pct,
            x=[str(j) for j in range(n)],
            y=[str(i) for i in range(n)],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="Tealgrn",
            hovertemplate=(f"{home} %{{y}} – %{{x}} {away}<br>%{{z:.1f}}%<extra></extra>"),
            colorbar=dict(title="prob %"),
        )
    )
    top_i, top_j = np.unravel_index(grid.argmax(), grid.shape)
    fig.add_shape(
        type="rect", x0=top_j - 0.5, x1=top_j + 0.5, y0=top_i - 0.5, y1=top_i + 0.5,
        line=dict(color="#e4572e", width=3),
    )
    fig.update_layout(
        xaxis_title=f"{away} goals",
        yaxis_title=f"{home} goals",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=460,
    )
    return fig


def main() -> None:
    model = fit_model()
    matches = load_matches()

    # teams we can predict: rated AND actual WC 2026 entrants, for a clean dropdown
    rated = set(model.attack_)
    teams = sorted(t for t in FIFA_RANK if t in rated)
    n_qual = int((matches["season_id"] != WC_SEASON).sum()) if "season_id" in matches else 0

    st.title("⚽ World Cup 2026 — Match Predictor")
    st.caption(
        "An ensemble of a Dixon-Coles xG model (FIFA-anchored) and an Elo rating, fit on "
        f"{len(matches)} matches ({len(matches) - n_qual} World Cup + {n_qual} qualifiers). "
        "Pick two teams for a full scoreline distribution."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "- **xG** from shot location (distance + angle), logistic regression\n"
            "- **Ratings** via regularized Poisson regression on xG, not raw goals\n"
            "- **FIFA rank** as a Bayesian-style prior, stabilizing thin samples\n"
            "- **Elo rating** (goal-based) ensembled with the xG model for W/D/L\n"
            "- **Backtest:** ensemble is +36.7% RPS vs naive (P=1.000); "
            "out-of-sample on the real 2026 knockout stage: 79% top-1 (19/24), +45.1% RPS vs naive"
        )
        st.markdown("---")
        st.markdown(
            "Built from Sportradar event data. "
            "[Pipeline & code](https://github.com/JKemay/world-cup-2026-predictor) "
            "· [Live app](https://world-cup-2026-ml.streamlit.app) · resume project."
        )

    tab_predict, tab_ratings, tab_model = st.tabs(["🔮 Predict", "📊 Team ratings", "📈 Model & validation"])

    # ------------------------------------------------------------------ #
    # TAB 1 — Predict                                                      #
    # ------------------------------------------------------------------ #
    with tab_predict:
        di, df_ = (teams.index("France") if "France" in teams else 0,
                   teams.index("Iraq") if "Iraq" in teams else 1)
        c1, c2, c3 = st.columns([5, 1, 5])
        with c1:
            home = st.selectbox("🏠 Home team", teams, index=di)
        with c2:
            st.markdown("<div style='text-align:center;padding-top:1.9em;font-size:1.4em'>vs</div>",
                        unsafe_allow_html=True)
        with c3:
            away = st.selectbox("✈️ Away team", teams, index=df_)

        if home == away:
            st.warning("Pick two different teams.")
            st.stop()

        grid, lam, mu = model.scoreline_grid(home, away, max_goals=6)
        s = grid_summary(grid)
        top_h, top_a = s["top_score"]

        # Ensemble W/D/L (blends xG model + Elo)
        ens_wdl = model.wdl(home, away)
        p_home, p_draw, p_away = float(ens_wdl[0]), float(ens_wdl[1]), float(ens_wdl[2])

        st.markdown("### Outcome")
        st.caption("W/D/L blends the xG model with an Elo rating; the scoreline grid is from the xG model.")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"🏠 {home} win", f"{p_home*100:.0f}%")
        m2.metric("Draw", f"{p_draw*100:.0f}%")
        m3.metric(f"✈️ {away} win", f"{p_away*100:.0f}%")

        # probability bar
        bar = go.Figure()
        for label, val, color in [
            (f"{home}", p_home, "#1b9e77"), ("Draw", p_draw, "#999999"),
            (f"{away}", p_away, "#e4572e"),
        ]:
            bar.add_trace(go.Bar(
                x=[val * 100], y=["W/D/L"], orientation="h", name=label,
                marker_color=color, text=f"{label} {val*100:.0f}%",
                textposition="inside", insidetextanchor="middle",
            ))
        bar.update_layout(
            barmode="stack", height=90, showlegend=False,
            margin=dict(l=10, r=10, t=4, b=4),
            xaxis=dict(range=[0, 100], showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        st.plotly_chart(bar, use_container_width=True)

        if st.toggle("Knockout tie (extra time + penalties if drawn)"):
            from footy.ratings.shootout import advancement_prob

            fwd = model.wdl(home, away)
            rev = model.wdl(away, home)
            wdl_neutral = np.array([(fwd[0] + rev[2]) / 2, (fwd[1] + rev[1]) / 2, (fwd[2] + rev[0]) / 2])
            wdl_neutral = wdl_neutral / wdl_neutral.sum()
            gap = model.elo_.ratings.get(home, 1500.0) - model.elo_.ratings.get(away, 1500.0)
            p_home_adv, p_away_adv = advancement_prob(wdl_neutral, gap)
            a1, a2 = st.columns(2)
            a1.metric(f"🏆 {home} advances", f"{p_home_adv*100:.0f}%")
            a2.metric(f"🏆 {away} advances", f"{p_away_adv*100:.0f}%")
            st.caption(
                "Neutral-venue W/D/L, with the drawn-after-90' branch resolved by a near-coin-flip "
                "extra-time-and-penalties model (a small, fixed skill-based lean from the Elo gap — "
                "not fitted to any specific tournament's shootouts, which are too few to fit reliably)."
            )

        left, right = st.columns([3, 2])
        with left:
            st.markdown("### Scoreline probabilities")
            st.plotly_chart(scoreline_figure(grid, home, away), use_container_width=True)
        with right:
            st.markdown("### Prediction card")
            st.markdown(
                f"**Expected goals**\n\n"
                f"- {home}: **{lam:.2f}**\n"
                f"- {away}: **{mu:.2f}**\n\n"
                f"**Most likely score**\n\n"
                f"## {home} {top_h}–{top_a} {away}\n"
                f"<span style='color:#888'>({s['top_prob']*100:.1f}% of all scorelines)</span>",
                unsafe_allow_html=True,
            )
            # top 5 most likely exact scores
            flat = [((i, j), grid[i, j]) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
            flat.sort(key=lambda kv: kv[1], reverse=True)
            rows = [{"Score": f"{i}–{j}", "Prob": f"{p*100:.1f}%"} for (i, j), p in flat[:5]]
            st.table(pd.DataFrame(rows))

    # ------------------------------------------------------------------ #
    # TAB 2 — Team ratings                                                 #
    # ------------------------------------------------------------------ #
    with tab_ratings:
        st.markdown("### Team ratings — WC 2026 qualified nations")
        rf = model.ratings_frame()
        rf_wc = (
            rf[rf["team"].isin(teams)]
            .sort_values("net", ascending=False)
            .reset_index(drop=True)
        )
        rf_wc.index = rf_wc.index + 1
        st.dataframe(
            rf_wc.rename(columns={
                "att_xg": "Attack xG",
                "def_xg_allowed": "Defense xG allowed",
                "net": "Net",
            }).style.format({
                "Attack xG": "{:.2f}",
                "Defense xG allowed": "{:.2f}",
                "Net": "{:+.2f}",
            }),
            use_container_width=True,
            height=460,
        )

        st.markdown("### Attack vs defense (xG per match vs an average opponent)")
        med_att = rf_wc["att_xg"].median()
        med_def = rf_wc["def_xg_allowed"].median()
        top10 = set(rf_wc.head(10)["team"])

        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=rf_wc["att_xg"],
            y=rf_wc["def_xg_allowed"],
            mode="markers+text",
            text=[t if t in top10 else "" for t in rf_wc["team"]],
            textposition="top center",
            marker=dict(size=9, color="#1b9e77", opacity=0.75),
            hovertext=rf_wc["team"],
            hovertemplate="%{hovertext}<br>Att: %{x:.2f}  Def allowed: %{y:.2f}<extra></extra>",
        ))
        # dashed median guide lines
        scatter.add_hline(y=med_def, line_dash="dash", line_color="#aaaaaa", line_width=1)
        scatter.add_vline(x=med_att, line_dash="dash", line_color="#aaaaaa", line_width=1)
        scatter.update_layout(
            xaxis_title="Attack xG (higher = stronger attack)",
            yaxis_title="Defense xG allowed (lower = stronger defense)",
            yaxis=dict(autorange="reversed"),
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(scatter, use_container_width=True)
        st.caption(
            "Y-axis reversed: teams toward the top concede fewer xG (better defense). "
            "Up and to the right = strong on both ends. Dashed lines show the median."
        )

    # ------------------------------------------------------------------ #
    # TAB 3 — Model & validation                                           #
    # ------------------------------------------------------------------ #
    with tab_model:
        st.markdown("### How good is the model?")
        st.markdown(
            """
**Leave-one-out backtest** on all 96 WC 2026 matches + 324 qualifiers (ensemble = xG/Dixon-Coles + Elo, 50/50):

| Metric | Value |
|---|---|
| Ensemble RPS | **0.1415** |
| Ensemble log-loss | **0.7962** |
| Top-1 accuracy | **67%** |

**Bootstrap confidence intervals** (10 000 resamples):

- Ensemble vs Naive baseline: strongly significant improvement (P = 1.000, +36.7% RPS)
- Ensemble vs Full xG model: ΔRPS = −0.0150, 95% CI [−0.0227, −0.0071], P = 1.000
- Ensemble vs FIFA-only (via Full model): suggestive but not conclusive, P = 0.961

The ensemble is a statistically significant improvement over both the naive baseline and the
full xG model. On the fuller, knockout-inclusive dataset, plain Elo alone now has a lower point-
estimate RPS than the 50/50 blend — but a leakage-free (nested) re-tune of the blend weight
did **not** clear 95% significance (P = 0.948), so the ensemble ships at 50/50, unchanged.

**Out-of-sample validation** — the real 2026 World Cup knockout stage: predicting all 24
Round-of-32-through-Round-of-16 matches with a model trained only on data available *before*
each match (strict chronological cutoff, no leave-one-out shortcuts):

| Metric | Value |
|---|---|
| Top-1 accuracy | **79% (19/24)** |
| RPS improvement vs naive | **+45.1%** |

4 of the 5 misses were 90-minute draws that went to penalty shootouts, so most of what reads
as "error" is really the model having no representation of the shootout branch at all — not a
90-minute prediction mistake. A thin, fixed-parameter advancement layer for that case is
available via the "Knockout tie" toggle above.

**Hyperparameter tuning**, re-run on the 96-match dataset, confirmed the defaults
(α = 0.05, FIFA scale = 1.0) still sit near the RPS minimum on a flat surface.
"""
        )

        figures = [
            (ROOT / "calibration.png", "Reliability — predicted vs observed"),
            (ROOT / "model_analysis.png", "Calibration, biggest surprises, attack/defense landscape"),
            (ROOT / "tune_alpha_fifa.png", "RPS surface over regularization × FIFA-prior weight"),
            (ROOT / "xg_pitch.png", "Expected goals by shot location"),
        ]
        for fig_path, caption in figures:
            if fig_path.exists():
                st.image(str(fig_path), caption=caption, use_container_width=True)

        st.markdown("---")
        st.markdown("Full write-up: `docs/METHODOLOGY.md`")


if __name__ == "__main__":
    main()
