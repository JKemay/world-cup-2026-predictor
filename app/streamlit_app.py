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

from footy.ratings.dixon_coles import DixonColesRatings, grid_summary  # noqa: E402
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
def fit_model() -> DixonColesRatings:
    matches = load_matches()
    fifa = fifa_strength(sorted({*matches.home, *matches.away}))
    return DixonColesRatings(
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
        "Dixon-Coles attack/defense ratings fit on expected goals (xG) from "
        f"{len(matches)} matches ({len(matches) - n_qual} World Cup + {n_qual} qualifiers), "
        "anchored to a FIFA-ranking prior. Pick two teams for a full scoreline distribution."
    )

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "- **xG** from shot location (distance + angle), logistic regression\n"
            "- **Ratings** via regularized Poisson regression on xG, not raw goals\n"
            "- **FIFA rank** as a Bayesian-style prior, stabilizing thin samples\n"
            "- **Dixon-Coles τ** correction on low-scoring cells\n"
            "- **Backtest:** +20.5% RPS vs naive, edges a FIFA-only baseline"
        )
        st.markdown("---")
        st.markdown(
            "Built from Sportradar event data. "
            "[Pipeline & code](https://github.com/JKemay/world-cup-2026-predictor) "
            "· resume project."
        )

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
    p_home, p_draw, p_away = s["home_win"], s["draw"], s["away_win"]
    top_h, top_a = s["top_score"]

    st.markdown("### Outcome")
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

    with st.expander("Team ratings (att / def, expected xG vs an average opponent)"):
        rf = model.ratings_frame()
        rf = rf[rf["team"].isin(teams)].reset_index(drop=True)
        rf.index = rf.index + 1
        st.dataframe(
            rf.rename(columns={"att_xg": "Attack (xG for)",
                               "def_xg_allowed": "Defense (xG allowed)", "net": "Net"})
            .style.format({"Attack (xG for)": "{:.2f}",
                           "Defense (xG allowed)": "{:.2f}", "Net": "{:+.2f}"}),
            use_container_width=True, height=420,
        )


if __name__ == "__main__":
    main()
