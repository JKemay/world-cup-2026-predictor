"""FIFA Men's World Ranking for the 48 World Cup 2026 teams, used as a prior.

The ranking only seeds a *prior* that stabilises team ratings on the thin
in-tournament sample — so exact values beyond the top tier are not critical.
Top-20 are the FIFA ranking as of 11 June 2026 (FIFA via Wikipedia); ranks
beyond 20 are close approximations and can be refined freely.
"""

from __future__ import annotations

import numpy as np

# Sportradar name → canonical name in FIFA_RANK
_SR_ALIASES: dict[str, str] = {
    "United States": "USA",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "Turkey": "Turkiye",
    "Türkiye": "Turkiye",
    "Czech Republic": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "DR Congo": "Congo DR",
    "Democratic Republic of Congo": "Congo DR",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
}


def normalize_team(name: str) -> str:
    """Map a Sportradar team name to the canonical name used in FIFA_RANK."""
    return _SR_ALIASES.get(name, name)


FIFA_RANK: dict[str, int] = {
    # top 20 — FIFA ranking, 11 June 2026
    "Argentina": 1, "Spain": 2, "France": 3, "England": 4, "Portugal": 5,
    "Brazil": 6, "Morocco": 7, "Netherlands": 8, "Belgium": 9, "Germany": 10,
    "Croatia": 11, "Colombia": 13, "Mexico": 14, "Senegal": 15, "Uruguay": 16,
    "USA": 17, "Japan": 18, "Switzerland": 19, "IR Iran": 20,
    # rank > 20 — approximate
    "Austria": 22, "Ecuador": 23, "Korea Republic": 24, "Australia": 25,
    "Turkiye": 26, "Norway": 28, "Panama": 30, "Egypt": 31, "Algeria": 32,
    "Canada": 33, "Sweden": 38, "Scotland": 39, "Ivory Coast": 40,
    "Paraguay": 41, "Tunisia": 42, "Czechia": 43, "Congo DR": 46,
    "Uzbekistan": 52, "Qatar": 53, "Saudi Arabia": 56, "Iraq": 58,
    "South Africa": 60, "Jordan": 62, "Curacao": 64,
    "Bosnia and Herzegovina": 68, "Cape Verde": 70, "Ghana": 73,
    "Haiti": 83, "New Zealand": 86,
}


def fifa_strength(teams) -> dict[str, float]:
    """Map each team to a standardized strength (higher = stronger) via -log(rank)."""
    raw = {t: -np.log(FIFA_RANK.get(t, 50)) for t in teams}
    vals = np.array(list(raw.values()))
    mean, std = float(vals.mean()), float(vals.std())
    return {t: (v - mean) / std for t, v in raw.items()}
