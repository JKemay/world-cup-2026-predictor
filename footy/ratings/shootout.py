"""Penalty-shootout / knockout-advancement probability, layered on top of W/D/L.

The core model predicts 90-minute outcomes only. For knockout fixtures, the
90-minute draw branch actually resolves via extra time + a penalty shootout —
a real "who advances" question the core model has no representation for. This
module adds a thin, opt-in layer for that, decoupled from ``EnsemblePredictor``
so it does not touch the existing W/D/L contract.

Design notes
------------
- The draw branch deliberately conflates extra time and penalties into one
  shrunk logistic function of the Elo rating gap — the pipeline has no
  separate ET/shootout event data to fit those branches independently.
- ``SHOOTOUT_ELO_SCALE = 2000.0`` is fixed *a priori*, not fitted. It is 5x
  flatter than match-Elo's implicit ~400-point scale, mapping a 100-point Elo
  gap to ~52.9% and a 300-point gap to ~58.5% — consistent with the
  football-research consensus that shootouts are close to a coin flip with
  only a mild quality lean. This constant must NOT be tuned against the small
  number of real shootouts observed in any single tournament (n=4 in the 2026
  WC knockout stage is pure noise for fitting a parameter) — same
  fixed-constants discipline as ``sos_weighting`` in dixon_coles.py.
- No home advantage: knockout draws needing extra time are neutral-venue by
  the time they reach a shootout.
"""

from __future__ import annotations

SHOOTOUT_ELO_SCALE: float = 2000.0


def shootout_win_prob(rating_gap: float, scale: float = SHOOTOUT_ELO_SCALE) -> float:
    """P(team A wins the extra-time-and-penalties branch), given an Elo gap.

    Parameters
    ----------
    rating_gap : float
        ``elo_A - elo_B``. No home advantage applied.
    scale : float
        Logistic scale (larger = flatter = closer to a coin flip). Fixed a
        priori at :data:`SHOOTOUT_ELO_SCALE`; do not fit against observed
        shootouts.
    """
    return 1.0 / (1.0 + 10.0 ** (-rating_gap / scale))


def advancement_prob(
    wdl: "list[float] | tuple[float, float, float]",
    rating_gap: float,
    scale: float = SHOOTOUT_ELO_SCALE,
) -> tuple[float, float]:
    """P(team A advances), P(team B advances) for a knockout fixture.

    Combines the 90-minute W/D/L with the shootout branch: team A advances if
    it wins in 90 minutes, or if the match is drawn and it wins the
    extra-time-and-penalties branch.

    Parameters
    ----------
    wdl : sequence of 3 floats
        ``[p_home_win, p_draw, p_away_win]`` from the core model, for the
        fixture oriented with team A as "home" (e.g. from a neutral-venue
        average — see :func:`footy.evaluate.backtest.temporal_backtest`).
    rating_gap : float
        ``elo_A - elo_B``.
    """
    p_win, p_draw, p_loss = wdl
    p_a_shootout = shootout_win_prob(rating_gap, scale=scale)
    p_a = p_win + p_draw * p_a_shootout
    return p_a, 1.0 - p_a
