"""Tests for footy/ratings/shootout.py — penalty-shootout / advancement layer."""

import pytest

from footy.ratings.shootout import SHOOTOUT_ELO_SCALE, advancement_prob, shootout_win_prob


class TestShootoutWinProb:
    def test_zero_gap_is_coin_flip(self):
        assert shootout_win_prob(0.0) == pytest.approx(0.5)

    def test_monotone_increasing_in_gap(self):
        gaps = [-300, -100, 0, 100, 300]
        probs = [shootout_win_prob(g) for g in gaps]
        assert probs == sorted(probs)

    def test_antisymmetric(self):
        for gap in [50.0, 150.0, 400.0]:
            assert shootout_win_prob(gap) == pytest.approx(1.0 - shootout_win_prob(-gap))

    def test_bounded_in_unit_interval(self):
        for gap in [-5000.0, -100.0, 0.0, 100.0, 5000.0]:
            p = shootout_win_prob(gap)
            assert 0.0 < p < 1.0

    def test_shrinkage_guard_large_gap_stays_near_coinflip(self):
        """A 400-Elo-point gap (a large gap in this pipeline's rating scale)
        must not push the shootout probability far from 50/50 — protects the
        a-priori SHOOTOUT_ELO_SCALE from silently becoming a sharp predictor."""
        assert shootout_win_prob(400.0) < 0.65

    def test_custom_scale_flatter_is_closer_to_half(self):
        p_default = shootout_win_prob(200.0, scale=SHOOTOUT_ELO_SCALE)
        p_flatter = shootout_win_prob(200.0, scale=SHOOTOUT_ELO_SCALE * 10)
        assert abs(p_flatter - 0.5) < abs(p_default - 0.5)


class TestAdvancementProb:
    def test_sums_to_one(self):
        p_a, p_b = advancement_prob([0.4, 0.3, 0.3], rating_gap=50.0)
        assert p_a + p_b == pytest.approx(1.0)

    def test_each_in_unit_interval(self):
        p_a, p_b = advancement_prob([0.4, 0.3, 0.3], rating_gap=50.0)
        assert 0.0 < p_a < 1.0
        assert 0.0 < p_b < 1.0

    def test_no_draw_possible_reduces_to_ninety_minute_result(self):
        """wdl=[0.6, 0, 0.4] with no draw mass: advancement == 90-minute win prob."""
        p_a, p_b = advancement_prob([0.6, 0.0, 0.4], rating_gap=0.0)
        assert p_a == pytest.approx(0.6)
        assert p_b == pytest.approx(0.4)

    def test_certain_draw_reduces_to_shootout_prob(self):
        """wdl=[0, 1, 0]: advancement equals the shootout win probability alone."""
        gap = 150.0
        p_a, _ = advancement_prob([0.0, 1.0, 0.0], rating_gap=gap)
        assert p_a == pytest.approx(shootout_win_prob(gap))

    def test_favorite_gets_more_than_ninety_minute_win_share_when_favored_in_shootout_too(self):
        """A favorite with a positive rating gap picks up extra advancement
        probability from the draw branch beyond its raw 90' win share."""
        wdl = [0.4, 0.3, 0.3]
        p_a, _ = advancement_prob(wdl, rating_gap=200.0)
        assert p_a > wdl[0]

    def test_symmetric_fixture_gives_fifty_fifty(self):
        """Draw-heavy, evenly-matched fixture (equal win shares, zero gap) ->
        both teams advance with equal probability."""
        p_a, p_b = advancement_prob([0.35, 0.3, 0.35], rating_gap=0.0)
        assert p_a == pytest.approx(p_b)
        assert p_a == pytest.approx(0.5)
