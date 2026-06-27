"""Tests for footy/ratings/bivariate_poisson.py."""

import math

import numpy as np
from scipy.stats import poisson

from footy.ratings.bivariate_poisson import bivpois_grid, bivpois_pmf


class TestBivariatePMF:
    def test_independence_fallback(self):
        """lam3=0 should equal independent Poisson product."""
        lh, la = 1.5, 0.8
        max_g = 6
        grid_bv = bivpois_grid(lh, la, 0.0, max_g)
        ind = np.outer(poisson.pmf(np.arange(max_g + 1), lh),
                       poisson.pmf(np.arange(max_g + 1), la))
        ind /= ind.sum()
        np.testing.assert_allclose(grid_bv, ind, atol=1e-8)

    def test_grid_sums_to_one(self):
        grid = bivpois_grid(1.5, 0.9, 0.3)
        assert abs(grid.sum() - 1.0) < 1e-8

    def test_all_nonnegative(self):
        grid = bivpois_grid(1.5, 0.9, 0.3)
        assert (grid >= 0).all()

    def test_lambda3_increases_draw_mass(self):
        lh, la = 1.5, 1.2
        g0 = bivpois_grid(lh, la, 0.0)
        g3 = bivpois_grid(lh, la, 0.3)
        draw0 = sum(g0[i, i] for i in range(g0.shape[0]))
        draw3 = sum(g3[i, i] for i in range(g3.shape[0]))
        assert draw3 > draw0

    def test_symmetry(self):
        lh, la, l3 = 1.5, 0.9, 0.2
        for x in range(4):
            for y in range(4):
                assert abs(bivpois_pmf(x, y, lh, la, l3) - bivpois_pmf(y, x, la, lh, l3)) < 1e-12

    def test_known_value(self):
        """P(0,0) = exp(-(lh+la+l3)) with only the k=0 term."""
        lh, la, l3 = 1.0, 1.0, 0.5
        expected = math.exp(-(lh + la + l3))
        assert abs(bivpois_pmf(0, 0, lh, la, l3) - expected) < 1e-10
