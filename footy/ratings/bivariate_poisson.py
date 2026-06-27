"""Bivariate Poisson PMF and scoreline grid (Karlis & Ntzoufras 2003)."""
import math

import numpy as np


def bivpois_pmf(x: int, y: int, lam_h: float, lam_a: float, lam3: float) -> float:
    """Joint PMF P(X=x, Y=y) under bivariate Poisson with covariance term lam3.

    KN2003 formula:
        exp(-(lam_h + lam_a + lam3)) * (lam_h^x/x!) * (lam_a^y/y!)
          * sum_{k=0}^{min(x,y)} C(x,k)*C(y,k)*k! * (lam3/(lam_h*lam_a))^k

    When lam3==0, reduces to independent Poisson product.
    """
    prefix = math.exp(-(lam_h + lam_a + lam3))
    prefix *= math.pow(lam_h, x) / math.factorial(x)
    prefix *= math.pow(lam_a, y) / math.factorial(y)

    total = 0.0
    for k in range(min(x, y) + 1):
        if k == 0:
            # (lam3/(lam_h*lam_a))^0 = 1 always (0^0 = 1 convention)
            ratio = 1.0
        else:
            if lam3 == 0.0:
                break  # higher-k terms vanish when lam3=0
            ratio = math.pow(lam3 / (lam_h * lam_a), k)
        total += math.comb(x, k) * math.comb(y, k) * math.factorial(k) * ratio

    return prefix * total


def bivpois_grid(lam_h: float, lam_a: float, lam3: float, max_goals: int = 6) -> np.ndarray:
    """Return (max_goals+1, max_goals+1) normalized joint probability grid."""
    size = max_goals + 1
    grid = np.empty((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            grid[i, j] = bivpois_pmf(i, j, lam_h, lam_a, lam3)
    grid /= grid.sum()
    return grid
