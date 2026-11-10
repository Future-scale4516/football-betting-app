"""
Removing the bookmaker margin ("de-vigging").

Multiplicative (the old method) shrinks every price by the same factor, which
overstates longshots and understates favourites, because books really load more
margin onto longshots. Two better methods:

  power  - find k so that sum(p_i ** k) = 1 where p_i = 1/odds_i.
  shin   - Shin (1993): assumes a share z of money is from insiders and solves
           for z. Moves the margin to longshots, the usual finding for football.

Use them on ONE sharp book's prices (Pinnacle), not an average across books.
Every function takes decimal odds and returns probabilities summing to 1.
"""
from math import sqrt


def _implied(odds):
    return [1.0 / o for o in odds]


def multiplicative(*odds):
    imps = _implied(odds)
    t = sum(imps)
    return [i / t for i in imps]


def power(*odds):
    imps = _implied(odds)
    if abs(sum(imps) - 1) < 1e-12:
        return imps
    lo, hi = 0.0, 1.0
    # sum(imps**k) is decreasing in k; margin>0 means k>1 is needed.
    hi = 1.0
    while sum(i ** hi for i in imps) > 1 and hi < 100:
        hi *= 2
    lo = 0.0 if sum(imps) < 1 else 1.0
    if sum(imps) < 1:                    # negative margin: k<1
        hi = 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(i ** mid for i in imps) > 1:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    return [i ** k for i in imps]


def shin(*odds):
    imps = _implied(odds)
    book = sum(imps)
    n = len(imps)
    if book <= 1 + 1e-12 or n < 2:
        return multiplicative(*odds)

    def probs(z):
        return [(sqrt(z * z + 4 * (1 - z) * i * i / book) - z) / (2 * (1 - z))
                for i in imps]

    lo, hi = 0.0, 0.4
    for _ in range(100):                  # sum(probs) > 1 for small z
        mid = (lo + hi) / 2
        if sum(probs(mid)) > 1:
            lo = mid
        else:
            hi = mid
    p = probs((lo + hi) / 2)
    s = sum(p)
    return [x / s for x in p]


METHODS = {"multiplicative": multiplicative, "power": power, "shin": shin}
DEFAULT_METHOD = "shin"


def devig(*odds, method=DEFAULT_METHOD):
    return METHODS[method](*odds)
