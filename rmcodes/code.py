"""Reed-Muller code construction over GF(2).

RM(r, m) is the set of evaluation vectors of all Boolean polynomials in
x_1, ..., x_m of degree at most r, evaluated at every point of GF(2)^m.

Conventions used throughout this package:
  * Codeword coordinates are indexed by integers 0 .. 2^m - 1. Coordinate x
    is the point of GF(2)^m whose i-th component x_i is bit i of x.
  * Message bits are coefficients of monomials, ordered by degree
    (constant first, then x_1 .. x_m, then x_1 x_2, x_1 x_3, ...).
"""

from itertools import combinations
from math import comb

import numpy as np


class ReedMullerCode:
    def __init__(self, r: int, m: int):
        if not 0 <= r <= m:
            raise ValueError("need 0 <= r <= m")
        self.r, self.m = r, m
        self.n = 2 ** m
        self.k = sum(comb(m, i) for i in range(r + 1))
        self.d_min = 2 ** (m - r)

        # Monomials as tuples of variable indices, grouped by degree.
        self.monomials = [s for d in range(r + 1) for s in combinations(range(m), d)]

        # points[x, i] = x_i, the i-th coordinate of point x.
        points = (np.arange(self.n)[:, None] >> np.arange(m)) & 1
        self.points = points.astype(np.uint8)

        # Generator matrix: row j is the evaluation vector of monomial j.
        G = np.ones((self.k, self.n), dtype=np.uint8)
        for j, mono in enumerate(self.monomials):
            for var in mono:
                G[j] &= self.points[:, var]
        self.G = G

    @property
    def rate(self) -> float:
        return self.k / self.n

    @property
    def t(self) -> int:
        """Guaranteed number of correctable errors, floor((d_min - 1) / 2)."""
        return (self.d_min - 1) // 2

    def encode(self, u: np.ndarray) -> np.ndarray:
        """Encode message bits u of shape (k,) or (batch, k)."""
        return (np.atleast_2d(u).astype(np.int64) @ self.G % 2).astype(np.uint8).reshape(
            *np.shape(u)[:-1], self.n
        )

    def degree_slices(self):
        """Yield (degree, slice of message indices) for each degree 0..r."""
        start = 0
        for d in range(self.r + 1):
            stop = start + comb(self.m, d)
            yield d, slice(start, stop)
            start = stop

    def __repr__(self) -> str:
        return f"RM({self.r},{self.m}) [n={self.n}, k={self.k}, d={self.d_min}]"
