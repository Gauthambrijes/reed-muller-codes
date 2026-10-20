"""Weight distributions of Reed-Muller codes and union bounds for BPSK/AWGN.

* min_weight_count(r, m): number of minimum-weight codewords of RM(r, m),
      A_{2^(m-r)} = 2^r * prod_{i=0}^{m-r-1} (2^(m-i) - 1) / (2^(m-r-i) - 1).
  (The minimum-weight codewords are the indicators of the (m-r)-dimensional
  affine subspaces ("flats") of GF(2)^m.)

* weight_distribution(r, m): full distribution {w: A_w}, from
    - r = 1: biorthogonal code, A_{n/2} = 2n - 2, A_n = 1;
    - r = 2: Sloane-Berlekamp. Every quadratic form has rank 2h, and the weights
      2^(m-1) +- 2^(m-1-h) (1 <= h <= floor(m/2)) each occur
          A = 2^(h(h+1)) * prod_{i=m-2h+1}^{m} (2^i - 1) / prod_{i=1}^{h} (4^i - 1)
      times; the rest of the 2^k codewords other than 0 and 1 have weight 2^(m-1);
    - otherwise exhaustive enumeration (k <= 22).

* union_bound_bler: for ML decoding over BPSK/AWGN with all-zero codeword,
      P_block <= sum_w A_w Q( sqrt(2 R w Eb/N0) ),
  the pairwise-error union bound. Keeping only w = d_min gives the "truncated"
  union bound, a tight estimate of ML block error rate at high SNR.
  (Bhattacharyya form: Theorem 2 of Abbe, Sberlo, Shpilka, Ye, "Reed-Muller
  codes", Foundations and Trends in Communications and Information Theory.)
"""

from fractions import Fraction
from math import comb

import numpy as np
from scipy.special import erfc

from .code import ReedMullerCode


def Q(x):
    return 0.5 * erfc(np.asarray(x, dtype=float) / np.sqrt(2.0))


def min_weight_count(r: int, m: int) -> int:
    a = Fraction(2 ** r)
    for i in range(m - r):
        a *= Fraction(2 ** (m - i) - 1, 2 ** (m - r - i) - 1)
    assert a.denominator == 1
    return int(a)


def _rm2_distribution(m: int) -> dict:
    n, k = 2 ** m, 1 + m + comb(m, 2)
    dist = {0: 1, n: 1}
    total = 2
    for h in range(1, m // 2 + 1):
        num = Fraction(2 ** (h * (h + 1)))
        for i in range(m - 2 * h + 1, m + 1):
            num *= 2 ** i - 1
        for i in range(1, h + 1):
            num /= 4 ** i - 1
        assert num.denominator == 1
        a = int(num)
        for w in (n // 2 - 2 ** (m - 1 - h), n // 2 + 2 ** (m - 1 - h)):
            dist[w] = dist.get(w, 0) + a
            total += a
    dist[n // 2] = dist.get(n // 2, 0) + 2 ** k - total
    return dict(sorted(dist.items()))


def _enumerate_distribution(code: ReedMullerCode) -> dict:
    if code.k > 22:
        raise ValueError("too many codewords to enumerate")
    w = np.zeros(2 ** code.k, dtype=np.int64)
    step = 1 << min(code.k, 16)
    for s in range(0, 2 ** code.k, step):
        msgs = ((np.arange(s, s + step)[:, None] >> np.arange(code.k)) & 1).astype(np.uint8)
        w[s:s + step] = code.encode(msgs).sum(axis=1)
    vals, cnt = np.unique(w, return_counts=True)
    return dict(zip(vals.tolist(), cnt.tolist()))


def weight_distribution(r: int, m: int) -> dict:
    """{weight: number of codewords}, including weight 0."""
    n = 2 ** m
    if r == 0:
        return {0: 1, n: 1}
    if r == 1:
        return {0: 1, n // 2: 2 * n - 2, n: 1}
    if r == 2:
        return _rm2_distribution(m)
    return _enumerate_distribution(ReedMullerCode(r, m))


def union_bound_bler(r: int, m: int, ebn0_db, truncated: bool = False):
    """Union bound on ML block error rate over BPSK/AWGN."""
    code = ReedMullerCode(r, m)
    ebn0 = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    if truncated:
        terms = {code.d_min: min_weight_count(r, m)}
    else:
        terms = {w: a for w, a in weight_distribution(r, m).items() if w > 0}
    return sum(a * Q(np.sqrt(2 * code.rate * w * ebn0)) for w, a in terms.items())
