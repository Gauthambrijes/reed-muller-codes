"""Recursive projection-aggregation (RPA) decoding of Reed-Muller codes.

Implements the soft-decision RPA decoder of M. Ye and E. Abbe, "Recursive
projection-aggregation decoding of Reed-Muller codes" (IEEE Trans. IT 2020,
arXiv:1902.01470), Algorithms 3-4. Note that the paper writes RM(m, r); this
package writes RM(r, m). Both mean length 2^m and order r.

Mathematical summary
--------------------
Write E = GF(2)^m and let c = Eval(f) with deg f <= r.

* Projection. For a nonzero direction b, the coset {z, z + b} carries the bit
  c(z) + c(z + b) = (D_b f)(z), the discrete derivative of f along b. Since
  deg D_b f <= r - 1 and D_b f is constant on cosets of {0, b}, the projected
  word (c(z) + c(z + b) : cosets) is a codeword of RM(r - 1, m - 1)
  (Lemma 1 of the paper with s = 1).

* Projected LLR. Any two distinct coordinates of a random RM codeword are
  i.i.d. uniform bits (Lemma 2), so the LLR of c(z) + c(z + b) given the
  channel LLRs L1 = L(z), L2 = L(z + b) is the "box-plus" (eq. 12)

        L1 [+] L2 = ln(e^(L1 + L2) + 1) - ln(e^L1 + e^L2)
                  = 2 artanh(tanh(L1 / 2) tanh(L2 / 2)).

  The min-sum approximation sign(L1) sign(L2) min(|L1|, |L2|) overestimates
  the magnitude and never differs from the exact value by more than ln 2
  (see `boxplus_minsum_gap` and the tests).

* Recursion. Each of the n - 1 projected words is decoded as an
  RM(r - 1, m - 1) code; RM(1, *) is decoded exactly with the fast Hadamard
  transform.

* Aggregation (Algorithm 4). Each decoded projection gives an estimate of
  c(z) + c(z + b), hence a vote (1 - 2 yhat_b) L(z + b) for L(z). The new LLR
  is the average of the n - 1 votes. The projection/aggregation round is
  repeated up to floor(m / 2) times, or until every LLR changes by at most
  theta |L(z)| (theta = 0.05 as in the paper).

* Complexity (Proposition 3): O(n^r log n) operations.

For r = 2 there is only one level of recursion, so the collapsed
projection-aggregation (CPA) decoder, which projects directly onto
(r - 1)-dimensional subspaces, is the same algorithm as RPA.
"""

from functools import lru_cache

import numpy as np

from .code import ReedMullerCode
from .decoders import MajorityLogicDecoder, fast_hadamard_transform


# --------------------------------------------------------------------------
# Check-node style LLR combination
# --------------------------------------------------------------------------

def boxplus(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Exact LLR of the XOR of two independent bits with LLRs a and b."""
    return np.logaddexp(a + b, 0.0) - np.logaddexp(a, b)


def boxplus_minsum(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Min-sum approximation of `boxplus`."""
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def boxplus_minsum_gap(a, b):
    """minsum(a, b) - boxplus(a, b).

    The exact identity is
        boxplus(a, b) = minsum(a, b) + ln(1 + e^-|a + b|) - ln(1 + e^-|a - b|).
    The correction lies in [-ln 2, ln 2] and always has the opposite sign to
    minsum(a, b) (|a + b| >= |a - b| exactly when sign(a) = sign(b)). Hence
    |gap| <= ln 2 (approached when |a| = |b| is large), and min-sum never
    underestimates the reliability: |minsum| >= |boxplus|.
    """
    return boxplus_minsum(a, b) - boxplus(a, b)


# --------------------------------------------------------------------------
# Fixed-point emulation
# --------------------------------------------------------------------------

def make_quantizer(int_bits: int, frac_bits: int):
    """Round to a signed Q(int_bits).(frac_bits) grid and saturate symmetrically.

    Q7.8 is int_bits=7, frac_bits=8: a 16-bit word with step 1/256 and range
    +-(2^15 - 1) / 256. Saturation is symmetric (+-32767, not -32768) so the
    quantizer is an odd function, which keeps the decoder's error probability
    independent of the transmitted codeword (Proposition 4 of the paper needs
    the decoder to commute with sign flips).
    """
    scale = 2.0 ** frac_bits
    top = 2.0 ** (int_bits + frac_bits) - 1

    def q(x):
        return np.clip(np.round(x * scale), -top, top) / scale

    return q


# --------------------------------------------------------------------------
# Coset bookkeeping
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _projection_tables(m: int):
    """Index tables for projecting a length-2^m word onto all n - 1 directions.

    For direction b with lowest set bit j, each coset {z, z + b} is represented
    by its element with bit j = 0, and the coset's coordinate in GF(2)^(m-1) is
    that representative with bit j deleted. Returns
      reps  (n-1, n/2): representative of coset w, for w = 0 .. n/2 - 1
      mates (n-1, n/2): reps ^ b
      coord (n-1, n)  : coset coordinate of every point z
      shift (n-1, n)  : z ^ b
    """
    n = 2 ** m
    z = np.arange(n)
    reps, mates, coord, shift = [], [], [], []
    for b in range(1, n):
        j = (b & -b).bit_length() - 1
        rep = np.where((z >> j) & 1, z ^ b, z)
        cz = (rep & ((1 << j) - 1)) | ((rep >> (j + 1)) << j)
        r = z[((z >> j) & 1) == 0]  # increasing z with bit j = 0 is increasing coordinate
        reps.append(r)
        mates.append(r ^ b)
        coord.append(cz)
        shift.append(z ^ b)
    return tuple(np.array(a, dtype=np.int64) for a in (reps, mates, coord, shift))


@lru_cache(maxsize=None)
def _characters(m: int) -> np.ndarray:
    """chars[a, x] = (-1)^(a . x) for a, x in GF(2)^m."""
    x = np.arange(2 ** m)
    parity = np.array([bin(v).count("1") & 1 for v in range(2 ** m)], dtype=np.int8)
    return (1 - 2 * parity[x[:, None] & x[None, :]]).astype(np.float64)


def _fht_decode_pm(L: np.ndarray, m: int) -> np.ndarray:
    """Soft ML decoding of RM(1, m); returns the decision as a +-1 vector."""
    spec = fast_hadamard_transform(L)
    a = np.abs(spec).argmax(axis=1)
    sign = np.where(spec[np.arange(len(a)), a] >= 0, 1.0, -1.0)
    return sign[:, None] * _characters(m)[a]


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------

class RPADecoder:
    """Soft RPA decoder for RM(r, m), r >= 1.

    Parameters
    ----------
    min_sum     : use the min-sum projection instead of exact box-plus.
    n_iter      : maximum projection-aggregation rounds per recursion level
                  (default floor(m / 2), as in the paper).
    theta       : early-exit threshold (paper: 0.05).
    llr_format  : optional (int_bits, frac_bits); every stored LLR (channel,
                  projected, aggregated) is rounded to that fixed-point grid.
                  This emulates fixed-point storage, not accumulator width.
    """

    def __init__(self, code: ReedMullerCode, min_sum=False, n_iter=None, theta=0.05,
                 llr_format=None):
        if code.r < 1:
            raise ValueError("RPA needs r >= 1")
        self.code = code
        self.combine = boxplus_minsum if min_sum else boxplus
        self.n_iter = n_iter
        self.theta = theta
        self.q = make_quantizer(*llr_format) if llr_format else (lambda x: x)
        self._majority = MajorityLogicDecoder(code)

    # -- recursion ---------------------------------------------------------
    def _decode_pm(self, L: np.ndarray, r: int, m: int) -> np.ndarray:
        """Decode LLRs L (batch, 2^m) of RM(r, m); return a +-1 vector."""
        if r == 1:
            return _fht_decode_pm(L, m)
        if r == m:  # RM(m, m) is all of GF(2)^n: bitwise decisions
            return np.where(L >= 0, 1.0, -1.0)

        n = 2 ** m
        reps, mates, coord, shift = _projection_tables(m)
        n_iter = self.n_iter if self.n_iter is not None else max(1, m // 2)
        L = L.copy()
        active = np.ones(len(L), dtype=bool)

        for _ in range(n_iter):
            idx = np.nonzero(active)[0]
            if len(idx) == 0:
                break
            La = L[idx]
            # Projection: (B, n-1, n/2) LLRs of the derivative codewords.
            proj = self.q(self.combine(La[:, reps], La[:, mates]))
            B = len(idx)
            # Recursive decoding of n - 1 copies of RM(r-1, m-1).
            yhat = self._decode_pm(proj.reshape(B * (n - 1), n // 2), r - 1, m - 1)
            yhat = yhat.reshape(B, n - 1, n // 2)
            # Aggregation: vote (1 - 2 yhat_b([z + b])) L(z + b), averaged.
            votes = np.take_along_axis(yhat, np.broadcast_to(coord, (B,) + coord.shape), axis=2)
            new = self.q((votes * La[:, shift]).sum(axis=1) / (n - 1))
            stable = np.all(np.abs(new - La) <= self.theta * np.abs(La), axis=1)
            L[idx] = new
            active[idx[stable]] = False
        return np.where(L >= 0, 1.0, -1.0)

    # -- public API --------------------------------------------------------
    def decode_codeword(self, llr: np.ndarray, batch: int = 64) -> np.ndarray:
        """Channel LLRs (batch, n) -> estimated codeword bits (batch, n).

        The output is a hard decision on the final LLRs, so it is not
        guaranteed to be a codeword (the paper's list version fixes this).
        """
        llr = self.q(np.atleast_2d(np.asarray(llr, dtype=np.float64)))
        out = np.empty(llr.shape, dtype=np.uint8)
        for s in range(0, len(llr), batch):
            pm = self._decode_pm(llr[s:s + batch], self.code.r, self.code.m)
            out[s:s + batch] = pm < 0
        return out

    def decode(self, llr: np.ndarray, batch: int = 64) -> np.ndarray:
        """Channel LLRs -> message estimate (batch, k).

        The RPA output is mapped to a message with Reed's majority-logic
        decoder, which inverts the encoder exactly when the output is a
        codeword and otherwise returns a nearby codeword's message.
        """
        return self._majority.decode(self.decode_codeword(llr, batch))
