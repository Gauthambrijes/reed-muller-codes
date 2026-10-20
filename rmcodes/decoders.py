"""Decoders for Reed-Muller codes.

* majority_logic_decode : Reed's hard-decision majority-logic decoder.
* ml_decode_rm1         : soft maximum-likelihood decoding of RM(1, m) using
                          the fast Hadamard transform.
* ml_decode_bruteforce  : soft maximum-likelihood decoding by correlating with
                          every codeword (only practical for small k).

Soft decoders take BPSK channel outputs y = (1 - 2c) + noise.
All decoders work on a batch of received words and return message estimates.
"""

from itertools import product

import numpy as np

from .code import ReedMullerCode


# --------------------------------------------------------------------------
# Reed majority-logic decoding (hard decision)
# --------------------------------------------------------------------------

def _coset_index(code: ReedMullerCode, mono: tuple) -> np.ndarray:
    """Coordinates grouped into cosets of the subspace spanned by `mono`.

    Returns an array of shape (2^(m-d), 2^d). Row c holds the 2^d points that
    agree with c on the variables outside `mono` and take every value on the
    variables in `mono`. Summing the received word over one row is one
    independent check-sum (vote) for the coefficient of `mono`.
    """
    m = code.m
    inside = list(mono)
    outside = [i for i in range(m) if i not in mono]
    inside_vals = np.array(
        [sum(b << v for b, v in zip(bits, inside)) for bits in product((0, 1), repeat=len(inside))],
        dtype=np.int64,
    )
    outside_vals = np.array(
        [sum(b << v for b, v in zip(bits, outside)) for bits in product((0, 1), repeat=len(outside))],
        dtype=np.int64,
    )
    return outside_vals[:, None] | inside_vals[None, :]


class MajorityLogicDecoder:
    """Reed's algorithm, decoding from the highest degree down.

    For each monomial of degree d, every coset sum gives a vote equal to the
    monomial's coefficient (all other terms of degree <= d sum to zero over
    the coset, and higher-degree terms have already been removed). The
    coefficient is the majority of the 2^(m-d) votes. Corrects any pattern of
    up to 2^(m-r-1) - 1 errors.
    """

    def __init__(self, code: ReedMullerCode):
        self.code = code
        self.cosets = {mono: _coset_index(code, mono) for mono in code.monomials}

    def decode(self, z: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """z: hard-decision bits, shape (batch, n). Returns u_hat, shape (batch, k)."""
        code = self.code
        z = np.atleast_2d(z).astype(np.uint8).copy()
        batch = z.shape[0]
        u_hat = np.zeros((batch, code.k), dtype=np.uint8)

        for d, sl in reversed(list(code.degree_slices())):
            monos = code.monomials[sl]
            n_votes = 2 ** (code.m - d)
            for j, mono in zip(range(sl.start, sl.stop), monos):
                votes = z[:, self.cosets[mono]].sum(axis=2) % 2      # (batch, n_votes)
                ones = votes.sum(axis=1)
                bit = (2 * ones > n_votes).astype(np.uint8)
                ties = 2 * ones == n_votes
                if rng is not None and ties.any():                   # break ties randomly
                    bit[ties] = rng.integers(0, 2, ties.sum())
                u_hat[:, j] = bit
            # Remove the degree-d part before moving to lower degrees.
            z ^= (u_hat[:, sl].astype(np.int64) @ code.G[sl] % 2).astype(np.uint8)
        return u_hat


# --------------------------------------------------------------------------
# Soft maximum-likelihood decoding
# --------------------------------------------------------------------------

def fast_hadamard_transform(y: np.ndarray) -> np.ndarray:
    """Unnormalised Walsh-Hadamard transform along the last axis.

    out[a] = sum_x y[x] * (-1)^(a . x), computed in O(n log n).
    """
    out = np.array(y, dtype=np.float64, copy=True)
    n = out.shape[-1]
    h = 1
    while h < n:
        out = out.reshape(*out.shape[:-1], n // (2 * h), 2, h)
        a, b = out[..., 0, :].copy(), out[..., 1, :].copy()
        out[..., 0, :], out[..., 1, :] = a + b, a - b
        out = out.reshape(*out.shape[:-3], n)
        h *= 2
    return out


def ml_decode_rm1(code: ReedMullerCode, y: np.ndarray) -> np.ndarray:
    """Exact ML decoding of RM(1, m) over the BPSK/AWGN channel.

    Codewords are a0 + a.x. Their BPSK images are +/- (-1)^(a.x), so the
    correlation of y with every codeword is +/- the Hadamard transform of y.
    The ML codeword maximises |FHT(y)[a]|, with a0 = 1 if that value is negative.
    """
    if code.r != 1:
        raise ValueError("ml_decode_rm1 needs a first-order code")
    y = np.atleast_2d(y)
    spec = fast_hadamard_transform(y)
    a = np.abs(spec).argmax(axis=1)
    a0 = spec[np.arange(len(a)), a] < 0
    u_hat = np.zeros((len(a), code.k), dtype=np.uint8)
    u_hat[:, 0] = a0
    u_hat[:, 1:] = (a[:, None] >> np.arange(code.m)) & 1
    return u_hat


class BruteForceMLDecoder:
    """Exact ML decoding by correlation with all 2^k codewords."""

    MAX_K = 16

    def __init__(self, code: ReedMullerCode):
        if code.k > self.MAX_K:
            raise ValueError(f"k = {code.k} is too large for brute-force ML")
        self.code = code
        msgs = ((np.arange(2 ** code.k)[:, None] >> np.arange(code.k)) & 1).astype(np.uint8)
        self.messages = msgs
        self.bpsk = (1 - 2 * code.encode(msgs).astype(np.float32))   # (2^k, n)

    def decode(self, y: np.ndarray, chunk: int = 256) -> np.ndarray:
        y = np.atleast_2d(y).astype(np.float32)
        out = np.empty((y.shape[0], self.code.k), dtype=np.uint8)
        for s in range(0, y.shape[0], chunk):
            corr = y[s:s + chunk] @ self.bpsk.T
            out[s:s + chunk] = self.messages[corr.argmax(axis=1)]
        return out
