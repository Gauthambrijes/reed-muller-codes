import itertools
from math import comb

import numpy as np
import pytest

from rmcodes import (
    BruteForceMLDecoder,
    MajorityLogicDecoder,
    ReedMullerCode,
    fast_hadamard_transform,
    ml_decode_rm1,
)

SMALL = [(1, 3), (1, 4), (2, 4), (1, 5), (2, 5), (3, 5)]
rng = np.random.default_rng(0)


@pytest.mark.parametrize("r,m", SMALL + [(2, 6), (3, 6)])
def test_parameters(r, m):
    code = ReedMullerCode(r, m)
    assert code.n == 2 ** m
    assert code.k == sum(comb(m, i) for i in range(r + 1))
    assert gf2_rank(code.G) == code.k


def gf2_rank(A):
    A = A.copy() % 2
    rank = 0
    for col in range(A.shape[1]):
        pivots = np.nonzero(A[rank:, col])[0]
        if len(pivots) == 0:
            continue
        p = rank + pivots[0]
        A[[rank, p]] = A[[p, rank]]
        others = np.nonzero(A[:, col])[0]
        others = others[others != rank]
        A[others] ^= A[rank]
        rank += 1
        if rank == A.shape[0]:
            break
    return rank


@pytest.mark.parametrize("r,m", [(1, 3), (1, 4), (2, 4), (1, 5)])
def test_minimum_distance(r, m):
    """Enumerate every codeword and confirm d_min = 2^(m-r)."""
    code = ReedMullerCode(r, m)
    msgs = np.array(list(itertools.product((0, 1), repeat=code.k)), dtype=np.uint8)
    weights = code.encode(msgs).sum(axis=1)
    assert weights[1:].min() == code.d_min
    assert len(np.unique(code.encode(msgs), axis=0)) == 2 ** code.k  # injective encoding


def test_known_codes():
    # RM(1,3) is the [8,4,4] extended Hamming code; RM(m-2, m) is extended Hamming in general.
    assert repr(ReedMullerCode(1, 3)) == "RM(1,3) [n=8, k=4, d=4]"
    assert repr(ReedMullerCode(2, 4)) == "RM(2,4) [n=16, k=11, d=4]"
    assert repr(ReedMullerCode(1, 5)) == "RM(1,5) [n=32, k=6, d=16]"


@pytest.mark.parametrize("r,m", SMALL + [(2, 6)])
def test_majority_corrects_up_to_t_errors(r, m):
    code = ReedMullerCode(r, m)
    dec = MajorityLogicDecoder(code)
    trials = 500
    u = rng.integers(0, 2, (trials, code.k), dtype=np.uint8)
    c = code.encode(u)
    for w in range(code.t + 1):
        e = np.zeros_like(c)
        for i in range(trials):
            e[i, rng.choice(code.n, w, replace=False)] = 1
        assert np.array_equal(dec.decode(c ^ e), u), f"failed at weight {w}"


def test_fht_matches_definition():
    y = rng.standard_normal((3, 16))
    x = np.arange(16)
    H = (-1.0) ** np.array([[bin(a & b).count("1") for b in x] for a in x])
    assert np.allclose(fast_hadamard_transform(y), y @ H.T)


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_ml_rm1_noiseless_and_matches_bruteforce(m):
    code = ReedMullerCode(1, m)
    u = rng.integers(0, 2, (300, code.k), dtype=np.uint8)
    x = 1.0 - 2.0 * code.encode(u)
    assert np.array_equal(ml_decode_rm1(code, x), u)

    y = x + 0.9 * rng.standard_normal(x.shape)
    bf = BruteForceMLDecoder(code)
    # Both are exact ML, so the chosen codewords must have equal correlation.
    c1 = code.encode(ml_decode_rm1(code, y))
    c2 = code.encode(bf.decode(y))
    corr1 = (y * (1 - 2.0 * c1)).sum(axis=1)
    corr2 = (y * (1 - 2.0 * c2)).sum(axis=1)
    assert np.allclose(corr1, corr2, atol=1e-4)


@pytest.mark.parametrize("r,m", [(2, 4), (2, 5)])
def test_bruteforce_noiseless(r, m):
    code = ReedMullerCode(r, m)
    u = rng.integers(0, 2, (200, code.k), dtype=np.uint8)
    assert np.array_equal(BruteForceMLDecoder(code).decode(1.0 - 2.0 * code.encode(u)), u)
