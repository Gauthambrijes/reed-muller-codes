import itertools

import numpy as np
import pytest

from rmcodes import (
    BruteForceMLDecoder,
    ReedMullerCode,
    RPADecoder,
    boxplus,
    boxplus_minsum,
    make_quantizer,
    min_weight_count,
    union_bound_bler,
    weight_distribution,
)
from rmcodes.bounds import _enumerate_distribution

rng = np.random.default_rng(1)


# ---------------------------------------------------------------- box-plus

def test_boxplus_is_tanh_rule():
    a, b = rng.normal(0, 4, (2, 10_000))
    assert np.allclose(boxplus(a, b), 2 * np.arctanh(np.tanh(a / 2) * np.tanh(b / 2)), atol=1e-9)


def test_minsum_gap_at_most_ln2_and_overestimates():
    a, b = rng.normal(0, 8, (2, 100_000))
    exact, ms = boxplus(a, b), boxplus_minsum(a, b)
    assert np.all(np.abs(ms - exact) <= np.log(2) + 1e-12)
    assert np.all(np.abs(ms) >= np.abs(exact) - 1e-12)
    assert np.all(np.sign(ms) == np.sign(exact))


def test_quantizer_is_odd():
    q = make_quantizer(7, 8)
    x = rng.normal(0, 200, 10_000)
    assert np.array_equal(q(-x), -q(x))
    assert q(1e9) == (2 ** 15 - 1) / 256


# ---------------------------------------------------------------- weights

@pytest.mark.parametrize("r,m", [(1, 4), (1, 6), (2, 4), (2, 5), (3, 4), (2, 6)])
def test_min_weight_count(r, m):
    dist = _enumerate_distribution(ReedMullerCode(r, m))
    d = 2 ** (m - r)
    assert min(w for w in dist if w > 0) == d
    assert dist[d] == min_weight_count(r, m)


@pytest.mark.parametrize("m", [3, 4, 5, 6])
def test_sloane_berlekamp_matches_enumeration(m):
    assert weight_distribution(2, m) == _enumerate_distribution(ReedMullerCode(2, m))


def test_rm28_distribution():
    dist = weight_distribution(2, 8)
    assert sum(dist.values()) == 2 ** 37
    assert dist[64] == min_weight_count(2, 8) == 43180
    assert all(dist[w] == dist[256 - w] for w in dist)  # contains the all-one word


def test_union_bound_tightens_to_truncated():
    full = union_bound_bler(2, 8, [2.0, 4.0, 6.0])
    trunc = union_bound_bler(2, 8, [2.0, 4.0, 6.0], truncated=True)
    assert np.all(full >= trunc)
    assert full[-1] / trunc[-1] < 1.01


# ---------------------------------------------------------------- RPA

def _llr(code, u, ebn0_db, rng):
    sigma = np.sqrt(1 / (2 * code.rate * 10 ** (ebn0_db / 10)))
    y = 1 - 2.0 * code.encode(u) + sigma * rng.standard_normal((len(u), code.n))
    return 2 * y / sigma ** 2


@pytest.mark.parametrize("r,m", [(2, 4), (2, 5), (2, 6), (3, 5)])
@pytest.mark.parametrize("min_sum", [False, True])
def test_rpa_noiseless(r, m, min_sum):
    code = ReedMullerCode(r, m)
    u = rng.integers(0, 2, (40, code.k), dtype=np.uint8)
    llr = 4.0 * (1 - 2.0 * code.encode(u))
    assert np.array_equal(RPADecoder(code, min_sum=min_sum).decode(llr), u)


@pytest.mark.parametrize("r,m", [(2, 5), (3, 5)])
@pytest.mark.parametrize("min_sum", [False, True])
def test_rpa_codeword_symmetry(r, m, min_sum):
    """Proposition 4: decoding is equivariant under adding a codeword."""
    code = ReedMullerCode(r, m)
    dec = RPADecoder(code, min_sum=min_sum)
    L = rng.normal(2.0, 2.0, (30, code.n))                 # noisy all-zero word
    c = code.encode(rng.integers(0, 2, (30, code.k), dtype=np.uint8))
    lhs = dec.decode_codeword(L * (1 - 2.0 * c))
    assert np.array_equal(lhs, dec.decode_codeword(L) ^ c)


def test_rpa_close_to_ml_on_rm25():
    code = ReedMullerCode(2, 5)
    u = rng.integers(0, 2, (3000, code.k), dtype=np.uint8)
    llr = _llr(code, u, 3.0, rng)
    ml_err = (BruteForceMLDecoder(code).decode(llr) != u).any(axis=1).sum()
    rpa_err = (RPADecoder(code).decode(llr) != u).any(axis=1).sum()
    assert ml_err > 10
    assert rpa_err <= 1.5 * ml_err + 5
