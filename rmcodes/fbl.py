"""Finite-blocklength bounds for the binary-input AWGN (BI-AWGN) channel.

Channel: BPSK x in {+1, -1}, Y = x + sigma Z, uniform inputs. Everything is in
nats. With LLR l = 2y / sigma^2, the information density of one use is

    i(x; y) = ln 2 - ln(1 + exp(-x l)),

and by symmetry its law does not depend on x, so take x = +1. For a block of
n uses let S = sum of n i.i.d. copies, and M = 2^k messages.

* Normal approximation (Polyanskiy-Poor-Verdu 2010):
      ln M ~ n C - sqrt(n V) Q^{-1}(eps) + (1/2) ln n,
  with C = E[i] and V = Var[i] (computed by Gauss-Hermite quadrature).

* DT achievability bound (PPV Theorem 17), random coding:
      eps <= E[ exp( -(S - ln((M - 1) / 2))^+ ) ].

* Meta-converse (PPV Theorem 27) with the uniform-input output distribution
  as auxiliary Q. For a symmetric channel the log-likelihood ratio
  ln dP_{Y|X=x}/dQ_Y has the law of S for every codeword x, so any code with
  average error eps satisfies beta_{1-eps}(P_S, Q_S) <= 1/M, where under Q the
  statistic has density e^{-s} relative to P. The Neyman-Pearson beta is
  computed exactly on a lattice (below).

Numerics, kept rigorous: i is increasing in y, so its law is discretised on a
lattice of spacing h by integrating the Gaussian density between the lattice
preimages. "Floor" assigns each bin its lower edge (S- <= S path by path) and
"ceil" its upper edge (S+ >= S); the n-fold convolutions are done by FFT.
The DT bound is evaluated with S- (the DT integrand is non-increasing in S),
plus n * P(i < t_min) for the truncated lower tail. The converse uses S+ with
a slack of n h:  beta_alpha >= sum_{s > tau + n h} e^{-s} P(S+ = s) - n p_t e^{-tau - n h},
which is a valid lower bound on the true beta, hence a valid lower bound on
eps. With h = 5e-4 the slack is at most 0.128 nats for n = 256.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import log_ndtr, ndtr

LN2 = np.log(2.0)


def sigma_from_ebn0(ebn0_db, rate):
    return np.sqrt(1.0 / (2.0 * rate * 10 ** (np.asarray(ebn0_db, dtype=float) / 10)))


def _info_density(y, sigma, s=1.0):
    """i_s(+1; y) = ln 2 - ln(1 + exp(-s l)); s = 1 is the information density."""
    l = 2.0 * y / sigma ** 2
    return LN2 - np.logaddexp(0.0, -s * l)


@lru_cache(maxsize=None)
def capacity_dispersion(sigma: float):
    """C and V (nats) of the BI-AWGN channel with uniform inputs."""
    z, w = np.polynomial.hermite_e.hermegauss(150)  # weight exp(-z^2/2)
    w = w / w.sum()
    i = _info_density(1.0 + sigma * z, sigma)
    C = float(np.sum(w * i))
    V = float(np.sum(w * (i - C) ** 2))
    return C, V


def normal_approx_eps(n, k, sigma):
    from scipy.stats import norm
    C, V = capacity_dispersion(float(sigma))
    return float(norm.sf((n * C - k * LN2 + 0.5 * np.log(n)) / np.sqrt(n * V)))


# ---------------------------------------------------------------- lattice law of S

@dataclass
class LatticeLaw:
    offset: float      # value of lattice index 0
    h: float
    pmf: np.ndarray    # (possibly defective) pmf on offset + j h
    trunc: float       # per-letter probability of the truncated lower tail


def _letter_pmf(sigma, h, s=1.0, z_lo=-12.0):
    """Per-letter floor and ceil pmfs of i_s on the lattice t_min + j h."""
    y_lo = 1.0 + sigma * z_lo
    t_min = np.floor(_info_density(y_lo, sigma, s) / h) * h
    J = int(np.ceil((LN2 - t_min) / h)) + 1
    edges = t_min + h * np.arange(J + 1)                    # bins [edges[j], edges[j+1])
    inside = edges < LN2
    # preimage y(t) = -(sigma^2 / (2 s)) ln(2 e^{-t} - 1), for t < ln 2
    ye = np.full(edges.shape, np.inf)
    ye[inside] = -(sigma ** 2 / (2 * s)) * np.log(2.0 * np.exp(-edges[inside]) - 1.0)
    z = (ye - 1.0) / sigma
    # Gaussian mass between consecutive preimages, accurate in both tails.
    a, b = z[:-1], z[1:]
    masses = np.where(b <= 0, ndtr(b) - ndtr(a),
                      np.where(a >= 0, ndtr(-a) - ndtr(-b), (0.5 - ndtr(a)) + (0.5 - ndtr(-b))))
    masses = np.maximum(masses, 0.0)
    trunc = float(np.exp(log_ndtr(z[0])))
    floor_pmf = masses.copy()                      # defective: truncated mass dropped
    ceil_pmf = np.zeros(J + 1)
    ceil_pmf[1:] = masses                          # bin j -> its upper edge t_min + (j+1) h
    ceil_pmf[0] = trunc                            # truncated tail -> t_min (>= true value)
    return t_min, floor_pmf, ceil_pmf, trunc


def _nfold(pmf, n):
    L = (len(pmf) - 1) * n + 1
    N = next_fast_len(L, real=True)
    out = irfft(rfft(pmf, N) ** n, N)[:L]
    return np.maximum(out, 0.0)


def block_laws(n, sigma, h=5e-4, s=1.0, ceil=True):
    t_min, fl, ce, trunc = _letter_pmf(sigma, h, s)
    S_floor = LatticeLaw(n * t_min, h, _nfold(fl, n), trunc)
    S_ceil = LatticeLaw(n * t_min, h, _nfold(ce, n), trunc) if ceil else None
    return S_floor, S_ceil


# ---------------------------------------------------------------- bounds

def dt_bound(n, k, sigma, h=5e-4, laws=None):
    """Upper bound on the error probability of the best (n, 2^k) code."""
    S_floor, _ = laws or block_laws(n, sigma, h)
    s = S_floor.offset + S_floor.h * np.arange(len(S_floor.pmf))
    gamma = np.log(2.0 ** k - 1.0) - LN2
    f = np.exp(-np.maximum(s - gamma, 0.0))
    return float(min(1.0, np.sum(S_floor.pmf * f) + n * S_floor.trunc))


def rcus_bound(n, k, sigma, h=1e-3, s_range=(0.2, 1.0), iters=12):
    """RCU_s achievability bound (Martinez & Guillen i Fabregas), minimised over s.

    For i.i.d. uniform random codes, by Markov's inequality on the pairwise
    error event, eps <= E[ exp( -(i_s(X^n; Y^n) - ln(M - 1))^+ ) ] for every
    s > 0, with i_s(+1; y) = ln 2 - ln(1 + e^{-s l}) per letter. Evaluated with
    the floor lattice (rigorous for each s); s by golden-section search.
    """
    lnM1 = np.log(2.0 ** k - 1.0)

    def bound(sv):
        S, _ = block_laws(n, sigma, h, s=sv, ceil=False)
        t = S.offset + S.h * np.arange(len(S.pmf))
        return float(min(1.0, np.sum(S.pmf * np.exp(-np.maximum(t - lnM1, 0.0))) + n * S.trunc))

    g = (np.sqrt(5) - 1) / 2
    a, b = s_range
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = bound(c), bound(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = bound(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = bound(d)
    return min(fc, fd, bound(1.0))


def metaconverse_bound(n, k, sigma, h=5e-4, laws=None):
    """Lower bound on the error probability of any (n, 2^k) code."""
    _, S = laws or block_laws(n, sigma, h)
    s = S.offset + S.h * np.arange(len(S.pmf))
    p = S.pmf
    slack = n * S.h
    lnM = k * LN2
    # tail[j] = P(S+ > s_j); q_tail[j] = sum_{s_i > s_j} e^{-s_i} p_i  (Q-mass)
    tail = np.concatenate([np.cumsum(p[::-1])[::-1][1:], [0.0]])
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        # Dropping terms only lowers beta_low, so clipping keeps the bound valid.
        q = np.where((p > 0) & (s > -700), np.exp(-np.maximum(s, -700)) * p, 0.0)
    q_tail = np.concatenate([np.cumsum(q[::-1])[::-1][1:], [0.0]])
    # smallest lattice index with the same tail probability (tau+ for alpha = tail[j])
    first = np.searchsorted(-tail, -tail, side="left")
    shift = int(np.ceil(slack / S.h))
    idx = np.minimum(first + shift, len(s) - 1)
    with np.errstate(over="ignore"):
        beta_low = q_tail[idx] - n * S.trunc * np.exp(np.minimum(-s[first] - slack, 700))
    ok = np.nonzero(beta_low > np.exp(-lnM))[0]
    if len(ok) == 0:
        return 0.0
    j = ok.max()
    return float(1.0 - tail[j])   # P(S+ <= s_j)


def required_ebn0(n, k, eps, which, lo=-2.0, hi=12.0, tol=0.002, h=5e-4):
    """Smallest Eb/N0 (dB) at which the given bound reaches eps (bisection)."""
    rate = k / n

    def err(eb):
        sig = float(sigma_from_ebn0(eb, rate))
        if which == "normal":
            return normal_approx_eps(n, k, sig)
        if which == "rcus":
            return rcus_bound(n, k, sig)
        laws = block_laws(n, sig, h)
        return dt_bound(n, k, sig, h, laws) if which == "dt" else metaconverse_bound(n, k, sig, h, laws)

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if err(mid) > eps:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eps_curves(n, k, ebn0_db, h=5e-4):
    """eps of the converse, normal approximation, RCU_s and DT bounds at each Eb/N0."""
    out = {"converse": [], "normal": [], "rcus": [], "dt": []}
    for eb in np.atleast_1d(ebn0_db):
        sig = float(sigma_from_ebn0(eb, k / n))
        laws = block_laws(n, sig, h)
        out["converse"].append(metaconverse_bound(n, k, sig, h, laws))
        out["normal"].append(normal_approx_eps(n, k, sig))
        out["rcus"].append(rcus_bound(n, k, sig))
        out["dt"].append(dt_bound(n, k, sig, h, laws))
    return {key: np.array(v) for key, v in out.items()}


__all__ = ["capacity_dispersion", "normal_approx_eps", "dt_bound", "rcus_bound", "metaconverse_bound",
           "required_ebn0", "eps_curves", "sigma_from_ebn0"]


