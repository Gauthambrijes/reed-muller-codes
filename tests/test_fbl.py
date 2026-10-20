import numpy as np
import pytest

from rmcodes import fbl


def test_capacity_at_rate_half_shannon_limit():
    # BI-AWGN with uniform inputs: C = 1/2 bit at Eb/N0 = 0.187 dB for R = 1/2.
    sigma = float(fbl.sigma_from_ebn0(0.187, 0.5))
    C, V = fbl.capacity_dispersion(sigma)
    assert abs(C / np.log(2) - 0.5) < 2e-4
    rng = np.random.default_rng(0)
    y = 1 + sigma * rng.standard_normal(1_000_000)
    i = np.log(2) - np.logaddexp(0, -2 * y / sigma ** 2)
    assert abs(i.var() - V) < 0.01


def test_lattice_bounds_match_monte_carlo():
    n, k, eb = 64, 32, 3.0
    sigma = float(fbl.sigma_from_ebn0(eb, k / n))
    rng = np.random.default_rng(1)
    y = 1 + sigma * rng.standard_normal((400_000, n))
    l = 2 * y / sigma ** 2
    S = (np.log(2) - np.logaddexp(0, -l)).sum(axis=1)
    dt_mc = np.mean(np.exp(-np.maximum(S - (np.log(2.0 ** k - 1) - np.log(2)), 0)))
    dt = fbl.dt_bound(n, k, sigma)
    assert dt >= dt_mc * 0.97 and dt <= dt_mc * 1.15   # floor lattice: slightly conservative
    Ss = (np.log(2) - np.logaddexp(0, -0.7 * l)).sum(axis=1)
    rcus_mc = np.mean(np.exp(-np.maximum(Ss - np.log(2.0 ** k - 1), 0)))
    assert fbl.rcus_bound(n, k, sigma) <= rcus_mc * 1.15  # minimised over s, so <= value at s = 0.7


@pytest.mark.parametrize("n,k", [(64, 32), (128, 29)])
def test_bound_ordering(n, k):
    eb = 2.5
    e = fbl.eps_curves(n, k, eb)
    assert e["converse"][0] <= e["normal"][0] <= e["rcus"][0] <= e["dt"][0]
