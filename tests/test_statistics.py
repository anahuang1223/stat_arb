import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from src.statistical_tests import benjamini_hochberg, half_life, hedge_ratio


# Check that BH results match the statsmodels implementation.
def test_bh_vs_statsmodels():
    rng = np.random.default_rng(0)
    p = pd.Series(rng.uniform(0, 0.2, 40))
    ours = benjamini_hochberg(p, alpha=0.05)
    reject, adj, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
    np.testing.assert_allclose(ours["pvalue_adj"], adj)
    np.testing.assert_array_equal(ours["significant"], reject)


# Check that BH produces the expected significance decisions and adj pval.
def test_bh_hand_example():
    p = pd.Series([0.04, 0.001, 0.5, 0.03, 0.01], index=list("abcde"))
    out = benjamini_hochberg(p, alpha=0.05)
    assert out["significant"].to_dict() == {
        "a": True, "b": True, "c": False, "d": True, "e": True,
    }
    assert out.loc["a", "pvalue_adj"] == pytest.approx(0.05)


# Check that BH accounts for the number of tests and is stricter than the raw pvals.
def test_bh_stricter_than_raw():
    out = benjamini_hochberg(pd.Series([0.04] + [0.9] * 19), alpha=0.05)
    assert out.loc[0, "pvalue_adj"] == pytest.approx(0.8)
    assert not out["significant"].any()


# Check that all pairs remain significant when every pval is sufficiently small.
def test_bh_many_small():
    out = benjamini_hochberg(pd.Series([0.04] * 20), alpha=0.05)
    assert out["significant"].all()


# Check that invalid pvals (NaN) raise a ValueError.
def test_bh_invalid():
    with pytest.raises(ValueError):
        benjamini_hochberg(pd.Series([0.1, np.nan]))


# Check that an empty input returns an empty DataFrame.
def test_bh_empty():
    assert benjamini_hochberg(pd.Series([], dtype=float)).empty


# Generate an AR(1) spread: current spread = phi * previous spread + random noise.
# In half_life(), the mean-reversion slope is approx phi - 1.
def _ar1(phi, n_obs, noise_sd, rng):
    vals = np.zeros(n_obs)
    for t in range(1, n_obs):
        vals[t] = phi * vals[t - 1] + rng.normal(0, noise_sd)
    return vals


# Generate a cointegrated pair: X is a random walk and Y = 2 + beta * X + an AR(1) spread.
def _coint_pair(n_obs=1000, beta=1.5, phi=0.9, seed=1):
    rng = np.random.default_rng(seed)
    x_series = pd.Series(np.cumsum(rng.normal(0, 1, n_obs)), name="X")
    y_series = pd.Series(2.0 + beta * x_series + _ar1(phi, n_obs, 0.5, rng), name="Y")
    return y_series, x_series


# Check that hedge_ratio() recovers the true beta within 0.05.
def test_hedge_ratio():
    y_series, x_series = _coint_pair(beta=1.5)
    _, beta = hedge_ratio(y_series, x_series)
    assert beta == pytest.approx(1.5, abs=0.05)


# Check that half_life() is within 10% of the theoretical AR(1) half-life.
def test_half_life_ar1():
    phi = 0.95
    spread = pd.Series(_ar1(phi, 20000, 1.0, np.random.default_rng(2)))
    assert half_life(spread) == pytest.approx(-np.log(2) / np.log(phi), rel=0.1)


# Check that a random walk with no mean reversion has a very long half-life.
def test_half_life_random_walk():
    rng = np.random.default_rng(3)
    assert half_life(pd.Series(np.cumsum(rng.normal(size=2000)))) > 100


# Check that a spread alternating between +1 and -1 returns NaN since the slope is <= -1.
def test_half_life_overshoot():
    alternating = pd.Series([1.0, -1.0] * 50)
    assert np.isnan(half_life(alternating))
