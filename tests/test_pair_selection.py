import numpy as np
import pandas as pd

from src.pair_selection import candidate_pairs, select_pairs


# Generate log prices where A/B are cointegrated with an AR(1) spread and C/D are
# independent random walks.
def _log_prices(n_obs=1500, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    x_vals = np.cumsum(rng.normal(0, 0.01, n_obs))
    spread = np.zeros(n_obs)
    for t in range(1, n_obs):
        spread[t] = phi * spread[t - 1] + rng.normal(0, 0.005)
    return pd.DataFrame({
        "A": 0.5 + 1.2 * x_vals + spread,
        "B": x_vals,
        "C": np.cumsum(rng.normal(0, 0.01, n_obs)),
        "D": np.cumsum(rng.normal(0, 0.01, n_obs)),
    })

# Check that unique pairs are generated within each sector in sorted order.
def test_candidate_pairs_within_sector():
    universe = {"s1": ["B", "A", "C"], "s2": ["D", "E"]}
    pairs = candidate_pairs(universe, ["A", "B", "C", "D", "E"])
    assert pairs == [
        ("s1", "A", "B"), ("s1", "A", "C"), ("s1", "B", "C"), ("s2", "D", "E"),
    ]


# Check that tickers with missing data are excluded.
def test_candidate_pairs_skips_missing():
    universe = {"s1": ["A", "B", "C"]}
    assert candidate_pairs(universe, ["A", "C"]) == [("s1", "A", "C")]

# Check that the pipeline selects the expected cointegrated pair (A/B)
# and rejects C/D based on the simulated data.
def test_select_pairs_finds_coint():
    universe = {"s1": ["A", "B"], "s2": ["C", "D"]}
    out = select_pairs(_log_prices(), universe)
    assert out.loc["A/B", "selected"]
    assert out.loc["A/B", "beta"] > 0
    assert not out.loc["C/D", "selected"]

# Check that a BH-significant pair is still rejected if its half-life
# exceeds the allowed maximum.
def test_select_pairs_half_life_filter():
    universe = {"s1": ["A", "B"]}
    out = select_pairs(_log_prices(), universe, max_half_life=2)
    assert out.loc["A/B", "significant"]
    assert not out.loc["A/B", "selected"]
