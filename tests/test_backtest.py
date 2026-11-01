import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    backtest_pair, pair_weights, portfolio_returns, walk_forward, walk_forward_windows,
)
from src.portfolio import equal_weight_returns


# Check that weights have gross exposure 1 and the X leg is hedged by beta.
def test_pair_weights_scaling():
    weights = pair_weights(pd.Series([0.0, 1.0, -1.0]), beta=0.5)
    assert weights["weight_y"].tolist() == pytest.approx([0.0, 2 / 3, -2 / 3])
    assert weights["weight_x"].tolist() == pytest.approx([0.0, -1 / 3, 1 / 3])
    assert weights.abs().sum(axis=1).iloc[1:].tolist() == pytest.approx([1.0, 1.0])


# Check that a position earns nothing on the day it is opened and earns the next day's return.
def test_portfolio_returns_one_day_lag():
    prices = pd.DataFrame({"Y": [100.0, 110.0, 121.0], "X": [100.0, 100.0, 100.0]})
    weights = pd.DataFrame({"weight_y": [0.0, 1.0, 1.0], "weight_x": [0.0, 0.0, 0.0]})
    returns = portfolio_returns(prices, weights)
    # Y rises 10% on day 1, but the position is only decided at day 1's close.
    assert returns["gross_return"].tolist() == pytest.approx([0.0, 0.0, 0.10])


# Check that costs are charged on entry and exit only, in proportion to the weight traded.
def test_portfolio_returns_costs():
    prices = pd.DataFrame({"Y": [100.0] * 4, "X": [100.0] * 4})
    weights = pd.DataFrame({"weight_y": [0.5, 0.5, 0.0, 0.0], "weight_x": [-0.5, -0.5, 0.0, 0.0]})
    returns = portfolio_returns(prices, weights, cost_bps=10)
    assert returns["turnover"].tolist() == pytest.approx([1.0, 0.0, 1.0, 0.0])
    assert returns["net_return"].tolist() == pytest.approx([-0.001, 0.0, -0.001, 0.0])


# Check that the backtest starts at start with a valid z-score and no position carried in.
def test_backtest_pair_start():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=200)
    x_vals = np.cumsum(rng.normal(0, 0.01, 200))
    log_prices = pd.DataFrame(
        {"A": 0.1 + 1.5 * x_vals + rng.normal(0, 0.02, 200), "B": x_vals}, index=dates
    )
    result = backtest_pair(
        log_prices, "A", "B", alpha=0.1, beta=1.5, spread_std=0.02, start=str(dates[100].date())
    )
    assert result.index[0] == dates[100]
    assert result["zscore"].notna().all()
    assert result["position"].isin([-1, 0, 1]).all()
    # first day has no held position, so it cannot earn a return.
    assert result["gross_return"].iloc[0] == 0.0


# Check that backtest_pair stops at end and flattens the position on the last day.
def test_backtest_pair_end_flattens():
    dates = pd.bdate_range("2020-01-01", periods=60)
    # spread oscillates, then drops to -2.5 formation stds so a long position opens.
    spread = np.r_[np.tile([0.1, -0.1], 20), np.full(20, -0.25)]
    log_prices = pd.DataFrame({"A": spread, "B": np.zeros(60)}, index=dates)
    result = backtest_pair(
        log_prices, "A", "B", alpha=0.0, beta=1.0, spread_std=0.1,
        start=dates[40], end=dates[50],
    )
    assert result.index[-1] == dates[50]
    assert result["position"].iloc[-2] == 1
    assert result["position"].iloc[-1] == 0


# Generate log prices where A/B are cointegrated and C is an independent random walk.
def _walk_forward_prices(n_obs=900, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_obs)
    x_vals = np.cumsum(rng.normal(0, 0.01, n_obs))
    spread = np.zeros(n_obs)
    for t in range(1, n_obs):
        spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 0.01)
    return pd.DataFrame({
        "A": 0.2 + 1.1 * x_vals + spread,
        "B": x_vals,
        "C": np.cumsum(rng.normal(0, 0.01, n_obs)),
    }, index=dates)


# Check that each trading period starts after its formation period and trading periods do not
# overlap or leave gaps.
def test_walk_forward_windows_no_overlap():
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    windows = walk_forward_windows(dates, formation_months=12, trading_months=6)
    assert (windows["formation_end"] < windows["trading_start"]).all()
    assert (windows["trading_start"].iloc[1:].to_numpy()
            > windows["trading_end"].iloc[:-1].to_numpy()).all()
    # first trading day of each window follows the last trading day of the previous one.
    for prev_end, next_start in zip(windows["trading_end"].iloc[:-1], windows["trading_start"].iloc[1:]):
        assert dates.get_loc(next_start) == dates.get_loc(prev_end) + 1
    assert windows["trading_start"].iloc[0] == pd.Timestamp("2016-01-01")


# Check that walk_forward selects the cointegrated pair and only trades inside trading periods.
def test_walk_forward_selects_and_trades_out_of_sample():
    log_prices = _walk_forward_prices()
    pair_results, selections = walk_forward(
        log_prices, {"s": ["A", "B", "C"]}, formation_months=12, trading_months=6
    )
    assert "A/B" in set(selections.index)
    windows = walk_forward_windows(log_prices.index, 12, 6)
    for window_id, rows in pair_results.groupby("window"):
        window = windows.loc[window_id]
        assert rows["date"].min() == window["trading_start"]
        assert rows["date"].max() == window["trading_end"]


# Check that equal weighting averages the pairs each day and fills days with no pairs with 0.
def test_equal_weight_returns():
    dates = pd.bdate_range("2020-01-01", periods=3)
    pair_results = pd.DataFrame({
        "date": [dates[0], dates[0], dates[1]],
        "pair": ["A/B", "C/D", "A/B"],
        "net_return": [0.02, 0.0, -0.01],
    })
    portfolio = equal_weight_returns(pair_results, dates)
    assert portfolio.tolist() == pytest.approx([0.01, -0.01, 0.0])


# Check that windows trading on or before validation_end are validation and later ones holdout.
def test_walk_forward_windows_period():
    dates = pd.bdate_range("2015-01-01", "2018-12-31")
    windows = walk_forward_windows(dates, 12, 6, validation_end="2016-12-31")
    assert windows["period"].tolist() == ["validation"] * 2 + ["holdout"] * 4
    assert (windows.loc[windows["period"] == "holdout", "trading_start"] > "2016-12-31").all()


# Check that max_pairs caps how many pairs are traded in each window.
def test_walk_forward_max_pairs():
    log_prices = _walk_forward_prices()
    # D is a second cointegrated partner for B, so A/B, A/D, and B/D can all be selected.
    log_prices["D"] = log_prices["B"] * 0.9 + np.random.default_rng(1).normal(0, 0.01, len(log_prices))
    universe = {"s": ["A", "B", "C", "D"]}
    _, uncapped = walk_forward(log_prices, universe, 12, 6)
    _, capped = walk_forward(log_prices, universe, 12, 6, max_pairs=1)
    assert uncapped.groupby("window").size().max() > 1
    assert capped.groupby("window").size().max() == 1

