import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_return, extract_trades, max_drawdown, performance_summary, sharpe_ratio,
)


# Check that the Sharpe ratio matches a hand calculation from the daily mean and std.
def test_sharpe_ratio_values():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0])
    expected = returns.mean() / returns.std() * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)


# Check that a flat return series gives NaN instead of dividing by zero.
def test_sharpe_ratio_flat_returns():
    assert np.isnan(sharpe_ratio(pd.Series([0.0, 0.0, 0.0])))


# Check that a year of returns annualizes to its total return.
def test_annualized_return_one_year():
    returns = pd.Series([0.0] * 251 + [0.10])
    assert annualized_return(returns) == pytest.approx(0.10)


# Check that the drawdown is measured from the running peak, not the start.
def test_max_drawdown_from_peak():
    # equity goes 1.1, 0.55, 0.66, so the worst fall is from 1.1 to 0.55.
    assert max_drawdown(pd.Series([0.10, -0.50, 0.20])) == pytest.approx(-0.50)


# Check that a loss on the first day counts as a drawdown from the starting capital.
def test_max_drawdown_first_day_loss():
    assert max_drawdown(pd.Series([-0.10, 0.05])) == pytest.approx(-0.10)


# Check that trades run from entry through the flat exit day and compound their net returns.
def test_extract_trades():
    dates = pd.bdate_range("2020-01-01", periods=7)
    pair_results = pd.DataFrame({
        "date": dates,
        "window": 0,
        "period": "validation",
        "pair": "A/B",
        "position": [0, 1, 1, 0, 0, -1, 0],
        "net_return": [0.0, -0.001, 0.02, 0.01, 0.0, -0.001, -0.03],
    })
    trades = extract_trades(pair_results)
    assert trades["entry_date"].tolist() == [dates[1], dates[5]]
    assert trades["exit_date"].tolist() == [dates[3], dates[6]]
    assert trades["direction"].tolist() == [1, -1]
    assert trades["holding_days"].tolist() == [2, 1]
    assert trades["trade_return"].iloc[0] == pytest.approx(0.999 * 1.02 * 1.01 - 1)
    assert trades["trade_return"].iloc[1] == pytest.approx(0.999 * 0.97 - 1)


# Check that the same pair in two windows gives separate trades.
def test_extract_trades_by_window():
    dates = pd.bdate_range("2020-01-01", periods=3)
    window_rows = pd.DataFrame({
        "date": dates, "pair": "A/B", "position": [1, 1, 0], "net_return": [0.0, 0.01, 0.01],
    })
    pair_results = pd.concat([window_rows.assign(window=0), window_rows.assign(window=1)])
    assert extract_trades(pair_results)["window"].tolist() == [0, 1]


# Check that the summary reports the hit rate and trade count, and handles no trades.
def test_performance_summary_trades():
    returns = pd.Series([0.01, -0.02, 0.015])
    trades = pd.DataFrame({"trade_return": [0.02, -0.01, 0.005, 0.0], "holding_days": [3, 5, 2, 6]})
    summary = performance_summary(returns, trades)
    assert summary["n_trades"] == 4
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["avg_holding_days"] == pytest.approx(4.0)

    empty = performance_summary(returns, trades.iloc[:0])
    assert empty["n_trades"] == 0
    assert np.isnan(empty["hit_rate"])
